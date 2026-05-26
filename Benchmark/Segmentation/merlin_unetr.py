import torch
import torch.nn as nn
from merlin import Merlin


class MerlinEncoder(nn.Module):
    def __init__(self, freeze: bool = True):
        super().__init__()
        merlin_model = Merlin(ImageEmbedding=False)
        self.i3resnet = merlin_model.model.encode_image.i3_resnet
        self.i3resnet.return_skips = True

        if freeze:
            for p in self.i3resnet.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: (B, 1, H, W, D)

        由于 I3ResNet stem 的不对称下采样：
          conv1: H/W stride=2, D stride=1
          maxpool: H/W stride=2, D stride=2
        各 skip 的实际 shape（以输入96×96×96为例）:
          c1(layer1): (B,  256, H/4,  W/4,  D/2)  = (B,256,24,24,48)
          c2(layer2): (B,  512, H/8,  W/8,  D/4)  = (B,512,12,12,24)
          c3(layer3): (B, 1024, H/16, W/16, D/8)  = (B,1024,6,6,12)
          c4(layer4): (B, 2048, H/32, W/32, D/16) = (B,2048,3,3,6)
        """
        _, _, skips = self.i3resnet(x)
        return skips[2], skips[3], skips[4], skips[5]


class AsymDeconvBlock(nn.Module):
    """
    不对称上采样：H/W 方向 ×stride_hw，D 方向 ×stride_d
    用于补偿 Merlin encoder 中 H/W 和 D 下采样倍数不一致的问题
    """
    def __init__(self, in_ch, out_ch, stride_hw=2, stride_d=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(
                in_ch, out_ch,
                kernel_size=(stride_hw, stride_hw, stride_d),
                stride=(stride_hw, stride_hw, stride_d),
                bias=False,
            ),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNETRDecoder(nn.Module):
    """
    针对 Merlin encoder 不对称下采样的 decoder。

    encoder skip 的下采样倍数：
        层      H/W    D
        layer1  /4     /2
        layer2  /8     /4
        layer3  /16    /8
        layer4  /32    /16

    decoder 每步需要的上采样倍数（从深到浅）：
        c4→c3: H/W×2, D×2   → 对称 stride=2
        c3→c2: H/W×2, D×2   → 对称 stride=2
        c2→c1: H/W×2, D×2   → 对称 stride=2
        c1→/2: H/W×2, D×1   → 不对称！H/W需×2，D已是/2不需再×2
        /2→/1: H/W×2, D×2   → 对称 stride=2，恢复到原始分辨率
    """
    def __init__(self, num_classes,
                 enc_chs=(256, 512, 1024, 2048),
                 dec_chs=(256, 128, 64, 32)):
        super().__init__()
        c1, c2, c3, c4 = enc_chs
        d3, d2, d1, d0 = dec_chs

        # c4(/32hw,/16d) → up → cat c3(/16hw,/8d)
        # 需要 H/W×2, D×2 → 对称
        self.up3   = AsymDeconvBlock(c4, d3, stride_hw=2, stride_d=2)
        self.conv3 = ConvBlock(d3 + c3, d3)

        # → up → cat c2(/8hw,/4d)
        # 需要 H/W×2, D×2 → 对称
        self.up2   = AsymDeconvBlock(d3, d2, stride_hw=2, stride_d=2)
        self.conv2 = ConvBlock(d2 + c2, d2)

        # → up → cat c1(/4hw,/2d)
        # 需要 H/W×2, D×2 → 对称
        self.up1   = AsymDeconvBlock(d2, d1, stride_hw=2, stride_d=2)
        self.conv1 = ConvBlock(d1 + c1, d1)

        # c1 此时是 (/4hw, /2d)
        # 回到原始分辨率需要：H/W×4, D×2
        # 分两步：先 H/W×2,D×1；再 H/W×2,D×2
        self.up0a   = AsymDeconvBlock(d1, d0, stride_hw=2, stride_d=1)  # /4hw→/2hw, /2d不变
        self.conv0a = ConvBlock(d0, d0)
        self.up0b   = AsymDeconvBlock(d0, d0, stride_hw=2, stride_d=2)  # /2hw→/1hw, /2d→/1d
        self.conv0b = ConvBlock(d0, d0)

        self.seg_head = nn.Conv3d(d0, num_classes, kernel_size=1)

    def forward(self, c1, c2, c3, c4):
        x = self.conv3(torch.cat([self.up3(c4), c3], dim=1))
        x = self.conv2(torch.cat([self.up2(x),  c2], dim=1))
        x = self.conv1(torch.cat([self.up1(x),  c1], dim=1))
        x = self.conv0a(self.up0a(x))
        x = self.conv0b(self.up0b(x))
        return self.seg_head(x)


class MerlinUNETR(nn.Module):
    """
    输入:  (B, 1, H, W, D)，H==W==D 时输出与输入等分辨率
    输出:  (B, num_classes, H, W, D)
    """
    def __init__(self, num_classes: int, freeze_encoder: bool = True,
                 dec_chs=(256, 128, 64, 32)):
        super().__init__()
        self.encoder = MerlinEncoder(freeze=freeze_encoder)
        self.decoder = UNETRDecoder(
            num_classes=num_classes,
            enc_chs=(256, 512, 1024, 2048),
            dec_chs=dec_chs,
        )

    def forward(self, x):
        c1, c2, c3, c4 = self.encoder(x)
        return self.decoder(c1, c2, c3, c4)