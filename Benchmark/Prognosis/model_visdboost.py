import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial


# ============================================================
# 官方 resnet.py 原样（lavis/utils/resnet.py）
# ============================================================

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes, out_planes,
        kernel_size=3, dilation=dilation,
        stride=stride, padding=dilation, bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1),
        out.size(2), out.size(3), out.size(4),
        dtype=out.dtype)
    if not no_cuda and out.is_cuda:
        zero_pads = zero_pads.cuda()
    return torch.cat([out.data, zero_pads], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1      = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1        = nn.BatchNorm3d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2        = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride     = stride
        self.dilation   = dilation

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet(nn.Module):
    def __init__(self, block, layers, shortcut_type='A', no_cuda=False):
        # ↑ 默认改为 'A'，与官方训练一致（downsample 无可学习参数）
        self.inplanes = 64
        self.no_cuda  = no_cuda
        super().__init__()

        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7,
            stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1     = nn.BatchNorm3d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                # avg_pool 方式：无可学习参数，checkpoint 里不存这些 key
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:  # 'B'
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = [block(self.inplanes, planes,
                        stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x  = self.relu(self.bn1(self.conv1(x)))
        x  = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


# ============================================================
# ViSD-Boost Encoder
# ============================================================

class VISDBOOSTEncoder(nn.Module):
    """
    输入:  (B, 1, D, H, W)
    输出:  (B, 512)  —— 对 x4 做 GlobalAvgPool 后展平
    """
    def __init__(self, checkpoint_vqvae_path: str, checkpoint_path: str):
        super().__init__()

        # shortcut_type='A'：downsample 用 avg_pool，无可学习参数
        # 与两个 checkpoint 的 keys 完全吻合
        self.res_model = resnet18(shortcut_type='A', no_cuda=False)
        self.gap = nn.AdaptiveAvgPool3d(1)

        self._load_weights(checkpoint_vqvae_path, checkpoint_path)

        for p in self.res_model.parameters():
            p.requires_grad = False

    def _load_weights(self, ckpt_vqvae_path: str, ckpt_path: str):
        PREFIX = "visual_encoder.res_model."

        def extract(path):
            raw = torch.load(path, map_location="cpu")
            sd  = raw.get("model", raw)
            return {k[len(PREFIX):]: v
                    for k, v in sd.items()
                    if k.startswith(PREFIX)}

        # vqvae 只有 bn running stats；main checkpoint 有完整权重
        # main 覆盖 vqvae 的同名 key，其余 vqvae 独有 key 保留
        merged = {**extract(ckpt_path)}

        missing, unexpected = self.res_model.load_state_dict(merged, strict=False)

        print(f"[VISDBOOSTEncoder] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  MISSING keys:")
            for k in missing:
                print(f"    {k}")
        if unexpected:
            print("  UNEXPECTED keys:")
            for k in unexpected:
                print(f"    {k}")
        # shortcut_type='A' 时预期：missing=0, unexpected=0

    def forward(self, x):
        _, _, _, x4 = self.res_model(x)   # x4: (B, 512, D', H', W')
        return self.gap(x4).flatten(1)     # (B, 512)


# ============================================================
# Linear Probe 模型
# ============================================================

class VISDBOOSTPrognosis(nn.Module):
    def __init__(self,
                 num_classes: int,
                 checkpoint_vqvae_path: str,
                 checkpoint_path: str,
                 feature_dim: int = 512):
        super().__init__()
        self.encoder    = VISDBOOSTEncoder(checkpoint_vqvae_path, checkpoint_path)
        # self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        return feats