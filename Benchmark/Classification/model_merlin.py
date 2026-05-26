import torch
import torch.nn as nn
from merlin import Merlin

# ============================================================
# 2. Merlin Feature Extractor（替换原来的 ViT backbone）
# ============================================================

class MerlinEncoder(nn.Module):
    """
    将 Merlin 封装为与 ViT 接口一致的 feature extractor。
    Merlin(ImageEmbedding=True).forward(image) 返回 (img_emb,)
    img_emb shape: (B, feature_dim)  — ResNet152 出来通常是 2048
    """
    def __init__(self):
        super().__init__()
        # 加载官方权重（首次运行自动从 HuggingFace 下载）
        self.merlin = Merlin(ImageEmbedding=True)
        # 冻结所有 Merlin 参数（linear probe 只训练分类头）
        for param in self.merlin.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x: (B, C, D, H, W)  —— Merlin DataLoader 出来的 3D CT tensor
        returns: (B, feature_dim)
        """
        outputs = self.merlin(x)   # 返回 tuple，第0项是 image embedding
        return outputs[0]          # (B, feature_dim)


# ============================================================
# 3. Linear Probe 模型（替换原来的 ViT + 分类头）
# ============================================================

class MerlinLinearProbe(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 2048):
        """
        Args:
            num_classes:  你的分类任务类别数
            feature_dim:  Merlin 输出的特征维度，默认 2048（ResNet152 全局池化后）
        """
        super().__init__()
        self.encoder = MerlinEncoder()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)          # (B, feature_dim)，不带梯度
        logits = self.classifier(feats)  # (B, num_classes)
        return logits