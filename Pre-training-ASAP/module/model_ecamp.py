# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_3d_sincos_pos_embed
from .bert_config import BertConfig
from .bert_modeling import MultimodalBertMaskedLM
from .ct_bert import CXRBertModel, CXRBertConfig


import ipdb


class SimilarityHead(nn.Module):
    def __init__(self):
        super().__init__()
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, img_features, sentence_features, sorted=True):
        # normalized features
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        sentence_features = sentence_features / (sentence_features.norm(dim=-1, keepdim=True) + 1e-8)

        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_img_per_sent = logit_scale * sentence_features @ img_features.transpose(-2, -1)
        logits_img_per_sent = sentence_features @ img_features.transpose(-2, -1)

        logits_img_per_sent_output = logits_img_per_sent
        if not sorted:
            return logits_img_per_sent_output
        sorted_logits_sent, _ = torch.sort(logits_img_per_sent_output, descending=True, dim=1)
        return sorted_logits_sent


class FeatureHead(nn.Module):
    """Feature head to be used for feature fusion.

    :param config: Configuration for BERT.
    """

    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.dense_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dense_to_output = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_to_hidden(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense_to_output(hidden_states)

        return hidden_states


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=768, norm_layer=None, flatten=True, in_chan_last=True) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = np.prod(self.grid_size)
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.in_chans = in_chans
        self.in_chan_last = in_chan_last
        
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, L, Dim = x.shape
        assert Dim == np.prod(self.patch_size) * self.in_chans, \
            f"Input image total size {Dim} doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})"        
        
        # input image is pathified 3D image
        if self.in_chan_last:
            x = x.reshape(B * L, *self.img_size, self.in_chans).permute(0, 4, 1, 2, 3).contiguous() # When patchification follows HWDC
        else:
            x = x.reshape(B * L, self.in_chans, *self.img_size).contiguous()
        
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()

        x = x.reshape(B, L, self.embed_dim).contiguous()
        return x
    

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))  # learnable query
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, L, D]
        Q = self.query.expand(x.size(0), -1, -1)     # [B, 1, D]
        K = self.key_proj(x)                         # [B, L, D]
        V = self.value_proj(x)                       # [B, L, D]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)  # [B, 1, L]
        attn_weights = F.softmax(attn_scores, dim=-1)                             # [B, 1, L]

        out = torch.matmul(attn_weights, V)  # [B, 1, D]
        return out


class ECAMP3D(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(224,224,112), patch_size=(16,16,8), in_chans=1,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio=0.75, using_ctbert=False, bert_decoder_layer=6, k=32):
        super().__init__()

        # --------------------------------------------------------------------------
        # image encoder specifics
        self.patch_embed = PatchEmbed3D(img_size=patch_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_size = patch_size

        self.num_patches = int(np.prod(img_size) / np.prod(patch_size))
        self.mask_ratio = mask_ratio
        self.using_ctbert = using_ctbert

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # image decoder and SR specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_layer = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * patch_size[2] * in_chans, bias=True)
        
        # Modify based on SimCroP, add total seg mask
        self.seg_num_classes = 9
        self.decoder_mask_pred_layer_0 = nn.Linear(decoder_embed_dim, self.seg_num_classes, bias=True)      # 9 is seg mask classes
        self.decoder_mask_pred_norm = norm_layer(self.seg_num_classes)
        self.decoder_mask_pred_act_fn = nn.ReLU(inplace=True)
        self.decoder_mask_pred_layer_1 = nn.Linear(self.seg_num_classes, self.seg_num_classes, bias=True)
        self.seg_mask_criterion = nn.BCEWithLogitsLoss()
        
        
        # self.decoder_act_fn = nn.ReLU(inplace=True)
        # self.ln = nn.LayerNorm(patch_size[0] * patch_size[1] * patch_size[2] * in_chans)
        # self.decoder_pred_layer_1 = nn.Linear(patch_size[0] * patch_size[1] * patch_size[2] * in_chans, patch_size[0] * patch_size[1] * patch_size[2] * in_chans, bias=True)

        # self.decoder_pred_conv_0 = nn.Conv3d(int(decoder_embed_dim/256), 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), padding_mode='replicate')
        # self.decoder_pred_conv_1 = nn.Conv3d(1, 1, kernel_size=(1,1,1))
        # self.relu = nn.ReLU(inplace=False)
        # self.decoder_mlp = nn.Linear(int(np.prod(patch_size)), int(np.prod(patch_size)), bias=True)

        # --------------------------------------------------------------------------
        # Bert encoder
        if self.using_ctbert:
            self.bert_encoder = CXRBertModel(CXRBertConfig(max_position_embeddings=512))
            self.bert_encoder.requires_grad_(False)
            self.bert_feature_proj = FeatureHead(embed_dim)
            
        self.bert_decoder = MultimodalBertMaskedLM(BertConfig(num_hidden_layers=bert_decoder_layer))
        self.bert_mlp = nn.Linear(embed_dim, embed_dim, bias=True)
        self.bert_topk_mlp = nn.Linear(embed_dim, embed_dim, bias=True)
        self.norm_pix_loss = norm_pix_loss


        # --------------------------------------------------------------------------
        # feature align
        self.finding_align_img_mlp = FeatureHead(embed_dim)
        self.align_text_mlp = FeatureHead(embed_dim)
        self.similarity_head = SimilarityHead()
        self.regression_head = nn.Linear(embed_dim, 1, bias=True)


        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], np.round(self.num_patches**(1/3)), num_tokens=0, cls_token=True)
        self.pos_embed.data.copy_(pos_embed.float())

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], np.round(self.num_patches**(1/3)), num_tokens=0, cls_token=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)


        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def patchify3D(self, imgs):
        """
        imgs: (N, 1, H, W, D)
        x: (N, L,  prod(patch_size) * 1)
        """
        p = self.patch_embed.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p[0] == 0

        h = w = d = imgs.shape[2] // p[0]
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p[0], w, p[1], d, p[2])).contiguous()
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p[0] * p[1] * p[2] * 1)).contiguous()
        return x


    def unpatchify3D(self, x):
        """
        x: (N, L,  Dim)
        imgs: (N, 1, H, W, D)
        """
        p = self.patch_embed.patch_size
        h = w = d = int(np.round(x.shape[1]**(1/3)))
        assert h * w * d == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, d, p[0], p[1], p[2], 1)).contiguous()
        x = torch.einsum('nhwdpqrc->nchpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p[0], w * p[1], d * p[2])).contiguous()
        return imgs


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, Dim = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, Dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep
    

    def mask_2_pixel(self, mask):
        """
        mask: [N, L], 1 is keep, 0 is remove, 
        pixel_mask: [N, 3, imgs_size, imgs_size], 1 is keep, 0 is remove
        """
        patch_size = self.patch_embed.patch_size

        mask = mask.reshape(shape=(mask.shape[0], int(np.round(mask.shape[1]**(1/3))), int(np.round(mask.shape[1]**(1/3))), int(np.round(mask.shape[1]**(1/3))))).contiguous()
        
        pixel_mask = torch.kron(mask, torch.ones((patch_size[0], patch_size[1], patch_size[2])).cuda())
        pixel_mask = pixel_mask.unsqueeze(1)
        return pixel_mask


    def image_encoder(self, x, mask_ratio):
        # patchify
        x = self.patchify3D(x)
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    

    def image_encoder_noise(self, x, mask_ratio, seg_mask):
        # patchify
        x = self.patchify3D(x)
        seg_mask = self.patchify3D(seg_mask)
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x_masked, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        organ_mask = (seg_mask != 0).any(dim=2).int()
        mask_noise = 1 - mask
        mask_noise = organ_mask * mask_noise
        noise_mask = torch.gather(mask_noise, dim=1, index=ids_keep)
        
        noise_prob = 0.5

        for b in range(x_masked.shape[0]):
            if torch.rand(1).item() < noise_prob:
                noise_mask_b = noise_mask[b].bool()
                if noise_mask_b.any():
                    num_noises = noise_mask_b.sum().item()
                    noise_b = torch.randn(num_noises, x_masked.shape[-1], device=x_masked.device) * 0.1  # small noise
                    x_masked[b, noise_mask_b] = x_masked[b, noise_mask_b] + noise_b


        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x_masked = blk(x_masked)
        x_masked = self.norm(x_masked)

        return x_masked, mask, noise_mask, ids_restore, ids_keep
    

    def image_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        decoder_outputs = []
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
            decoder_outputs.append(x)

        # Modify based on SimCroP, add total seg mask
        mask_pred = self.decoder_mask_pred_layer_0(decoder_outputs[1][:, 1:, :])
        mask_pred = self.decoder_mask_pred_norm(mask_pred)
        mask_pred = self.decoder_mask_pred_act_fn(mask_pred)
        mask_pred = self.decoder_mask_pred_layer_1(mask_pred)

        # ------ linear for dimension transformation ------
        x = self.decoder_norm(x[:, 1:, :])
        x = self.decoder_pred_layer(x)
        x_pred = x


        # ------ conv & super resolution for dimension transformation ------
        # remove cls token
        # x = x[:, 1:, :]
        # B, L, Dim = x.shape
        # h = w = d = int(np.round(L**(1/3)))
        # assert h * w * d == L

        # # predict the pixel values
        # self.channel_patch_size = (int(self.patch_size[0] / 2), int(self.patch_size[1] / 2), int(self.patch_size[2] / 2))  # super resolution for reconstruction
        # x = x.reshape(B * L, *self.channel_patch_size, x.shape[2] // np.prod(self.channel_patch_size)).permute(0, 4, 1, 2, 3).contiguous()
        
        # # ipdb.set_trace()
        # x_pred = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        # x_pred = self.decoder_pred_conv_0(x_pred)
        # x_pred = self.relu(x_pred)
        # residual = x_pred
        # x_pred = self.decoder_pred_conv_1(x_pred)
        # x_pred = self.relu(x_pred + residual)
        # x_pred = x_pred.reshape(B, L, *self.patch_size).contiguous()

        # # ipdb.set_trace()
        # x_pred = x_pred.reshape(B, L, -1).contiguous()
        # x_pred = self.decoder_mlp(x_pred)

        
        return mask_pred, x_pred
    

    def forward_report_decoder(
        self,
        latent=None,
        input_ids=None,
        report_feature=None,
        labels=None,
        attention_mask=None,
        token_type_ids=None,
        topk_idx=None,
        topk_mask=None,
        topk_similarity_patch=None,
        topk_img_feature=None,
        topk_img_feature_mask=None,
    ):
        r"""
        latent: [N, L, D]
        report_feature: [N, Length, D]
        labels: [N, Length]
        attention_mask: [N, Length]
        token_type_ids: [N, Length]
        """

        # latent = latent
        latent = self.bert_mlp(latent[:, 1:, :])

        # ---- topk index select ----
        # topk_img_feature = torch.gather(latent, dim=1, index=topk_idx.unsqueeze(-1).repeat(1, 1, latent.shape[-1]))

        # topk_img_feature = torch.concat((latent, topk_img_feature), dim=1) if topk_img_feature is not None else latent
        # topk_img_feature = self.bert_topk_mlp(topk_img_feature)
        # img_mask = torch.ones((topk_mask.shape[0], latent.size(1)), device=latent.device)
        # topk_mask = torch.concat((img_mask, topk_mask.int()), dim=1)
        # ---- topk index select ----
        
        gap_token = latent.mean(dim=1)
        gap_token = gap_token.unsqueeze(1)
        if report_feature is not None:
            report_feature = self.bert_feature_proj(report_feature)
        if topk_similarity_patch is not None:
            # topk_similarity_patch_effective_mask = (topk_similarity_patch.abs().sum(dim=-1) > 0)
            # topk_similarity_patch_cls = topk_similarity_patch.sum(dim=1) / (topk_similarity_patch_effective_mask.sum(dim=1, keepdim=True) + 1e-6)
            # topk_similarity_patch = torch.where(
            #     topk_similarity_patch_effective_mask.unsqueeze(-1).expand(-1,-1,topk_similarity_patch.shape[-1]),
            #     topk_similarity_patch,
            #     topk_similarity_patch_cls.unsqueeze(1).expand(-1,topk_similarity_patch.shape[1],-1))
            topk_similarity_patch_cls = torch.zeros((topk_similarity_patch.shape[0], topk_similarity_patch.shape[-1]), device=topk_similarity_patch.device)
            topk_similarity_patch = torch.concat((topk_similarity_patch_cls.unsqueeze(1), topk_similarity_patch), dim=1)

            pad_len = report_feature.shape[1] - topk_similarity_patch.shape[1]
            topk_similarity_patch = torch.concat((topk_similarity_patch, topk_similarity_patch_cls.unsqueeze(1).repeat(1, pad_len, 1)), dim=1)
            topk_similarity_patch = 0.02 * topk_similarity_patch

        mlm_loss = self.bert_decoder(
            latent=latent,
            gap_token=gap_token,
            report_feature=report_feature,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            topk_similarity_patch=topk_similarity_patch,
            topk_img_feature=topk_img_feature,
            topk_img_feature_mask=topk_img_feature_mask,
        )
        return mlm_loss
    

    def forward_loss(self, imgs, pred, seg_masks, pred_seg_masks, mask, noise_mask=None, ids_keep=None, epoch=0):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p[0]*p[1]*p[2]]
        seg_masks: [N, 3, H, W]
        pred_seg_masks: [N, L, p[0]*p[1]*p[2]]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        pixel_mask = self.mask_2_pixel(mask)

        # get back to image space for super resolution
        pred_img = self.unpatchify3D(pred)
        mask_pred_imgs = pred_img * pixel_mask
        mask_imgs = imgs * pixel_mask

        mim_loss = F.mse_loss(mask_pred_imgs, mask_imgs, reduction='mean')
        # mim_loss = mim_loss.sum() / pixel_mask.sum()
        

        # seg mask loss
        seg_mask_patches = self.patchify3D(seg_masks)
        seg_mask_patches_onehot = F.one_hot(seg_mask_patches.long(), num_classes=self.seg_num_classes)
        seg_mask_label_counts = seg_mask_patches_onehot.sum(dim=2)
        seg_mask_labels = seg_mask_label_counts / seg_mask_label_counts.sum(dim=2, keepdim=True)
        seg_mask_loss = self.seg_mask_criterion(pred_seg_masks, seg_mask_labels)

        # denoise loss
        if epoch < 200:
            noise_loss = torch.tensor(0.0).to(mim_loss.device)
            return mim_loss, seg_mask_loss, noise_loss
        else:
            imgs = self.patchify3D(imgs)
            pred_noise = torch.gather(pred, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, pred.shape[-1]))
            imgs = torch.gather(imgs, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, imgs.shape[-1]))

            noise_pred_imgs = pred_noise * noise_mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])
            noise_imgs = imgs * noise_mask.unsqueeze(-1).repeat(1, 1, imgs.shape[-1])
            noise_loss = F.mse_loss(noise_pred_imgs, noise_imgs, reduction='none')
            noise_loss = noise_loss.sum() / noise_mask.unsqueeze(-1).repeat(1, 1, imgs.shape[-1]).sum()


            return mim_loss, seg_mask_loss, noise_loss


    # def forward_loss(self, imgs, pred, seg_masks, pred_seg_masks, mask):
    #     """
    #     imgs: [N, 3, H, W]
    #     pred: [N, L, p[0]*p[1]*p[2]]
    #     seg_masks: [N, 3, H, W]
    #     pred_seg_masks: [N, L, p[0]*p[1]*p[2]]
    #     mask: [N, L], 0 is keep, 1 is remove, 
    #     """
    #     pixel_mask = self.mask_2_pixel(mask)

    #     # get back to image space for super resolution
    #     pred_img = self.unpatchify3D(pred)
    #     mask_pred_imgs = pred_img * pixel_mask
    #     mask_imgs = imgs * pixel_mask

    #     mim_loss = F.mse_loss(mask_pred_imgs, mask_imgs, reduction='mean')
    #     # mim_loss = mim_loss.sum() / pixel_mask.sum()
        
    #     # # denoise loss
    #     # imgs = self.patchify3D(imgs)
    #     # pred_noise = torch.gather(pred, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, pred.shape[-1]))
    #     # imgs = torch.gather(imgs, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, imgs.shape[-1]))

    #     # noise_pred_imgs = pred_noise * noise_mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])
    #     # noise_imgs = imgs * noise_mask.unsqueeze(-1).repeat(1, 1, imgs.shape[-1])
    #     # noise_loss = F.mse_loss(noise_pred_imgs, noise_imgs, reduction='mean')
    #     # # noise_loss = noise_loss.sum() / noise_mask.unsqueeze(-1).repeat(1, 1, imgs.shape[-1]).sum()

    #     # mim_loss = mim_loss + noise_loss

    #     # seg mask loss
    #     seg_mask_patches = self.patchify3D(seg_masks)
    #     seg_mask_patches_onehot = F.one_hot(seg_mask_patches.long(), num_classes=self.seg_num_classes)
    #     seg_mask_label_counts = seg_mask_patches_onehot.sum(dim=2)
    #     seg_mask_labels = seg_mask_label_counts / seg_mask_label_counts.sum(dim=2, keepdim=True)
    #     seg_mask_loss = self.seg_mask_criterion(pred_seg_masks, seg_mask_labels)

    #     return mim_loss, seg_mask_loss
    

    def finding_patch_align(self, latent, report_feature, sentence_mask_f, temperature=0.07):
        image_feature = self.finding_align_img_mlp(latent[:, 1:])
        # image_feature = latent
        report_feature = report_feature[:, 1:, :]       # remove cls token
        sentence_mask_f = sentence_mask_f[:, 1:]       # remove cls token
        report_feature = self.align_text_mlp(report_feature)

        # calculate sentence feature
        sentence_nums = sentence_mask_f.max(dim=1)[0]
        max_sentence_num = sentence_nums.max().item() + 1        # cls token and padding is included
        sentence_feature = torch.zeros(report_feature.shape[0], max_sentence_num, report_feature.shape[2],
                                       device=report_feature.device, dtype=report_feature.dtype)
        sentence_count = torch.zeros(report_feature.shape[0], max_sentence_num, 1, device=report_feature.device,
                                     dtype=report_feature.dtype)

        expanded_sentence_mask_f = sentence_mask_f.unsqueeze(-1).expand(-1, -1, report_feature.shape[2])
        sentence_feature.scatter_add_(1, expanded_sentence_mask_f, report_feature)
        sentence_count.scatter_add_(1, sentence_mask_f.unsqueeze(-1), torch.ones_like(report_feature[:, :, :1]))
        sentence_feature = sentence_feature / torch.clamp(sentence_count, min=1.0)
        
        # sentence feature without the padding token of findings
        sentence_feature = sentence_feature[:, 1:, :]  # remove padding token

        # ---- adaptive top k num ----

        # ---- argmax way ----
        # sentence_topk_num = self.regression_head(sentence_feature)  # [N, max_sentence_num-1, D]
        # sentence_topk_num = sentence_topk_num.argmax(dim=-1)  # [N, max_sentence_num-1]
        # sentence_topk_num = sentence_topk_num + 4

        # ---- sigmoid way ----
        sentence_topk_num = self.regression_head(sentence_feature).squeeze(-1)  # [N, max_sentence_num-1]
        sentence_topk_num = sentence_topk_num.sigmoid()
        sentence_topk_num = ((sentence_topk_num * 3) + 4).int()

        # ---- sigmoid new way ----
        # sentence_topk_num = self.regression_head(sentence_feature)  # [N, max_sentence_num-1]
        # sentence_topk_num = sentence_topk_num.sigmoid()
        # sentence_topk_num = sentence_topk_num.sum(dim=-1).int()
        # sentence_topk_num = sentence_topk_num + 4
        
        sentence_topk_num = 2 ** sentence_topk_num

        max_k = sentence_topk_num.max().item()
        topk_mask = torch.arange(max_k, device=sentence_topk_num.device).unsqueeze(0) < sentence_topk_num.unsqueeze(-1)

        # fetch top k similar image patches for each sentence
        # image_feature: [N, L, D], sentence_feature: [N, max_sentence_num, D]
        sent2img_similarity = self.similarity_head(image_feature, sentence_feature, sorted=False)
        topk_similarity, topk_idx = sent2img_similarity.topk(max_k, dim=2)

        topk_index = topk_idx.unsqueeze(-1).expand(-1, -1, -1, image_feature.shape[-1])
        topk_img_feature = image_feature.unsqueeze(1).expand(-1, max_sentence_num-1, -1, -1).gather(2, topk_index)

        topk_img_feature = topk_img_feature * topk_mask.unsqueeze(-1).int()

        # ---- global average pooling of top k image patch features for adaptive top k----
        gap_img_feature = topk_img_feature.sum(dim=2) / torch.clamp(topk_mask.sum(dim=2, keepdim=True), min=1.0)

        align_mask = torch.zeros(gap_img_feature.shape[0], gap_img_feature.shape[1], device=gap_img_feature.device)
        for i in range(gap_img_feature.shape[0]):
            align_mask[i, :sentence_nums[i]] = 1

        # calculate alignment loss
        # using contrastive loss (CLIP loss)
        # using align mask to filter padding sentences
        feature_sent = F.normalize(sentence_feature, p=2, dim=-1)
        feature_img = F.normalize(gap_img_feature, p=2, dim=-1)

        logits_sent2img = torch.matmul(feature_sent, feature_img.transpose(-2, -1)) / temperature
        
        
        # ---- Traditional contrastive loss ----
        logits_img2sent = logits_sent2img.transpose(-2, -1)
        labels = torch.arange(logits_sent2img.size(1)).unsqueeze(0).expand(logits_sent2img.size(0), -1).cuda()
        contrastive_loss_fct = nn.CrossEntropyLoss(reduction='none')
        contrastive_loss_sent2img = contrastive_loss_fct(logits_sent2img, labels)
        # contrastive_loss_sent2img = (contrastive_loss_sent2img * align_mask).sum(dim=-1) / (align_mask.sum(dim=-1) + 1e-6)
        contrastive_loss_sent2img = torch.masked_select(contrastive_loss_sent2img, align_mask.bool())
        contrastive_loss_sent2img = contrastive_loss_sent2img.mean()
        contrastive_loss_img2sent = contrastive_loss_fct(logits_img2sent, labels)
        # contrastive_loss_img2sent = (contrastive_loss_img2sent * align_mask).sum(dim=-1) / (align_mask.sum(dim=1) + 1e-6)
        contrastive_loss_img2sent = torch.masked_select(contrastive_loss_img2sent, align_mask.bool())
        contrastive_loss_img2sent = contrastive_loss_img2sent.mean()
        finding_patch_align_loss = contrastive_loss_sent2img + contrastive_loss_img2sent

        # ---- SigLip loss from Github ----
        # labels = -torch.ones((logits_sent2img.size(1), logits_sent2img.size(2)), device=logits_sent2img.device).unsqueeze(0).expand(logits_sent2img.size(0), -1, -1)
        # labels = 2 * torch.eye(logits_sent2img.size(1), device=logits_sent2img.device).unsqueeze(0).expand(logits_sent2img.size(0), -1, -1) + labels
        # finding_patch_align_loss = -F.logsigmoid(labels * logits_sent2img)
        # align_mask = align_mask.bool().unsqueeze(2) & align_mask.bool().unsqueeze(1)
        # align_mask = align_mask | align_mask.transpose(-1, -2)
        # finding_patch_align_loss = (finding_patch_align_loss * align_mask).sum(dim=(1, 2)) / (align_mask.sum(dim=(1, 2)) + 1e-6)
        # finding_patch_align_loss = finding_patch_align_loss.mean()
        
        # topk_similarity = topk_similarity.reshape(topk_similarity.shape[0], -1)
        topk_mask = topk_mask * align_mask.unsqueeze(-1).repeat(1, 1, topk_mask.shape[-1])
        # topk_mask = topk_mask.reshape(topk_mask.shape[0], -1)
        # topk_idx = topk_idx.reshape(topk_idx.shape[0], -1)

        topk_similarity = topk_similarity * topk_mask

        topk_similarity_patch = torch.zeros((
            image_feature.shape[0],
            topk_similarity.shape[1],
            image_feature.shape[1]),
            device=image_feature.device,
            dtype=topk_similarity.dtype)
        
        # topk_index_count_patch = torch.zeros((
        #     image_feature.shape[0],
        #     image_feature.shape[1]),
        #     device=image_feature.device,
        #     dtype=topk_mask.dtype)
        for i in range(topk_similarity.shape[0]):
            topk_similarity_patch[i].scatter_add_(1, topk_idx[i], topk_similarity[i])
        # topk_index_count_patch.scatter_add_(1, topk_idx, topk_mask)
        # topk_similarity_patch = topk_similarity_patch / torch.clamp(topk_index_count_patch, min=1.0)
        # topk_similarity_patch = topk_similarity_patch / torch.clamp(topk_mask.sum(dim=-1, keepdim=True), min=1.0)
        topk_similarity_patch_padding = torch.zeros((
            image_feature.shape[0],
            image_feature.shape[1]),
            device=image_feature.device,
            dtype=topk_similarity.dtype)
        topk_similarity_patch = torch.concat((topk_similarity_patch_padding.unsqueeze(1), topk_similarity_patch), dim=1)
        topk_similarity_patch = torch.gather(topk_similarity_patch, dim=1, index=sentence_mask_f.unsqueeze(-1).expand(-1, -1, topk_similarity_patch.shape[-1]))
        topk_similarity_patch = topk_similarity_patch.detach()

        # --- keep unique index ---
        # topk_idx_1 = topk_idx.unsqueeze(-1)
        # topk_idx_2 = topk_idx.unsqueeze(-2)
        # equal = (topk_idx_1 == topk_idx_2)
        # tril = torch.tril(torch.ones(topk_idx.shape[1], topk_idx.shape[1], dtype=torch.bool, device=topk_idx.device))
        # eq_tril = equal & tril
        # count = eq_tril.sum(dim=-1)
        # first_occurrence_mask = count == 1
        # topk_mask = topk_mask * first_occurrence_mask
        # --- keep unique index ---

        
        return finding_patch_align_loss, topk_similarity_patch, gap_img_feature, align_mask
    

    def get_topk_patch(self, image_feature, sentence_feature, max_sentence_num, k=32):
        # fetch top k similar image patches for each sentence
        # image_feature: [N, L, D], sentence_feature: [N, max_sentence_num, D]
        sent2img_similarity = self.similarity_head(image_feature, sentence_feature, sorted=False)
        _, topk_idx = sent2img_similarity.topk(k, dim=1)
        topk_idx = topk_idx.transpose(-2, -1)
        topk_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, image_feature.shape[-1])
        topk_img_feature = image_feature.unsqueeze(1).expand(-1, max_sentence_num-1, -1, -1).gather(2, topk_idx)
        
        # global average pooling of top k image patch features
        gap_img_feature = topk_img_feature.mean(dim=2)

        return gap_img_feature



    def forward(self, batch, epoch):
        mask_ratio = self.mask_ratio
        imgs = batch["image"]
        ids_f, attention_mask_f, type_ids_f, sentence_mask_f = batch["ids_f"], batch["attention_mask_f"], batch["type_ids_f"], batch["sentence_mask_f"]
        ids, labels, attention_mask, type_ids = batch["masked_ids"], batch["ids"], batch["attention_mask"], batch["type_ids"]
        seg_masks = batch["mask"]

        imgs = imgs.cuda()
        seg_masks = seg_masks.cuda()

        ids_f = ids_f.cuda()
        attention_mask_f = attention_mask_f.cuda()
        type_ids_f = type_ids_f.cuda()
        sentence_mask_f = sentence_mask_f.cuda()
        ids = ids.cuda()
        labels = labels.cuda()
        attention_mask = attention_mask.cuda()
        type_ids = type_ids.cuda()

        if epoch < 200:
            latent, img_mask, ids_restore = self.image_encoder(imgs, mask_ratio)                  # image feature
            noise_mask = None
            ids_keep = None
        else:
            latent, img_mask, noise_mask, ids_restore, ids_keep = self.image_encoder_noise(imgs, mask_ratio, seg_masks)

        if self.using_ctbert:
            finding_feature = self.bert_encoder(ids_f, attention_mask_f, type_ids_f).last_hidden_state
            report_feature = self.bert_encoder(ids, attention_mask, type_ids).last_hidden_state
            
            align_loss, topk_similarity_patch, topk_img_feature, topk_img_feature_mask = self.finding_patch_align(latent, finding_feature, sentence_mask_f)

            pred_seg_masks, pred_img = self.image_decoder(latent, ids_restore)  # [N, L, p*p*3]

            mim_loss, seg_mask_loss, noise_loss = self.forward_loss(imgs, pred_img, seg_masks, pred_seg_masks, img_mask, noise_mask, ids_keep, epoch)
            # mim_loss, seg_mask_loss = self.forward_loss(imgs, pred_img, seg_masks, pred_seg_masks, img_mask)

            mlm_loss = self.forward_report_decoder(
                latent=latent,
                report_feature=report_feature,
                labels=labels,
                attention_mask=attention_mask,
                token_type_ids=type_ids,
                topk_similarity_patch=topk_similarity_patch,
                topk_img_feature=topk_img_feature,
                topk_img_feature_mask=topk_img_feature_mask,
            )

            return mim_loss, seg_mask_loss, noise_loss, mlm_loss, align_loss


def ecamp(**kwargs):
    model = ECAMP3D(
        patch_size=(16,16,8), in_chans=1, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        bert_decoder_layer=6, **kwargs)
    return model
