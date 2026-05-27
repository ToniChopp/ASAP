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
import torch.nn as nn

import timm.models.vision_transformer
import ipdb


class PatchEmbed3D(nn.Module):
    """ 
    3D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=768, norm_layer=None, flatten=True, in_chan_last=True):
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
            x = x.reshape(B * L, *self.img_size, self.in_chans).permute(0, 4, 1, 2, 3) # When patchification follows HWDC
        else:
            x = x.reshape(B * L, self.in_chans, *self.img_size)
        
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)

        x = x.reshape(B, L, self.embed_dim)
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, image_size=(224, 224, 112), patch_size=(16,16,8), in_chans=1,
                 embed_dim=768, global_pool=False, is_fvlm=False, is_clip=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.patch_embed = PatchEmbed3D(img_size=patch_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = np.prod(image_size) // np.prod(patch_size)

        self.is_fvlm = is_fvlm
        if self.is_fvlm:
            self.is_fvlm = False
            self.is_clip = True
        self.is_clip = is_clip
        
        if not self.is_fvlm:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.global_pool = global_pool

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    
    def patchify3D(self, imgs):
        """
        imgs: (N, 1, H, W, D)
        x: (N, L,  prod(patch_size) * 1)
        """
        p = self.patch_embed.patch_size
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p[0] == 0

        # h = w = d = imgs.shape[2] // p[0]
        h = imgs.shape[-3] // p[0]
        w = imgs.shape[-2] // p[1]
        d = imgs.shape[-1] // p[2]
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p[0], w, p[1], d, p[2]))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p[0] * p[1] * p[2] * 1))
        return x


    def forward_features(self, x):
        # ipdb.set_trace()
        B = x.shape[0]
        x = self.patchify3D(x)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if not self.is_fvlm:
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
        else:
            x = x + self.pos_embed
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if not self.global_pool:
            x = x[:, 0, :]
            outcome = self.norm(x)
        else:
            if not self.is_fvlm and not self.is_clip:
                x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                outcome = self.fc_norm(x)
            else:
                x = x[:, 0, :]
                outcome = self.fc_norm(x)

        return outcome
    
    def forward(self, x, return_encoded_tokens=False):
        x = self.forward_features(x)
        return x



def vit_base_patch16(image_size=(224, 224, 112), patch_size=(16, 16, 8), **kwargs):
    model = VisionTransformer(
        image_size=image_size, patch_size=patch_size, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# def vit_large_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_huge_patch14(**kwargs):
#     model = VisionTransformer(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model