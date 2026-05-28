
'''
Author: xm_cmic
Date: 2024-04-18 21:15:50
LastEditors: xm_cmic
LastEditTime: 2024-05-08 22:20:52
FilePath: /src-0508/model/global_model.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

from typing import Optional, Tuple

import math
import numpy as np 
import pandas as pd

import os 
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from peft import LoraConfig, get_peft_model
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from model.modules import Prompt_Encoder
from model.vision_transformer import vit_base_patch16
from model.m3d_vit import m3d_vit

import transformers
from transformers import GPT2Config,GPT2Tokenizer,GPT2LMHeadModel
from transformers import BertPreTrainedModel, BertModel, BertTokenizer,AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast

import ipdb

class QFormer(nn.Module):
    def __init__(self, vis_dim=768, num_query=32, num_layers=4):
        super().__init__()
        
        self.query_tokens = nn.Parameter(torch.randn(1, num_query, vis_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vis_dim,
            nhead=8,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vis_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, vision_tokens):
        B, N, C = vision_tokens.shape
        
        query = self.query_tokens.expand(B, -1, -1)  # (B, Q, C)

        # cross attention: query attends to vision tokens
        query, _ = self.cross_attn(query, vision_tokens, vision_tokens)

        # self-attention refinement
        query = self.transformer(query)

        return query  # (B, Q, C)


class Global_VQA_Model(nn.Module):
    def __init__(self, tokenizer, language_model, vision_backbone='UNET', vision_pretrained=None, vision_learnable=False, is_train=True):
        super().__init__()        
        self.vision_backbone = {
            'UNET' : PlainConvUNet(input_channels=1, 
                                   n_stages=6, 
                                   features_per_stage=(64, 64, 128, 256, 512, 768), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2, 2),
                                   num_classes=1, 
                                   n_conv_per_stage_decoder=(2, 2, 2, 2, 2), 
                                   conv_bias=True, 
                                   norm_op=nn.InstanceNorm3d,
                                   norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                   dropout_op=None,
                                   dropout_op_kwargs=None,
                                   nonlin=nn.LeakyReLU, 
                                   nonlin_kwargs=None,
                                   deep_supervision=True,
                                   nonlin_first=False
                                   ),
            'ViT': vit_base_patch16(
                image_size=(224, 224, 112),
                patch_size=(16, 16, 8),
                drop_path_rate=0.1*int(is_train),
            ),
            'ViT_M3D': m3d_vit(
                image_size=(112, 224, 224),
                patch_size=(4, 16, 16),
            ),
        }[vision_backbone]
        if vision_pretrained:
            if vision_backbone == 'UNET':
                checkpoint = torch.load(vision_pretrained, map_location='cpu')
                new_state_dict = {k.replace('module.backbone.', ''): v for k, v in checkpoint['model_state_dict'].items()}
                # Modify the problematic layers by averaging across the channel dimension
                for key in ['encoder.stages.0.0.convs.0.conv.weight', 'encoder.stages.0.0.convs.0.all_modules.0.weight']:
                    weight = new_state_dict[key]
                    averaged_weight = weight.mean(dim=1, keepdim=True)  # Averaging across the channel dimension
                    new_state_dict[key] = averaged_weight
                self.vision_backbone.load_state_dict(new_state_dict,strict=False)
                    
                model_keys = set(self.vision_backbone.state_dict().keys())
                loaded_keys = set(new_state_dict.keys())
                # 交集: 被加载的键
                loaded = model_keys & loaded_keys
                print("Loaded keys:", loaded)
            elif vision_backbone == 'ViT':
                if vision_pretrained != "random":
                    checkpoint = torch.load(vision_pretrained, map_location=torch.device('cpu'))
                    checkpoint_model = checkpoint['model']
                    msg = self.vision_backbone.load_state_dict(checkpoint_model, strict=False)
                    print(msg)
                else:
                    pass
            if vision_learnable:
                pass 
            else:
                for param in self.vision_backbone.parameters():
                    param.requires_grad = False
        else:
            self.vision_backbone.apply(InitWeights_He(1e-2))
        
        
        self.language_model = language_model
        self.text_dim = self.language_model.config.hidden_size
        
        self.tokenizer = tokenizer

        vis_dim = { # dim of latent embedding
            'UNET': 768,
            'ViT': 768,
        }[vision_backbone]
        

        self.image_proj_mlp = {
            'UNET' : nn.Sequential(
                        nn.Linear(vis_dim, 1024),
                        nn.GELU(),
                        nn.Linear(1024, self.text_dim),
                        nn.GELU(),
                    ),
            'ViT' : nn.Sequential(
                        nn.LayerNorm(vis_dim),
                        nn.Linear(vis_dim, 2048),
                        nn.GELU(),
                        nn.Linear(2048, 2048),
                        nn.GELU(),
                        nn.Linear(2048, self.text_dim),
                        nn.LayerNorm(self.text_dim),
                    ),
        }[vision_backbone]
        
        self.qformer = QFormer(vis_dim=vis_dim, num_query=64, num_layers=4)


        # ["abnormality", "presence", "size", "location", "disorder"]
        self.task_weight_proj = {0: 2.0, 1: 1.0, 2: 1.0, 3: 1.5, 4: 2.0}
        self.neg_signal_proj = {0: 2.0, 1: 0.5}
        
    
    def get_image_features(self, input_image):
        # Image Encoder and Pixel Decoder
        latent_embeddings = self.vision_backbone(input_image)
        
        vision_tokens = self.qformer(latent_embeddings)

        image_tokens = self.image_proj_mlp(vision_tokens)
        return image_tokens

    
    def forward(self, input_image, input_ids, task_idx, neg_signal, labels=None):
        image_features = self.get_image_features(input_image).bfloat16()
        input_embedding = self.language_model.get_input_embeddings()(input_ids)
        input_embedding = torch.cat([image_features,input_embedding], dim=1)

        output = self.language_model(
            inputs_embeds=input_embedding,
            labels=labels,
        )
        logits = output.logits
        
        max_key = max(self.task_weight_proj.keys())
        lookup = torch.zeros(max_key + 1).to(logits.device)
        for k, v in self.task_weight_proj.items():
            lookup[k] = v
        task_weights = lookup[task_idx]
        
        neg_mask = (task_idx == 0) | (task_idx == 4)
        neg_mask = neg_mask.float()
        lookup = torch.zeros(2).to(logits.device)
        for k, v in self.neg_signal_proj.items():
            lookup[k] = v
        neg_signal_weights = lookup[neg_signal.long()]
        neg_signal_weights = neg_signal_weights * neg_mask + (1 - neg_mask) * 1.0
        
        logits = logits.float()
        logits = logits.view(-1, logits.size(-1))
        shift_labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = shift_labels[..., 1:].contiguous()
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(logits.device)

        loss = nn.functional.cross_entropy(input=logits, target=shift_labels, ignore_index=-100, reduction='none')
        loss = loss.reshape(labels.shape[0], labels.shape[1])
        token_count = (labels != -100).sum(dim=1)
        sample_loss = loss.sum(dim=1) / token_count

        final_weights = task_weights * neg_signal_weights.squeeze(-1)
        loss = (sample_loss * final_weights).mean()

        return CausalLMOutputWithPast(
            logits=output.logits,
            loss=loss,
            past_key_values=output.past_key_values if hasattr(output, "past_key_values") else None,
            hidden_states=output.hidden_states if hasattr(output, "hidden_states") else None,
            attentions=output.attentions if hasattr(output, "attentions") else None,
        )


    def generate(self, input_sentence, input_image, input_ids, labels):
        image_features = self.get_image_features(input_image).bfloat16()
        # input_embedding = self.language_model.get_input_embeddings()(input_ids)
        # input_embedding = torch.cat([image_features,input_embedding], dim=1)
        # output = self.language_model(
        #     inputs_embeds=input_embedding,
        #     labels=labels
        # )
        # print('Loss:',output.loss)
        
        
        model_inputs = self.tokenizer(input_sentence, return_tensors='pt').to(self.language_model.device)
        generate_input_ids = model_inputs['input_ids']
        generate_input_embedding = self.language_model.get_input_embeddings()(generate_input_ids)
        generate_input_embedding = torch.cat([image_features, generate_input_embedding], dim=1)
        
        with torch.no_grad():
            beam_output = self.language_model.generate(
                inputs_embeds = generate_input_embedding,
                max_new_tokens=128,
                num_beams=3,
                do_sample=False,
                # early_stopping=True
            )
        output_sentence = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
        return output_sentence

    
if __name__ == '__main__':
    model =  PlainConvUNet(input_channels=1, 
                                   n_stages=6, 
                                   features_per_stage=(64, 64, 128, 256, 512, 768), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
                                   num_classes=1,
                                   n_conv_per_stage_decoder=(2, 2, 2, 2, 2), 
                                   conv_bias=True, 
                                   norm_op=nn.InstanceNorm3d,
                                   norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                   dropout_op=None,
                                   dropout_op_kwargs=None,
                                   nonlin=nn.LeakyReLU, 
                                   nonlin_kwargs=None,
                                   deep_supervision=True,
                                   nonlin_first=False
                                   ).cuda()
    input_image = torch.rand((1, 1, 256, 256, 64)).cuda()
    latent_embeddings = model(input_image)
    print(latent_embeddings.shape)
    image_embedding = rearrange(latent_embeddings[-1], 'b dim h w d -> b (h w d) dim')
    print(image_embedding.shape)