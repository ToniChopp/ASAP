import torch
from collections import OrderedDict as odict
import ipdb

# ------ Merge the weights of BERT and MAE pre-trained models ------
bert_checkpoint = "./pytorch_model.bin"
mae_checkpoint = "./mae_pretrain_vit_base.pth"

bert_pretrained_weights = torch.load(bert_checkpoint, map_location='cpu', weights_only=True)
mae_pretrained_weights = torch.load(mae_checkpoint, map_location='cpu', weights_only=True)["model"]
matched_keys = odict()
for k, v in bert_pretrained_weights.items():
    k = "bert_encoder." + k
    matched_keys[k] = v

for k, v in mae_pretrained_weights.items():
    if k.startswith("decoder") or k == "cls_token" or k =="pos_embed" or k == "mask_token" or k.startswith("patch"):
        continue
    else:
        matched_keys[k] = v

print(matched_keys.keys())
save_keys = dict()
save_keys["model"] = matched_keys
torch.save(save_keys, "checkpoint_CXRBERT_imagenetMAE.pth")


