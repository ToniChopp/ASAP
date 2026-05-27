import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate
import os
from src.args import parse_arguments
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import warnings
warnings.filterwarnings("ignore")

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
text_encoder.resize_token_embeddings(len(tokenizer))

backbone = 'swin_unetr'
backbone = 'vit'

args = parse_arguments()
if backbone == 'ct_clip':

    image_encoder = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 480,
        patch_size = 30,
        temporal_patch_size = 15,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    )
    clip = CTCLIP(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_image = 2097152,
        dim_text = 768,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False
    )
    clip.load("/scratch/medimgfmod/zchenhi/checkpoints/CT-CLIP/CT_CLIP_zeroshot.pt")

elif backbone == 'swin_unetr':

    from baseline.swin import Swin
    # feature_size = 48
    # checkpoint_path = "./exps_base/checkpoint_15000_epoch_9.pt"
    # feature_size = 96
    # checkpoint_path = "./exps_large/checkpoint_5000_epoch_8.pt"
    feature_size = 192
    checkpoint_path = "./exps_huge/checkpoint_15000_epoch_5.pt"

    swin = Swin(in_channels=1, feature_size=feature_size)

    clip = CTCLIP(
        image_encoder=swin, text_encoder=text_encoder,
        dim_image=20736, dim_text=768, dim_latent=768, #196608 20736
        extra_latent_projection=False, use_mlm=False,
        downsample_image_embeds=False, use_all_token_embeds=False
    )
    clip.load(checkpoint_path, strict=False)

elif backbone == 'vit':
    from baseline.vision_transformer import vit_base_patch16
    from baseline.vit_m3d import m3d_vit
    if args.name != "m3d":
        vit = vit_base_patch16(
            image_size=(args.roi_x, args.roi_y, args.roi_z),
            patch_size=(args.patch_size_x, args.patch_size_y, args.patch_size_z),
            drop_path_rate=0.1,
            is_fvlm=(args.name=="fvlm"),
            is_clip=(args.name=="ctclip")
        )
    else:
        vit = m3d_vit(
            image_size=(args.roi_x, args.roi_y, args.roi_z),
            patch_size=(args.patch_size_x, args.patch_size_y, args.patch_size_z),
        )

    clip = CTCLIP(
        image_encoder=vit,
        text_encoder=text_encoder,
        dim_image=768, dim_text=768, dim_latent=768, #dim_image=196608
        extra_latent_projection=False, use_mlm=False,
        downsample_image_embeds=False, use_all_token_embeds=False,
    )

    clip.load(args.checkpoint_path, strict=False)

### superpod project
# inference = CTClipInference(
#     clip,
#     data_folder = '/project/medimgfmod/zchenhi/data/CT-RATE/valid_preprocessed/',
#     reports_file = "/project/medimgfmod/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
#     labels = "/project/medimgfmod/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
#     batch_size = 1,
#     results_folder="./exps_base/",    #inference_zeroshot
#     num_train_steps = 1,
# )
    
if "CT-Rate" in args.checkpoint_path:
    results_folder = "./valid_output_retrieval/CT-Rate/"
    results_folder = os.path.join(results_folder, args.name, args.checkpoint_path.split('/')[-1].split('.')[0])
    print("Results will be saved to ", results_folder)

    inference = CTClipInference(
        clip,
        args,
        data_folder = '../../../Data/CT-Rate/',
        reports_file = "./dataset/CT-Rate_val_reports.csv",
        labels = "./dataset/CT-Rate_val_labels.csv",
        batch_size = 16,
        results_folder=results_folder,    # The path to save results
        num_train_steps = 1,
        is_retrieval=True,
    )

elif "AHPH" in args.checkpoint_path:
    results_folder = "./valid_output_retrieval/AHPH/"
    results_folder = os.path.join(results_folder, args.name, args.checkpoint_path.split('/')[-1].split('.')[0])
    print("Results will be saved to ", results_folder)

    inference = CTClipInference(
        clip,
        args,
        data_folder = '../../../Data/AHPH10K/',
        reports_file = "./dataset/val_reports.csv",
        labels = "./dataset/val_labels.csv",
        batch_size = 16,
        results_folder=results_folder,    # The path to save results
        num_train_steps = 1,
        is_retrieval=True,
    )

inference.infer()
