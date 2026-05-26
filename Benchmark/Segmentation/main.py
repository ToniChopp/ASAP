# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
from functools import partial
import logging
logging.disable(logging.WARNING)
import numpy as np
import torch
from copy import deepcopy
cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from datetime import timedelta
import wandb

# from network.unetr_old import UNETR
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader
from unetr import UNETR
from swin_unetr import SwinUNETR
from merlin_unetr import MerlinUNETR
# from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

import ipdb


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--output_dir", default="./output", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_path", default="./pretrained_path/checkpoint.pth", type=str, help="pretrained checkpoint location"
)

parser.add_argument("--task", default="LUNA16", type=str, help="dataset task name")
parser.add_argument("--data_dir", default="./data/", type=str, help="dataset directory")
parser.add_argument("--json_path", default="dataset_100.json", type=str, help="dataset json file")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=2000, type=int, help="max number of training epocmax_epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=50, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", default=False, help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--seed", default=42, type=int, help="random seed")
parser.add_argument("--model_name", default="unetr", type=str, help="model name")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_ratio", default=4, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-1000.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=1000.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=-1.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=3.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=48, type=int, help="roi size in z direction")
parser.add_argument("--patch_size_x", default=16, type=int, help="patch size in x direction")
parser.add_argument("--patch_size_y", default=16, type=int, help="patch size in y direction")
parser.add_argument("--patch_size_z", default=8, type=int, help="patch size in z direction")
parser.add_argument("--pos", default=1, type=int, help="positive sample radius")
parser.add_argument("--neg", default=1, type=int, help="negative sample radius")
parser.add_argument("--num_samples", default=16, type=int, help="number of pos-neg crop samples during training")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--start_val_epochs", default=10, type=int, help="number of epochs to start validation")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--stage", default="train", type=str, help="train or test")
parser.add_argument("--patience", default=10, type=int, help="patience for early stopping")


def main():
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parser.parse_args()

    wandb_init = False
    wandb.init(
        project=f"Segmentation_{args.task}",
        name=args.output_dir.split('/')[-2] + "/" + args.output_dir.split('/')[-1],
        config={
            "lr": args.optim_lr,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "warmup_epochs": args.warmup_epochs,
            "max_epochs": args.max_epochs,
        }
    )
    wandb_init = True

    test_dices = []
    # Fine-tuning for 5 times with different random seeds
    random.seed(args.seed)
    random_seeds = random.sample(range(0, 10000), 5)
    for seed_id in range(5):
        args_run = deepcopy(args)

        args_run.seed = random_seeds[seed_id]

        print("#########################")
        print("####### Seed is %d #######" % args_run.seed)
        print("#########################")

    
        args_run.amp = not args_run.noamp
        args_run.cache_dir = args_run.data_dir + "/cache"
        if not os.path.exists(args_run.cache_dir):
            os.makedirs(args_run.cache_dir)
        print(args_run.json_path)

        if args_run.local_rank in [-1, 0]:
            os.makedirs(args_run.output_dir, exist_ok=True)

        # Setup CUDA, GPU & distributed training
        if args_run.local_rank == -1:
            print('##############################')   
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args_run.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args_run.local_rank)
            device = torch.device("cuda", args_run.local_rank)
            torch.distributed.init_process_group(backend='nccl',
                                                timeout=timedelta(minutes=60)
                                                )
            args_run.n_gpu = 1
        args_run.device = device

        set_seed(args_run)

        # Train:
        if args_run.stage == "train":
            if args_run.distributed:
                args_run.ngpus_per_node = torch.cuda.device_count()
                print("Found total gpus", args_run.ngpus_per_node)
                args_run.world_size = args_run.ngpus_per_node * args_run.world_size
                mp.spawn(main_worker, nprocs=args_run.ngpus_per_node, args=(args_run,))
            else:
                main_worker(gpu=0, args=args_run, wandb_init=wandb_init)

        # Test
        args_run.stage = "test"
        args_run.pretrained_path = os.path.join(args_run.output_dir, "model_{args_run.seed}.pt")
        # args_run.output_dir = os.path.join(args_run.output_dir, "test")
        args_run.max_epochs = 0
        if args_run.rank == 0:
            avg_dice, std_dice = main_worker(gpu=0, args=args_run, wandb_init=wandb_init)
            test_dices.append(avg_dice)

    if args.local_rank in [-1, 0]:
        print("#########################")
        print("####### Final Results #######")
        print("Test Dice over 3 runs: ", test_dices)
        print("Average Test Dice: ", np.mean(test_dices))
        print("Std of Test Dice: ", np.std(test_dices))
        wandb.log({"Final_AVG_Dice": np.mean(test_dices), "Final_STD_Dice": np.std(test_dices)})
        print("#########################")
    



def main_worker(gpu, args, wandb_init=False):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    args.test_external = False
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    is_m3d = False
    is_merlin = False
    is_visdboost = False
    if "M3D" in args.pretrained_path or "m3d" in args.pretrained_path:
        is_m3d = True
    if "merlin" in args.pretrained_path.lower():
        is_merlin = True
    if "ViSD-Boost" in args.pretrained_path or "visdboost" in args.pretrained_path.lower():
        is_visdboost = True

    if (args.model_name is None) or args.model_name == "unetr":
        if not is_merlin and not is_visdboost:
            model = UNETR(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y, args.roi_z),
                patch_size=(args.patch_size_x, args.patch_size_y, args.patch_size_z) if hasattr(args, 'patch_size_x') else (16, 16, 8),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_ratio=args.mlp_ratio,
                num_heads=args.num_heads,
                pos_embed="perceptron",
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                is_m3d=is_m3d,
            ).cuda()
        elif is_merlin:
            model = MerlinUNETR(
                freeze_encoder=False,
                num_classes=args.out_channels,
            ).cuda()

        # model = UNETR(
        #     in_channels=args.in_channels,
        #     out_channels=args.out_channels,
        #     img_size=(args.roi_x, args.roi_y, args.roi_z),
        #     feature_size=args.feature_size,
        #     hidden_size=args.hidden_size,
        #     mlp_dim=3072,
        #     num_heads=args.num_heads,
        #     pos_embed="perceptron",
        #     norm_name=args.norm_name,
        #     conv_block=True,
        #     res_block=True,
        #     dropout_rate=args.dropout_rate
        # ).cuda()
        #model = build_hybird(in_channel=args.in_channels, n_classes=args.out_channels, img_size=96).cuda()
        #model = MobileUNETR(
        #    in_channel=args.in_channels,
        #    n_classes=args.out_channels,
        #    channels=(32, 64, 96, 128, 192),
        #    depths=(1, 1, 3, 3, 1),
        #    kernels=(3, 3, 3, 3, 3),
        #    resolution=96,
        #    sparse=False
        #).cuda()
        
        # print(model)

        if args.stage == "test":
            pretrained_dict = torch.load(args.pretrained_path, map_location='cpu', weights_only=False)["state_dict"]
            msg = model.load_state_dict(pretrained_dict, strict=True)
            print(msg)
        else:
            if args.pretrained_path != "random":
                if "M3D" in args.pretrained_path:
                    pretrained_dict = torch.load(args.pretrained_path, map_location='cpu', weights_only=False)
                elif "merlin" in args.pretrained_path.lower():
                    pass
                elif "ViSD-Boost" in args.pretrained_path or "visdboost" in args.pretrained_path.lower():
                    pass
                else:
                    pretrained_dict = torch.load(args.pretrained_path, map_location='cpu', weights_only=False)['model']
                model_dict = model.state_dict()
                matched_dict = {}

                if is_m3d:
                    del pretrained_dict["patch_embedding.position_embeddings"]
                    del pretrained_dict["patch_embedding.patch_embeddings.1.weight"]
                    del pretrained_dict["patch_embedding.patch_embeddings.1.bias"]

                if "fVLM_official" in args.pretrained_path or args.task == "ACDC" or "HLIP" in args.pretrained_path:
                    if "patch_embed.proj.weight" in pretrained_dict:
                        del pretrained_dict["patch_embed.proj.weight"]
                        del pretrained_dict["patch_embed.proj.bias"]

                if "merlin" not in args.pretrained_path.lower() and "ViSD-Boost" not in args.pretrained_path:
                    for k, v in pretrained_dict.items():
                        # if args.task == "LUNA16":
                        #     if k == "pos_embed":
                        #         matched_dict["vit." + k] = v[:, 1:, :]
                        #         continue
                        if ("vit." + k) in model_dict.keys() and "cls_token" not in k and "pos_embed" not in k:
                            matched_dict["vit." + k] = v
        
                    msg = model.load_state_dict(matched_dict, strict=False)
                    print(msg)

            if args.resume_jit:
                if not args.noamp:
                    print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                    args.amp = args.noamp
                model = torch.jit.load(args.pretrained_path, map_location='cpu')
    
    elif args.model_name == "swin_unetr":
        model = SwinUNETR(
            img_size=(args.roi_z, args.roi_y, args.roi_x),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
            use_v2=True
        )

        if args.stage == "test":
            pretrained_dict = torch.load(args.pretrained_path, map_location='cpu', weights_only=False)["state_dict"]
            msg = model.load_state_dict(pretrained_dict, strict=True)
            print(msg)
        else:
            if args.pretrained_path != "random":
                pretrained_dict = torch.load(args.pretrained_path, map_location='cpu', weights_only=False)
                model_dict = model.state_dict()
                msg = model.load_state_dict(model_dict, strict=False)
                print(msg)

    
    else:
        raise ValueError("Unsupported model " + str(args.model_name))

    dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
    )
    post_label = AsDiscrete(to_onehot=args.out_channels, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(params=[
        {'params': model.encoder.parameters(), 'lr': 0.1 * args.optim_lr},
        {'params': model.decoder.parameters(), 'lr': args.optim_lr},
    ], lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        #optimizer = torch.optim.AdamW(params=[
        #    {'params': model.cnn.parameters(), 'lr': 0.1 * args.optim_lr},
        #    {'params': model.mae.parameters(), 'lr': 0.1 * args.optim_lr},
        #    {'params': model.decoder.parameters(), 'lr': args.optim_lr},
        #        ], lr=args.optim_lr, weight_decay=args.reg_weight)
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            params=[
        {'params': model.encoder.parameters(), 'lr': 0.1 * args.optim_lr},
        {'params': model.decoder.parameters(), 'lr': args.optim_lr},
    ], lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    if args.stage == "test":
        args.max_epochs=0


    mean_dice, std_dice = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return mean_dice, std_dice


if __name__ == "__main__":
    main()
