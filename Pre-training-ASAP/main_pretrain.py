# Some codes are borrowed from MAE and MRM-pytorch
import argparse
import datetime
import json
import shutil
import numpy as np
import os
import time
from datetime import datetime
import yaml
from pathlib import Path
from typing import Iterable

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DistributedSampler, DataLoader
from torch.optim import AdamW


import timm.optim.optim_factory as optim_factory
import wandb

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched

from module.model_asap import asap
from module.pretrain_datasets import ASAPDataset

import ipdb


def dump(file_path: str, args):
    r"""Save config at the specified file path.

    Parameters
    ----------
    file_path: str
        (YAML) path to save config at.
    """
    yaml.dump(stream=open(file_path, 'w'), data=args)


# --------------------------------------------------------
# References:
# MRM-pytorch: https://github.com/RL4M/MRM-pytorch
# --------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('ASAP pre-training', add_help=False)
    parser.add_argument('--description', type=str, default='asap_pretrain')
    parser.add_argument('--batch_size', default=14, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=130, type=int)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='asap', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./dataset', type=str, help='dataset path')
    parser.add_argument('--dataset_path', default='../../../Data/CT-Rate', type=str, help='volume path')
    parser.add_argument('--checkpoint_path', default='./checkpoints', type=str, help='checkpoint path')
    parser.add_argument('--max_caption_length', default=512, type=int,
                        help='max caption length')
    parser.add_argument("--spacing_x", type=float, default=1.0, help="spacing in x direction")
    parser.add_argument("--spacing_y", type=float, default=1.0, help="spacing in y direction")
    parser.add_argument("--spacing_z", type=float, default=1.0, help="spacing in z direction")
    parser.add_argument("--a_min", type=float, default=-1000.0, help="minimum value of intensity")
    parser.add_argument("--a_max", type=float, default=1000.0, help="maximum value of intensity")
    parser.add_argument("--b_min", type=float, default=-1.0, help="minimum value of intensity after ScaleIntensityRange")
    parser.add_argument("--b_max", type=float, default=1.0, help="maximum value of intensity after ScaleIntensityRange")
    parser.add_argument("--roi_x", type=int, default=224, help="roi size in x direction")
    parser.add_argument("--roi_y", type=int, default=224, help="roi size in y direction")
    parser.add_argument("--roi_z", type=int, default=112, help="roi size in z direction")
    parser.add_argument("--roi_scale", type=float, default=0.8, help="scale of the roi")
    parser.add_argument("--RandFlip_prob", type=float, default=0.2, help="probability of RandFlip")
    parser.add_argument("--RandShiftIntensity_prob", type=float, default=0.1, help="probability of RandShiftIntensity")
    
    # training settings
    parser.add_argument('--output_dir', default='./output_pretrain',
                        help='path where to save, empty for no saving')
    parser.add_argument('--job_dir', default='code_repo',
                        help='path where to save the codes')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint or pretrained visual encoder')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch, default as 0')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    use_wandb = True,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 40

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        with torch.cuda.amp.autocast():
            mim_loss, injection_loss, mlm_loss, align_loss = model(batch)
            
            loss = mim_loss + mlm_loss + align_loss + injection_loss

            mim_loss_item = mim_loss.item()
            mlm_loss_item = mlm_loss.item()
            align_loss_item = align_loss.item()
            injection_loss_item = injection_loss.item()

            loss = loss / accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(mim_loss=mim_loss_item)
            metric_logger.update(mlm_loss=mlm_loss_item)
            metric_logger.update(align_loss=align_loss_item)
            metric_logger.update(injection_loss=injection_loss_item)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce0 = misc.all_reduce_mean(mim_loss_item)
            loss_value_reduce1 = misc.all_reduce_mean(mlm_loss_item)
            loss_value_reduce2 = misc.all_reduce_mean(align_loss_item)
            loss_value_reduce3 = misc.all_reduce_mean(injection_loss_item)
            if use_wandb and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                wandb.log({'mim_loss': loss_value_reduce0, 'mlm_loss': loss_value_reduce1, \
                           'align_loss': loss_value_reduce2, 'injection_loss': loss_value_reduce3, \
                            'lr': lr}, step=epoch * len(data_loader) + data_iter_step)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    misc.init_distributed_mode(args)

    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = ASAPDataset(args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    use_wandb = False
    args.log_dir = os.path.join(args.output_dir, "wandb_log")
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        wandb.init(
            project=f"{args.model}",
            name=f"{args.description}",
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "warmup_epochs": args.warmup_epochs,
                "max_epochs": args.epochs,
            },
            dir=args.log_dir,
        )
        use_wandb = True
        shutil.copytree('./', os.path.join(args.output_dir, args.job_dir), dirs_exist_ok=False)
        print("Finish logging")


    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=dataset_train.collate_fn
    )

    # define the model
    model = asap(args.norm_pix_loss, args.mask_ratio)

    model.to(device)
    
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()


    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)


    optimizer = AdamW(param_groups, betas=(0.9, 0.95))


    print(optimizer)
    loss_scaler = NativeScaler()

    dump(os.path.join(args.output_dir, 'config.yaml'), args)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    start_datetime = datetime.now()
    write_description = False
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            use_wandb=use_wandb,
            args=args
        )

        if args.output_dir:
            if epoch < 40:
                if epoch == 0:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch
                    )
            elif epoch < 100:
                if epoch % 20 == 0:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch
                    )
            else:
                if (epoch % 10 == 0 or epoch + 1 == args.epochs):
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch
                    )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                if not write_description:
                    f.write(start_datetime.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                    f.write(args.description + "\n")
                    write_description = True
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
