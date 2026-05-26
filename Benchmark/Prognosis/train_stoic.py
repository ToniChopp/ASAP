# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.nn as nn

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy('file_system')

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
from sklearn import metrics

from models_prognosis import CoxViT
from timm.models.layers import trunc_normal_

from lifelines.utils import concordance_index

import ipdb
import wandb
from copy import deepcopy

logger = logging.getLogger(__name__)


task_names = [
    '1m_mort', '6m_mort', '12m_mort',
    # '1m_readmit', '6m_readmit', '12m_readmit',
    # '12m_ph'
]
TIME_MAP = np.array([1., 6., 12.], dtype=float)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def auc(pred_property_array, one_hot_labels, num_classes):
    AUROCs = []
    for i in range(num_classes):
        AUROCs.append(roc_auc_score(one_hot_labels[:, i], pred_property_array[:, i]))
    return AUROCs


def cox_nll(hazards, time_idx, event):
    """
    hazards: [B, T], 每个时间点的 hazard logit (未sigmoid)
    time_idx: [B], 事件发生的时间点 (0 ~ T-1)，或删失时间点
    event: [B], 事件是否发生 (1=发生, 0=删失)
    """
    # 转换为概率
    hazards = torch.sigmoid(hazards)   # [B, T]

    # 累积生存概率
    survival = torch.cumprod(1 - hazards, dim=1)  # [B, T]

    # 事件发生概率 = hazard * survival(到该时刻前)
    event_prob = hazards * torch.cat(
        [torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1
    )  # [B, T]

    # 取事件发生时间
    b_idx = torch.arange(hazards.size(0))
    likelihood = event_prob[b_idx, time_idx] ** event * survival[b_idx, time_idx] ** (1 - event)

    nll = -torch.log(likelihood + 1e-8).mean()
    return nll


def compute_c_index(logits, durations, events):
    """
    logits: [B, T] (未过sigmoid)
    durations: [B]  事件或删失时间 (int, 0~T-1)
    events: [B]     是否发生事件 (1/0)
    """
    # 累积 hazard 作为 risk score
    risk_scores = logits.sum(dim=1).detach().cpu().numpy()
    durations = durations.detach().cpu().numpy()
    events = events.detach().cpu().numpy()

    return concordance_index(durations, -risk_scores, events)


def simple_accuracy(preds, labels):
    # print(preds)
    # print(labels)
    return ((preds == labels) * 1).mean()


def classification_report(preds, labels):
    return metrics.classification_report(labels,preds)


def save_model_auc(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_bestauc_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def save_model_acc(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, args.name + "_bestacc_checkpoint.bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def load_weights(model, weight_path, args):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    if args.stage=='train':
        pretrained_weights = pretrained_weights['model']
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}
    print(len(load_weights))
    # print("load weights")
    # for k, _ in load_weights.items():
    #     print(k)

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    return model


def setup(args):
    
    # Prepare model
    if args.task == "CC-CCII":
        args.is_multilabel = False
    else:
        args.is_multilabel = True

    
    model = CoxViT(
        out_dim=768,
        model=args.name,
        task=args.task,
    )
    if args.stage=='train':
        if args.pretrained_path != "random":
            checkpoint = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
            if args.name != "m3d":
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            state_dict = model.state_dict()
            matched_keys = {}
            for k in checkpoint_model.keys():
                if ("backbone." + k) in state_dict:
                    matched_keys["backbone." + k] = checkpoint_model[k]
            # load pre-trained model
            msg = model.load_state_dict(matched_keys, strict=False)
            print(msg)

        # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.projection_head[0].weight, std=2e-5)
        trunc_normal_(model.survival_head.weight, std=2e-5)
        # trunc_normal_(model.readmit_head.weight, std=2e-5)
        # trunc_normal_(model.ph_head.weight, std=2e-5)

        for name, param in model.named_parameters():
            if name.startswith("backbone"):
                param.requires_grad = False

    else:
        if args.is_multilabel:
            args.pretrained_path = os.path.join(args.output_dir, "%s_bestauc_checkpoint.bin" % args.name)
        else:
            args.pretrained_path = os.path.join(args.output_dir, "%s_bestacc_checkpoint.bin" % args.name)
        model = load_weights(model, args.pretrained_path, args)

    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.5fM" % num_params)
    return args, model
    

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])


    all_label = []
    all_property = []

    loss_fct = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(epoch_iterator):
        # if step > 10:  # debug code 
        #     break
        x, targets = batch
        x = x.to(args.device)
        targets = targets.to(args.device)
        with torch.no_grad():
            logits = model(x)

            loss_bce = loss_fct(logits, targets.float())
            eval_loss = loss_bce
            eval_losses.update(eval_loss.item())


        if len(all_property) == 0:
            all_label.append(targets.detach().cpu().numpy())
            all_property.append(logits.sigmoid().detach().cpu().numpy())
        else:
            all_label[0] = np.append(
                all_label[0], targets.detach().cpu().numpy(), axis=0
            )
            all_property[0] = np.append(
                all_property[0], logits.sigmoid().detach().cpu().numpy(), axis=0
            )
        

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    
    all_label, all_property = all_label[0], all_property[0]
    aurocs = auc(all_property, all_label, num_classes=all_property.shape[1])
    auroc_avg = np.array(aurocs).mean()
    eval_result = auroc_avg


    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Average AUROC: %2.5f" % eval_result)

    writer.add_scalar("valid/loss", scalar_value=eval_losses.avg, global_step=global_step)

    return eval_result, eval_losses.avg


def test(args):
    # Test!
    eval_losses = AverageMeter()

    args.stage = 'test'
    args, model = setup(args)
    test_loader = get_loader(args)

    # logger.info("***** Running Test *****")
    # logger.info("  Num steps = %d", len(test_loader))
    # logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    
    epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    
    all_label = []
    all_property = []
    
    
    loss_fct = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(epoch_iterator):
        x, targets = batch
        x = x.to(args.device)
        targets = targets.to(args.device)
        with torch.no_grad():
            logits = model(x)
            
            loss_bce = loss_fct(logits, targets.float())
            eval_loss = loss_bce
            
            eval_losses.update(eval_loss.item())

            
        if len(all_property) == 0:
            all_label.append(targets.detach().cpu().numpy())
            all_property.append(logits.sigmoid().detach().cpu().numpy())
        else:
            all_label[0] = np.append(
                all_label[0], targets.detach().cpu().numpy(), axis=0
            )
            all_property[0] = np.append(
                all_property[0], logits.sigmoid().detach().cpu().numpy(), axis=0
            )
        
        epoch_iterator.set_description("Testing... (loss=%2.5f)" % eval_losses.val)

    all_label, all_property = all_label[0], all_property[0]
    aurocs = auc(all_property, all_label, num_classes=all_property.shape[1])
    auroc_avg = np.array(aurocs).mean()
    eval_result = auroc_avg

    logger.info("\n")
    logger.info("Test Results")
    logger.info("Crop ratio: %0.4f" % args.ratio)
    logger.info("Test Loss: %2.5f" % eval_losses.avg)
    wandb.log({"Test/AUROC": eval_result, "Test/Loss": eval_losses.avg})
    logger.info('The average AUROC is {auroc_avg:.5f}'.format(auroc_avg=auroc_avg))

    return aurocs


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))  #  tensorboard Supporting documents, in logs/name/

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, val_loader = get_loader(args)
    
    # ipdb.set_trace()
    # import nibabel as nib
    # for item in train_loader:
    #     image = item[0]
    #     image_tosave = image[0].squeeze().numpy()
    #     nii_img = nib.Nifti1Image(image_tosave, np.eye(4))
    #     nib.save(nii_img, 'image.nii.gz')

    # Prepare optimizer and scheduler
    if args.is_multilabel:
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=args.learning_rate,
        #                             momentum=0.9,
        #                             weight_decay=args.weight_decay)
        # optimizer_head = torch.optim.SGD(model.head.parameters(),
        #                             lr=args.learning_rate,
        #                             momentum=0.9,
        #                             weight_decay=args.weight_decay)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    print(optimizer)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.local_rank != -1:
        model = torch.compile(model)
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    # ipdb.set_trace()

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_auc, best_acc = 0, 0, 0
    min_loss = 10000000
    down = 0
    epoch_cnt = 0

    patience = args.patience

    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        
        loss_fct = nn.BCEWithLogitsLoss()
        for step, batch in enumerate(epoch_iterator):
            x, targets = batch
            x = x.to(args.device)
            targets = targets.to(args.device)

            logits = model(x)

            with autocast(device_type="cuda", 
                          dtype=torch.bfloat16, 
                          enabled=args.fp16):

                loss = loss_fct(logits, targets.float())

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                    wandb.log({"train/loss": losses.val, "train/lr": scheduler.get_lr()[0], "train/step": global_step})
                
                len_train = len(train_loader)

                if global_step % len_train == 0 and args.local_rank in [-1, 0]:
                    epoch_cnt += 1
                if global_step % len_train == 0 and args.local_rank in [-1, 0] and epoch_cnt >= args.start_epoch:
                    eval_result, val_loss = valid(args, model, writer, val_loader, global_step)
                    if args.is_multilabel:
                        writer.add_scalar("auroc", scalar_value=eval_result, global_step=global_step)
                        wandb.log({"valid/auroc": eval_result, "valid/step": global_step})
                    else:
                        writer.add_scalar("accuracy", scalar_value=eval_result, global_step=global_step)

                    if args.is_multilabel:
                        if best_auc <= eval_result:
                            save_model_auc(args, model)
                            best_auc = eval_result
                            down = 0
                        else:
                            down = down + 1
                            print(down)
                    else:
                        if best_acc < eval_result:
                            save_model_acc(args, model)
                            best_acc = eval_result
                            down = 0
                        else:
                            down = down + 1
                            print(down)


        losses.reset()
        if global_step % t_total == 0 or down >= patience:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    
    torch.cuda.empty_cache()
    logger.info("min_Loss: \t%f" % min_loss)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vit_tiny_patch16', 'vit_base_patch16', 'vit_large_patch16', 'vit_large_patch32'],
                        default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")

    parser.add_argument("--stage", type=str, default="train", help="train or test?")
    
    parser.add_argument("--task", choices=["CT-Rate", "RadChestCT", "CC-CCII", "LUNA16", "INSPECT", "INSPECT_filtered", "Stoic"],
                        default="INSPECT", type=str,
                        help="Which finetune task to take.")
    parser.add_argument("--pretrained_path", type=str, default="vit_base_patch16.pth",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

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
    parser.add_argument("--RandFlip_prob", type=float, default=0.2, help="probability of RandFlip")
    parser.add_argument("--RandShiftIntensity_prob", type=float, default=0.1, help="probability of RandShiftIntensity")
    parser.add_argument("--RandGaussianNoise_prob", type=float, default=0.2, help="probability of RandGaussianNoise")

    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")               
    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--data_volume", type=str)
    parser.add_argument("--gpu", type=str, default="7")

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--start_epoch", type=int, default=10)

    args = parser.parse_args()

    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    wandb.init(
        project=f"Prognose_{args.task}",
        name=args.output_dir.split('/')[-2] + "/" + args.output_dir.split('/')[-1],
        config={
            "lr": args.learning_rate,
            "batch_size": args.train_batch_size,
            "warmup_steps": args.warmup_steps,
            "max_steps": args.num_steps,
        }
    )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        print('##############################')   
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                            timeout=timedelta(minutes=60)
                                            )
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    log_handler = logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    test_aucs = []
    seeds = [42, 1824, 409]

    for seed_id in range(len(seeds)):
        args_run = deepcopy(args)
        args_run.seed = seeds[seed_id]

        print("#########################")
        print("####### Seed is %d #######" % args_run.seed)
        print("#########################")

        # Training
        if args_run.stage == "train":
            args_run, model = setup(args_run)
            train(args_run, model)
        # Testing
        test_auc = test(args_run)
        test_aucs.append(test_auc)

    print("#########################")
    print("####### Final Results #######")
    print("Test AUC over 3 runs: ", test_aucs)
    print("Average Test AUC: ", np.mean(test_aucs))
    print("Std of Test AUC: ", np.std(test_aucs))
    wandb.log({"Final AVG AUC": np.mean(test_aucs), "Final 12m Std AUC": np.std(test_aucs)})
    log_handler.close()


if __name__ == "__main__":
    main()
