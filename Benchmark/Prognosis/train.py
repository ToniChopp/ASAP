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
            if args.name == "m3d":
                checkpoint = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
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
            elif args.name == "merlin":
                pass
            elif args.name == "visdboost":
                pass
            else:
                checkpoint = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
                checkpoint_model = checkpoint['model']
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
    
    all_surv_h = []
    all_surv_duration = []
    all_surv_event = []
    # all_readmit_h = []
    # all_readmit_duration = []
    # all_readmit_event = []
    # all_ph_h = []
    # all_ph_duration = []
    # all_ph_event = []

    loss_fct = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(epoch_iterator):
        # if step > 10:  # debug code 
        #     break
        x, targets = batch
        x = x.to(args.device)
        targets = {task: t.to(args.device) for task, t in targets.items()}
        with torch.no_grad():
            logits = model(x)

            surv_h = logits['survival']
            # read_h = logits['readmit']
            # ph_h = logits['ph']
            
            # logits_value = torch.concat((surv_h, read_h, ph_h.unsqueeze(-1)), dim=-1)
            logits_value = surv_h
            loss_mortality = cox_nll(surv_h, targets["survival_time_idx"], targets["survival_event"])
            # loss_readmit = cox_nll(read_h, targets["readmit_time_idx"], targets["readmit_event"])
            loss_bce = loss_fct(logits_value, targets["data_value"].float())
            # eval_loss = 2 * loss_mortality + 2 * loss_readmit + loss_bce
            eval_loss = 2 * loss_mortality + loss_bce
            eval_losses.update(eval_loss.item())


        if len(all_property) == 0:
            all_label.append(targets["data_value"].detach().cpu().numpy())
            all_property.append(logits_value.sigmoid().detach().cpu().numpy())
            all_surv_h.append(surv_h.detach().cpu().numpy())
            all_surv_duration.append(targets["survival_time_idx"].detach().cpu().numpy())
            all_surv_event.append(targets["survival_event"].detach().cpu().numpy())
            # all_readmit_h.append(read_h.detach().cpu().numpy())
            # all_readmit_duration.append(targets["readmit_time_idx"].detach().cpu().numpy())
            # all_readmit_event.append(targets["readmit_event"].detach().cpu().numpy())
            # all_ph_h.append(ph_h.unsqueeze(-1).detach().cpu().numpy())
            # all_ph_duration.append(targets["ph_time_idx"].detach().cpu().numpy())
            # all_ph_event.append(targets["ph_event"].detach().cpu().numpy())
        else:
            all_label[0] = np.append(
                all_label[0], targets["data_value"].detach().cpu().numpy(), axis=0
            )
            all_property[0] = np.append(
                all_property[0], logits_value.sigmoid().detach().cpu().numpy(), axis=0
            )
            all_surv_h[0] = np.append(
                all_surv_h[0], surv_h.detach().cpu().numpy(), axis=0
            )
            all_surv_duration[0] = np.append(
                all_surv_duration[0], targets["survival_time_idx"].detach().cpu().numpy(), axis=0
            )
            all_surv_event[0] = np.append(
                all_surv_event[0], targets["survival_event"].detach().cpu().numpy(), axis=0
            )
            # all_readmit_h[0] = np.append(
            #     all_readmit_h[0], read_h.detach().cpu().numpy(), axis=0
            # )
            # all_readmit_duration[0] = np.append(
            #     all_readmit_duration[0], targets["readmit_time_idx"].detach().cpu().numpy(), axis=0
            # )
            # all_readmit_event[0] = np.append(
            #     all_readmit_event[0], targets["readmit_event"].detach().cpu().numpy(), axis=0
            # )
            # all_ph_h[0] = np.append(
            #     all_ph_h[0], ph_h.unsqueeze(-1).detach().cpu().numpy(), axis=0
            # )
            # all_ph_duration[0] = np.append(
            #     all_ph_duration[0], targets["ph_time_idx"].detach().cpu().numpy(), axis=0
            # )
            # all_ph_event[0] = np.append(
            #     all_ph_event[0], targets["ph_event"].detach().cpu().numpy(), axis=0
            # )
        

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    
    all_label, all_property = all_label[0], all_property[0]
    all_surv_h, all_surv_duration, all_surv_event = all_surv_h[0], all_surv_duration[0], all_surv_event[0]
    # all_readmit_h, all_readmit_duration, all_readmit_event = all_readmit_h[0], all_readmit_duration[0], all_readmit_event[0]
    # all_ph_h, all_ph_duration, all_ph_event = all_ph_h[0], all_ph_duration[0], all_ph_event[0]
    results = {}
    results['cindex_survival'] = compute_c_index(torch.tensor(all_surv_h), torch.tensor(all_surv_duration), torch.tensor(all_surv_event))
    # results['cindex_readmit'] = compute_c_index(torch.tensor(all_readmit_h), torch.tensor(all_readmit_duration), torch.tensor(all_readmit_event))
    # results['cindex_ph'] = compute_c_index(torch.tensor(all_ph_h), torch.tensor(all_ph_duration), torch.tensor(all_ph_event))
    aurocs = auc(all_property, all_label, len(task_names))
    auroc_avg = np.array(aurocs).mean()
    eval_result = auroc_avg


    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Average AUROC: %2.5f" % eval_result)
    logger.info("Valid C-Index survival: %2.5f" % results['cindex_survival'])
    # logger.info("Valid C-Index readmit: %2.5f" % results['cindex_readmit'])
    # logger.info("Valid C-Index ph: %2.5f" % results['cindex_ph'])

    writer.add_scalar("valid/loss", scalar_value=eval_losses.avg, global_step=global_step)

    return eval_result, eval_losses.avg, results['cindex_survival']


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
    
    all_surv_h = []
    all_surv_duration = []
    all_surv_event = []
    # all_readmit_h = []
    # all_readmit_duration = []
    # all_readmit_event = []
    # all_ph_h = []
    # all_ph_duration = []
    # all_ph_event = []
    
    loss_fct = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(epoch_iterator):
        x, targets = batch
        x = x.to(args.device)
        targets = {task: t.to(args.device) for task, t in targets.items()}
        with torch.no_grad():
            logits = model(x)
            surv_h = logits['survival']
            # read_h = logits['readmit']
            # ph_h = logits['ph']

            # logits_value = torch.concat((surv_h, read_h, ph_h.unsqueeze(-1)), dim=-1)
            logits_value = surv_h
            loss_mortality = cox_nll(surv_h, targets["survival_time_idx"], targets["survival_event"])
            # loss_readmit = cox_nll(read_h, targets["readmit_time_idx"], targets["readmit_event"])
            loss_bce = loss_fct(logits_value, targets["data_value"].float())
            eval_loss = 2 * loss_mortality + loss_bce
            
            eval_losses.update(eval_loss.item())

            
        if len(all_property) == 0:
            all_label.append(targets["data_value"].detach().cpu().numpy())
            all_property.append(logits_value.sigmoid().detach().cpu().numpy())
            all_surv_h.append(surv_h.detach().cpu().numpy())
            all_surv_duration.append(targets["survival_time_idx"].detach().cpu().numpy())
            all_surv_event.append(targets["survival_event"].detach().cpu().numpy())
            # all_readmit_h.append(read_h.detach().cpu().numpy())
            # all_readmit_duration.append(targets["readmit_time_idx"].detach().cpu().numpy())
            # all_readmit_event.append(targets["readmit_event"].detach().cpu().numpy())
            # all_ph_h.append(ph_h.unsqueeze(-1).detach().cpu().numpy())
            # all_ph_duration.append(targets["ph_time_idx"].detach().cpu().numpy())
            # all_ph_event.append(targets["ph_event"].detach().cpu().numpy())
        else:
            all_label[0] = np.append(
                all_label[0], targets["data_value"].detach().cpu().numpy(), axis=0
            )
            all_property[0] = np.append(
                all_property[0], logits_value.sigmoid().detach().cpu().numpy(), axis=0
            )
            all_surv_h[0] = np.append(
                all_surv_h[0], surv_h.detach().cpu().numpy(), axis=0
            )
            all_surv_duration[0] = np.append(
                all_surv_duration[0], targets["survival_time_idx"].detach().cpu().numpy(), axis=0
            )
            all_surv_event[0] = np.append(
                all_surv_event[0], targets["survival_event"].detach().cpu().numpy(), axis=0
            )
            # all_readmit_h[0] = np.append(
            #     all_readmit_h[0], read_h.detach().cpu().numpy(), axis=0
            # )
            # all_readmit_duration[0] = np.append(
            #     all_readmit_duration[0], targets["readmit_time_idx"].detach().cpu().numpy(), axis=0
            # )
            # all_readmit_event[0] = np.append(
            #     all_readmit_event[0], targets["readmit_event"].detach().cpu().numpy(), axis=0
            # )
            # all_ph_h[0] = np.append(
            #     all_ph_h[0], ph_h.unsqueeze(-1).detach().cpu().numpy(), axis=0
            # )
            # all_ph_duration[0] = np.append(
            #     all_ph_duration[0], targets["ph_time_idx"].detach().cpu().numpy(), axis=0
            # )
            # all_ph_event[0] = np.append(
            #     all_ph_event[0], targets["ph_event"].detach().cpu().numpy(), axis=0
            # )
        
        epoch_iterator.set_description("Testing... (loss=%2.5f)" % eval_losses.val)

    all_label, all_property = all_label[0], all_property[0]
    all_surv_h, all_surv_duration, all_surv_event = all_surv_h[0], all_surv_duration[0], all_surv_event[0]
    # all_readmit_h, all_readmit_duration, all_readmit_event = all_readmit_h[0], all_readmit_duration[0], all_readmit_event[0]
    # all_ph_h, all_ph_duration, all_ph_event = all_ph_h[0], all_ph_duration[0], all_ph_event[0]
    results = {}
    results['cindex_survival'] = compute_c_index(torch.tensor(all_surv_h), torch.tensor(all_surv_duration), torch.tensor(all_surv_event))
    # results['cindex_readmit'] = compute_c_index(torch.tensor(all_readmit_h), torch.tensor(all_readmit_duration), torch.tensor(all_readmit_event))
    # results['cindex_ph'] = compute_c_index(torch.tensor(all_ph_h), torch.tensor(all_ph_duration), torch.tensor(all_ph_event))
    aurocs = auc(all_property, all_label, len(task_names))
    auroc_avg = np.array(aurocs).mean()
    eval_result = auroc_avg

    logger.info("\n")
    logger.info("Test Results")
    logger.info("Crop ratio: %0.4f" % args.ratio)
    logger.info("Test Loss: %2.5f" % eval_losses.avg)
    logger.info("Test C-Index survival: %2.5f" % results['cindex_survival'])
    # logger.info("Test C-Index readmit: %2.5f" % results['cindex_readmit'])
    # logger.info("Test C-Index ph: %2.5f" % results['cindex_ph'])
    wandb.log({"Test/C-Index survival": results['cindex_survival'], "Test/AUROC": eval_result, "Test/Loss": eval_losses.avg})
    logger.info('The average AUROC is {auroc_avg:.5f}'.format(auroc_avg=auroc_avg))
    for i in range(len(task_names)):
        logger.info('The AUROC of {} is {}'.format(task_names[i], aurocs[i]))

    return results['cindex_survival'], aurocs


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
            targets = {task: t.to(args.device) for task, t in targets.items()}
            
            logits = model(x)

            with autocast(device_type="cuda", 
                          dtype=torch.bfloat16, 
                          enabled=args.fp16):
                # logits_value = torch.concat((logits["survival"], logits["readmit"], logits["ph"].unsqueeze(-1)), dim=-1)
                logits_value = logits["survival"]
                loss_mortality = cox_nll(logits["survival"], targets["survival_time_idx"], targets["survival_event"])
                # loss_readmit = cox_nll(logits["readmit"], targets["readmit_time_idx"], targets["readmit_event"])
                # loss_ph = cox_nll(logits["ph"].unsqueeze(-1), targets["ph_time_idx"], targets["ph_event"])
                loss_bce = loss_fct(logits_value, targets["data_value"].float())

                loss = 2 * loss_mortality + loss_bce

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
                    eval_result, val_loss, val_cindex = valid(args, model, writer, val_loader, global_step)
                    if args.is_multilabel:
                        writer.add_scalar("auroc", scalar_value=eval_result, global_step=global_step)
                        wandb.log({"valid/auroc": eval_result, "valid/step": global_step, "valid/C-Index": val_cindex})
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
    parser.add_argument('--model', choices=['vit_tiny_patch16', 'vit_base_patch16', 'vit_large_patch16', 'vit_large_patch32', 'resnet152', 'resnet18'],
                        default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")

    parser.add_argument("--stage", type=str, default="train", help="train or test?")
    
    parser.add_argument("--task", choices=["INSPECT"],
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

    test_c_indexes = []
    test_aucs_1m = []
    test_aucs_6m = []
    test_aucs_12m = []

    # run five times and calculate mean & std
    run_five = False
    if run_five:
        seeds = [42, 1824, 409, 1732, 920]
    else:
        seeds = [42]

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
        test_c_index, test_auc = test(args_run)
        test_c_indexes.append(test_c_index)
        test_aucs_1m.append(test_auc[0])
        test_aucs_6m.append(test_auc[1])
        test_aucs_12m.append(test_auc[2])

    print("#########################")
    print("####### Final Results #######")
    print("Test C-Index over 3 runs: ", test_c_indexes)
    print("1m Test AUC over 3 runs: ", test_aucs_1m)
    print("6m Test AUC over 3 runs: ", test_aucs_6m)
    print("12m Test AUC over 3 runs: ", test_aucs_12m)
    print("Average Test C-Index: ", np.mean(test_c_indexes))
    print("Std of Test C-Index: ", np.std(test_c_indexes))
    print("Average 1m Test AUC: ", np.mean(test_aucs_1m))
    print("Std of 1m Test AUC: ", np.std(test_aucs_1m))
    print("Average 6m Test AUC: ", np.mean(test_aucs_6m))
    print("Std of 6m Test AUC: ", np.std(test_aucs_6m))
    print("Average 12m Test AUC: ", np.mean(test_aucs_12m))
    print("Std of 12m Test AUC: ", np.std(test_aucs_12m))
    wandb.log({"Final AVG C-Index": np.mean(test_c_indexes), "Final Std C-Index": np.std(test_c_indexes)})
    wandb.log({"Final 1m AVG AUC": np.mean(test_aucs_1m), "Final 1m Std AUC": np.std(test_aucs_1m)})
    wandb.log({"Final 6m AVG AUC": np.mean(test_aucs_6m), "Final 6m Std AUC": np.std(test_aucs_6m)})
    wandb.log({"Final 12m AVG AUC": np.mean(test_aucs_12m), "Final 12m Std AUC": np.std(test_aucs_12m)})
    log_handler.close()


if __name__ == "__main__":
    main()
