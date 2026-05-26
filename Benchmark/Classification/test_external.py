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

import models_vit
from timm.models.layers import trunc_normal_

import ipdb

logger = logging.getLogger(__name__)


CT_RATE_CLASS_NAMES = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',\
                       'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema',\
                       'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',\
                       'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',\
                        'Interlobular septal thickening']

AHPH_10K_CLASS_NAMES = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',\
                       'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema',\
                       'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',\
                       'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',\
                        'Interlobular septal thickening']

RADCHESTCT_CLASS_NAMES = ["Medical material", "Calcification", "Cardiomegaly", "Pericardial effusion", "Hietal Hernia", \
                          "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity", \
                          "Pulmonary fibrotic sequela", "Pleural effusion", "Peribronchial thickening", "Consolidation", \
                          "Bronchiectasis", "Interlobular septal thickening"]



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
    args.is_multilabel = True

    num_classes = args.num_classes
    
    model = models_vit.__dict__[args.model](
        num_classes=num_classes,
        drop_path_rate=0,
        global_pool=True,
    )

    checkpoint = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
    checkpoint_model = checkpoint

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    for name, param in model.named_parameters():
        param.requires_grad = False

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
    all_preds, all_label = [], []
    all_property = []
    epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    
    if args.is_multilabel:
        loss_fct = nn.BCEWithLogitsLoss()
    else:
        loss_fct = nn.CrossEntropyLoss()

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            if args.is_multilabel:
                eval_loss = loss_fct(logits, y.float())
            else:
                y = y.squeeze()
                y = y.long()
                eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            
            if args.is_multilabel:
                preds = (logits.sigmoid() > 0.5) * 1
            else:
                preds = torch.argmax(logits, dim=-1)
            
        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            if args.is_multilabel:
                all_property.append(logits.sigmoid().detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            if args.is_multilabel:
                all_property[0] = np.append(
                    all_property[0], logits.sigmoid().detach().cpu().numpy(), axis=0
                )
        epoch_iterator.set_description("Testing... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    aurocs = []
    auroc_avg = 0

    if args.is_multilabel:
        all_property = all_property[0]
        accuracy = simple_accuracy(all_preds, all_label)
        aurocs = auc(all_property, all_label, args.num_classes)
        auroc_avg = np.array(aurocs).mean()
        micro_f1 = f1_score(all_label, all_preds, average='micro')
        macro_f1 = f1_score(all_label, all_preds, average='macro')
        precision = metrics.precision_score(all_label, all_preds, average='micro')
    else:
        accuracy = simple_accuracy(all_preds, all_label)
        Confusion = metrics.confusion_matrix(all_label, all_preds)
        print(Confusion)

    logger.info("\n")
    logger.info("Test Results")
    logger.info("Crop ratio: %0.4f" % args.ratio)
    logger.info("Test Loss: %2.5f" % eval_losses.avg)
    logger.info("Test Accuracy: %2.5f" % accuracy)
    logger.info("Test Micro F1: %2.5f" % micro_f1)
    logger.info("Test Macro F1: %2.5f" % macro_f1)
    if args.is_multilabel:
        logger.info("Test Precision: %2.5f" % precision)
        logger.info('The average AUROC is {auroc_avg:.5f}'.format(auroc_avg=auroc_avg))
        for i in range(args.num_classes):
            if args.task.startswith("CT-Rate"):
                logger.info(f'The AUROC of {AHPH_10K_CLASS_NAMES[i]} is {aurocs[i]:.5f}')
            elif args.task.startswith("AHPH-10K"):
                logger.info(f'The AUROC of {AHPH_10K_CLASS_NAMES[i]} is {aurocs[i]:.5f}')
        print(aurocs)


def test_external_radchestct(args):
    # Test!
    eval_losses = AverageMeter()

    args.stage = 'test'
    args.is_multilabel = True
    num_classes = args.num_classes
    model = models_vit.__dict__[args.model](
        num_classes=num_classes,
        drop_path_rate=0.1,
        global_pool=True,
    )
    model = load_weights(model, args.pretrained_path, args)
    model.to(args.device)
    test_loader = get_loader(args)
    # logger.info("***** Running Test *****")
    # logger.info("  Num steps = %d", len(test_loader))
    # logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    all_property = []
    epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    
    if args.is_multilabel:
        loss_fct = nn.BCEWithLogitsLoss()
    else:
        loss_fct = nn.CrossEntropyLoss()
    
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(torch.float16).to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            logits = torch.cat((logits[:, :13], logits[:, 14:]), dim=1)
            y_temp = torch.cat((y[:, :4], y[:, 1].unsqueeze(-1)), dim=1)
            y = torch.cat((y_temp, y[:, 4:]), dim=1)
            if args.is_multilabel:
                eval_loss = loss_fct(logits, y.float())
            else:
                y = y.squeeze()
                y = y.long()
                eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            
            if args.is_multilabel:
                preds = (logits.sigmoid() > 0.5) * 1
            else:
                preds = torch.argmax(logits, dim=-1)
            
        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            if args.is_multilabel:
                all_property.append(logits.sigmoid().detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            if args.is_multilabel:
                all_property[0] = np.append(
                    all_property[0], logits.sigmoid().detach().cpu().numpy(), axis=0
                )
        epoch_iterator.set_description("Testing... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0].astype(np.int64)
    aurocs = []
    auroc_avg = 0

    if args.is_multilabel:
        all_property = all_property[0]
        aurocs = auc(all_property, all_label, args.num_classes - 1)
        if aurocs[1] > aurocs[4]:   # Arterial wall calcification is better
            aurocs = aurocs[:4] + aurocs[5:]
            all_preds = np.concatenate((all_preds[:, :4], all_preds[:, 5:]), axis=1)
            all_label = np.concatenate((all_label[:, :4], all_label[:, 5:]), axis=1)
        else:   # Coronary artery wall calcification is better
            aurocs = aurocs[:1] + aurocs[2:]
            all_preds = np.concatenate((all_preds[:, :1], all_preds[:, 2:]), axis=1)
            all_label = np.concatenate((all_label[:, :1], all_label[:, 2:]), axis=1)
        auroc_avg = np.array(aurocs).mean()
        accuracy = simple_accuracy(all_preds, all_label)
        micro_f1 = f1_score(all_label, all_preds, average='micro')
        macro_f1 = f1_score(all_label, all_preds, average='macro')
        precision = metrics.precision_score(all_label, all_preds, average='micro')
    else:
        accuracy = simple_accuracy(all_preds, all_label)
        Confusion = metrics.confusion_matrix(all_label, all_preds)
        print(Confusion)

    logger.info("\n")
    logger.info("Test Results")
    logger.info("Crop ratio: %0.4f" % args.ratio)
    logger.info("Test Loss: %2.5f" % eval_losses.avg)
    logger.info("Test Accuracy: %2.5f" % accuracy)
    logger.info("Test Micro F1: %2.5f" % micro_f1)
    logger.info("Test Macro F1: %2.5f" % macro_f1)
    if args.is_multilabel:
        logger.info("Test Precision: %2.5f" % precision)
        logger.info('The average AUROC is {auroc_avg:.5f}'.format(auroc_avg=auroc_avg))
        if args.task.startswith("RadChestCT_external"):
            for i in range(16):
                logger.info('The AUROC of {} is {}'.format(RADCHESTCT_CLASS_NAMES[i], aurocs[i]))

        print(aurocs[:16])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vit_tiny_patch16', 'vit_base_patch16', 'vit_large_patch16', 'vit_large_patch32'],
                        default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")

    parser.add_argument("--stage", type=str, default="train", help="train or test?")
    
    parser.add_argument("--task", choices=["CT-Rate", "RadChestCT", "CC-CCII", "LUNA16", "CT-Rate_O", "CC-CCII_new", "INSPECT", "RSPECT", "Stoic", "AHPH-10K"],
                        default="CT-Rate",
                        help="Which finetune task to take.")
    parser.add_argument("--num_classes",default = 14, type=int, help="the number of class")
    parser.add_argument("--pretrained_path", type=str, default="checkpoint/ViT-B_16.npz",
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
    parser.add_argument("--patchsize_x", type=int, default=16, help="patch size in x direction")
    parser.add_argument("--patchsize_y", type=int, default=16, help="patch size in y direction")
    parser.add_argument("--patchsize_z", type=int, default=8, help="patch size in z direction")
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

    if args.task == "RSPECT" or args.task == "AHPH-10K":
        test(args)
    elif args.task == "RadChestCT_external":
        test_external_radchestct(args)

    log_handler.close()


if __name__ == "__main__":
    main()
