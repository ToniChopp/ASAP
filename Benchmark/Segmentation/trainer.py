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

import os
import shutil
import time
import datetime
import logging

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
import wandb
from torch.cuda.amp import GradScaler, autocast
from utils.utils import distributed_all_gather
from tqdm import tqdm

from monai.data import decollate_batch
from monai.transforms import AsDiscrete
import ipdb


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
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
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in tqdm(enumerate(loader)):

        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]

        data, target = data.cuda(args.rank), target.cuda(args.rank)
        # if args.model_name == "swinunetr":
        #         data = data.permute(0, 1, 4, 3, 2)
        for param in model.parameters():
            param.grad = None
            
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
    if args.rank == 0:
        time_current = datetime.datetime.now()
        print(
            time_current.strftime("%Y-%m-%d %H:%M:%S"),
            "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
    start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    start_time = time.time()
    val_dice = []
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)

            # if args.model_name == "swinunetr":
            #     data = data.permute(0, 1, 4, 3, 2)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            if args.task != "COVID19-20Cases_lung":
                acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            # else:
            #     acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            #     acc = acc[:, :2]
            else:
                post_pred_local = AsDiscrete(argmax=True, to_onehot=3, n_classes=3)
                post_label_local = AsDiscrete(to_onehot=3, n_classes=3)
                
                probs = torch.softmax(logits, dim=1)
                probs[:, 0] = probs[:, 0] + probs[:, 3]
                probs = probs[:, :3]
                
                # target 中的标签3需要先映射为0（背景），再做 to_onehot
                target_remapped = target.clone()
                target_remapped[target_remapped == 3] = 0  # 第4类→背景

                val_labels_list = decollate_batch(target_remapped)
                val_labels_convert = [post_label_local(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(probs)
                val_output_convert = [post_pred_local(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            
            acc = acc.cuda(args.rank)


            if args.distributed:
                acc_list = distributed_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            else:
                acc_list = acc.detach().cpu().numpy()
                # print(acc_list)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])



            if epoch == 0:
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])
                val_dice.append(avg_acc)
                time_current = datetime.datetime.now()
                print(
                    time_current.strftime("%Y-%m-%d %H:%M:%S"),
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
                continue

            if args.rank == 0:
                time_current = datetime.datetime.now()
                print(
                    time_current.strftime("%Y-%m-%d %H:%M:%S"),
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            val_dice.append(avg_acc)
            start_time = time.time()
    if epoch == 0:
        return np.mean(val_dice), np.std(val_dice)   
    return np.mean(val_dice)


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.output_dir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
    wandb_init=False,
):
    writer = None

    # if not wandb_init and args.rank == 0:
    #     wandb.init(
    #         project=f"Segmentation_{args.task}",
    #         name=args.output_dir.split('/')[-2] + "/" + args.output_dir.split('/')[-1],
    #         config={
    #             "lr": args.optim_lr,
    #             "num_samples": args.num_samples,
    #             "batch_size": args.batch_size,
    #             "warmup_epochs": args.warmup_epochs,
    #             "max_epochs": args.max_epochs,
    #         }
    #     )

    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    if args.stage == "test":
        test_avg_dice, test_std_dice = val_epoch(
            model,
            val_loader,
            epoch=0,
            acc_func=acc_func,
            model_inferer=model_inferer,
            args=args,
            post_label=post_label,
            post_pred=post_pred,
        )
        print("Test AVG DICE:", test_avg_dice)
        print("Test STD DICE:", test_std_dice)
        wandb.log({"Test Average Dice": test_avg_dice, "Test Std Dice": test_std_dice})
        return test_avg_dice, test_std_dice
    
    cnt = 0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Step:", epoch)
        epoch_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )

        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        # if args.rank == 0 and writer is not None:
        #     writer.add_scalar("train_loss", train_loss, epoch)
        if args.rank == 0:
            wandb.log({"Train Loss": train_loss, "Epoch": epoch})
        b_new_best = False
        if (epoch + 1) % args.val_every == 0 and epoch >= args.start_val_epochs:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                # if writer is not None:
                #     writer.add_scalar("val_acc", val_avg_acc, epoch)
                wandb.log({"Validation Dice": val_avg_acc, "Epoch": epoch})
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    cnt = 0
                    if args.rank == 0 and args.output_dir is not None:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler, filename=f"model_{args.seed}.pt"
                        )
                else:
                    cnt = cnt + 1
                    print("no improvement in {} epochs".format(cnt))
            
                if cnt >= args.patience:
                    print("Early stopping at epoch {}".format(epoch))
                    break
            if args.rank == 0 and args.output_dir is not None:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)
    val_acc_std = 0.0

    return val_acc_max, val_acc_std
