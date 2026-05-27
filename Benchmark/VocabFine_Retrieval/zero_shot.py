from pathlib import Path
from shutil import rmtree
from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel

from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from data_inference import CTReportDatasetinfer
import numpy as np
import tqdm
import pandas as pd

from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import math
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from ct_clip import CTCLIP
import nibabel as nib
import copy
import re

# helpers


def auc(pred_property_array, one_hot_labels, num_classes):
    AUROCs = []
    for i in range(num_classes):
        AUROCs.append(roc_auc_score(one_hot_labels[:, i], pred_property_array[:, i]))
    return AUROCs


def tensor_to_nifti(tensor, path, affine=np.eye(4)):
    """
    Save tensor as a NIfTI file.

    Args:
        tensor (torch.Tensor): The input tensor with shape (D, H, W) or (C, D, H, W).
        path (str): The path to save the NIfTI file.
        affine (np.ndarray, optional): The affine matrix for the NIfTI file. Defaults to np.eye(4).
    """

    tensor = tensor.cpu()

    if tensor.dim() == 4:
        # Assume single channel data if there are multiple channels
        if tensor.size(0) != 1:
            print("Warning: Saving only the first channel of the input tensor")
        tensor = tensor.squeeze(0)
    tensor=tensor.swapaxes(0,2)
    numpy_data = tensor.detach().numpy().astype(np.float32)
    nifti_img = nib.Nifti1Image(numpy_data, affine)
    nib.save(nifti_img, path)

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=-1)
    softmax_array = softmax(array)
    return softmax_array


class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.gamma = gamma
        self.T_cur = 0
        self.lr_min = 0
        self.iteration = 0

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.T_warmup:
            lr = self.eta_max * self.iteration / self.T_warmup
        else:
            self.T_cur = self.iteration - self.T_warmup
            T_i = self.T_0
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                T_i *= self.T_mult
                self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
            lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
                 (1 + math.cos(math.pi * self.T_cur / T_i))

        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._update_lr()
        self._update_T()

    def _update_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]

    def _update_T(self):
        if self.T_cur == self.T_0:
            self.T_cur = 0
            self.lr_min = 0
            self.iteration = 0
            self.T_0 *= self.T_mult
            self.eta_max *= self.gamma


def split_sentences(report):
    report = report.replace('\n', ' ')
    sents = re.split(r'[.;]\s*', report)
    sents = [s.strip() for s in sents if len(s.strip()) > 0]
    return sents


class CTClipInference(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        args,
        *,
        num_train_steps,
        batch_size,
        data_folder: "external_valid",
        reports_file: "data_reports.xslx",
        lr = 1e-4,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 100,
        save_model_every = 2000,
        results_folder = './results',
        labels = "labels.csv",
        accelerate_kwargs: dict = dict(),
        is_vocabfine=False,
        is_retrieval=False,
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.results_folder = results_folder
        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        self.is_vocabfine = is_vocabfine
        self.is_retrieval = is_retrieval

        if not self.is_retrieval and not self.is_vocabfine:
            raise ValueError("Either is_retrieval or is_vocabfine must be True.")

        all_parameters = set(CTClip.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        self.lr=lr
        # Load the pre-trained weights
        self.ds = CTReportDatasetinfer(args=args, data_folder=data_folder, csv_file=reports_file, mode="test", labels=labels)

        # Split dataset into train and validation sets


        self.dl = DataLoader(
            self.ds,
            num_workers=8,
            batch_size=self.batch_size,
            shuffle = False,
        )
        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)
        self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optim,
                                                  T_0=4000000,    # Maximum number of iterations
                                                  T_warmup=10000, # Number of warmup steps
                                                  eta_max=lr)   # Maximum learning rate


        (
 			self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.result_folder_txt = self.results_folder
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)



    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        CTClip = self.accelerator.unwrap_model(self.CTClip)
        CTClip.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())


        # logs
        logs = {}

        if self.is_vocabfine:
            with torch.no_grad():

                models_to_evaluate = ((self.CTClip, str(steps)),)

                for model, filename in models_to_evaluate:
                    model.eval()
                    predictedall=[]
                    realall=[]
                    logits = []

                    # text_latent_list = []
                    # image_latent_list = []
                    accession_names=[]
                    pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
                    for i, batch in tqdm.tqdm(enumerate(self.dl)):
                        # import ipdb; ipdb.set_trace()
                        valid_data, text, onehotlabels, acc_name = batch
                        
                        valid_data = copy.deepcopy(valid_data) 
                        text = copy.deepcopy(text) 
                        onehotlabels = copy.deepcopy(onehotlabels) 
                        acc_name = copy.deepcopy(acc_name)

                        # text_tokens=self.tokenizer(
                        #                 list(text), return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                        # text_latents, image_latents = model(text_tokens, valid_data.cuda(),  device=device, return_latents=True)                        
                        # text_latent_list.append(text_latents.detach().cpu().numpy())
                        # image_latent_list.append(image_latents.detach().cpu().numpy())
                        
                        plotdir = self.result_folder_txt
                        Path(plotdir).mkdir(parents=True, exist_ok=True)


                        B, L = onehotlabels.shape

                        texts = []
                        for l in range(L):
                            texts.append(f"{pathologies[l]}.")
                            texts.append(f"not {pathologies[l]}.")

                        text_tokens=self.tokenizer(
                                        texts, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

                        text_feature, image_feature = model(text_tokens, valid_data.cuda(), device=device)

                        text_feature = F.normalize(text_feature, dim=-1)
                        image_feature = F.normalize(image_feature, dim=-1)

                        output = image_feature @ text_feature.T  # (B, 2L)
                        output = output.view(B, L, 2)
                        output = apply_softmax(output)

                        append_out=output.detach().cpu().numpy()

                        predictedall.append(append_out)
                        realall.append(onehotlabels.detach().cpu().numpy())
                        accession_names.append(acc_name)

                    realall=np.concatenate(realall, axis=0)
                    predictedall=np.concatenate(predictedall, axis=0)

                    # np.savez(f"{plotdir}/labels_weights.npz", data=realall)
                    # np.savez(f"{plotdir}/predicted_weights.npz", data=predictedall)


                    # np.savez(f"{plotdir}/text_latents.npz", data=np.array(text_latent_list))
                    # np.savez(f"{plotdir}/image_latents.npz", data=np.array(image_latent_list))                  
                    
                    # with open(f"{plotdir}/accessions.txt", "w") as file:
                    #     for item in accession_names:
                    #         file.write(item + "\n")


                    pos_scores = predictedall[:, :, 0] - predictedall[:, :, 1]
                    aucs = auc(pos_scores, realall, len(pathologies))
                    dfs=evaluate_internal(pos_scores, realall, pathologies, plotdir)

                    dfs.to_csv(f'{plotdir}/aurocs.csv', index=False)

                    avg_auc = dfs.mean(axis=1)
                    print("Average test auc: ", avg_auc[0])


        if self.is_retrieval:
            with torch.no_grad():

                models_to_evaluate = ((self.CTClip, str(steps)),)

                for model, filename in models_to_evaluate:
                    model.eval()
                    predictedall=[]
                    realall=[]
                    logits = []

                    text_latent_list = []
                    image_latent_list = []
                    accession_names=[]
                    
                    sentence_nums = []
                    sentence_latent_list = []

                    for i, batch in tqdm.tqdm(enumerate(self.dl)):
                        # import ipdb; ipdb.set_trace()
                        valid_data, text, onehotlabels, acc_name = batch
                        
                        valid_data = copy.deepcopy(valid_data) 
                        text = copy.deepcopy(text) 
                        onehotlabels = copy.deepcopy(onehotlabels) 
                        acc_name = copy.deepcopy(acc_name)

                        text_tokens=self.tokenizer(
                                        list(text), return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                        text_latents, image_latents = model(text_tokens, valid_data.cuda(),  device=device)                        
                        text_latent_list.append(text_latents.detach().cpu().numpy())
                        image_latent_list.append(image_latents.detach().cpu().numpy())
                        
                        
                        for report_text in text:
                            sentences = split_sentences(report_text)
                            sentence_tokens = self.tokenizer(
                                        sentences, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                            input_image = torch.zeros(size=valid_data[0].shape).to(device)
                            sentence_latents, _ = model(sentence_tokens, input_image, device=device)
                            sentence_latent_list.append(sentence_latents.detach().cpu().numpy())
                            sentence_nums.append(len(sentence_latents))

                        accession_names.append(acc_name)
                        

                        plotdir = self.result_folder_txt
                        Path(plotdir).mkdir(parents=True, exist_ok=True)

                    sent2report = np.concatenate([np.full(count, i) for i, count in enumerate(sentence_nums)])

                    text_features_all=np.concatenate(text_latent_list, axis=0)
                    image_features_all=np.concatenate(image_latent_list, axis=0)
                    sentence_features_all = np.concatenate(sentence_latent_list, axis=0)

                    similarity_matrix = image_features_all @ text_features_all.T
                    # np.savez(f"{plotdir}/labels_weights.npz", data=realall)
                    # np.savez(f"{plotdir}/predicted_weights.npz", data=predictedall)


                    # np.savez(f"{plotdir}/text_latents.npz", data=np.array(text_latent_list))
                    # np.savez(f"{plotdir}/image_latents.npz", data=np.array(image_latent_list))                  
                    
                    # with open(f"{plotdir}/accessions.txt", "w") as file:
                    #     for item in accession_names:
                    #         file.write(item + "\n")

                    # image to text retrieval
                    N = similarity_matrix.shape[0]
                    ranks = np.argsort(-similarity_matrix, axis=1)

                    recall_at_1 = np.mean(ranks[:, 0] == np.arange(N))
                    recall_at_5 = np.mean([1 if np.arange(N)[i] in ranks[i, :5] else 0 for i in range(N)])
                    recall_at_10 = np.mean([1 if np.arange(N)[i] in ranks[i, :10] else 0 for i in range(N)])
                    recall_at_50 = np.mean([1 if np.arange(N)[i] in ranks[i, :50] else 0 for i in range(N)])
                    i2t_recall_at_50 = str(recall_at_50)
                    # ranks_pos = [
                    #     np.where(ranks[i] == i)[0][0] + 1
                    #     for i in range(N)
                    # ]

                    # medR = np.median(ranks_pos)
                    print("Volume to report retrieval: ")
                    print("Recall@1: ", recall_at_1, "Recall@5: ", recall_at_5, "Recall@10: ", recall_at_10, "Recall@50: ", recall_at_50)

                    sim_T = similarity_matrix.T
                    ranks_T = np.argsort(-sim_T, axis=1)

                    recall_at_1 = np.mean(ranks_T[:, 0] == np.arange(N))
                    recall_at_5 = np.mean([1 if np.arange(N)[i] in ranks_T[i, :5] else 0 for i in range(N)])
                    recall_at_10 = np.mean([1 if np.arange(N)[i] in ranks_T[i, :10] else 0 for i in range(N)])
                    recall_at_50 = np.mean([1 if np.arange(N)[i] in ranks_T[i, :50] else 0 for i in range(N)])
                    t2i_recall_at_50 = str(recall_at_50)
                    print("Report to volume retrieval: ")
                    print("Recall@1: ", recall_at_1, "Recall@5: ", recall_at_5, "Recall@10: ", recall_at_10, "Recall@50: ", recall_at_50)


                    similarity_matrix_sentence = image_features_all @ sentence_features_all.T

                    hits = []
                    for i in range(similarity_matrix_sentence.shape[0]):
                        ranked = np.argsort(-similarity_matrix_sentence[i])[:50]
                        hits.append(np.any(sent2report[ranked] == i))
                    recall_at_50 = np.mean(hits)
                    i2s_recall_at_50 = str(recall_at_50)
                    print("Volume to sentence retrieval: ")
                    print("Recall@50: ", recall_at_50)


                    similarity_matrix_sentence_T = sentence_features_all @ image_features_all.T
                    hits = []
                    for i in range(similarity_matrix_sentence_T.shape[0]):
                        ranked = np.argsort(-similarity_matrix_sentence_T[i])[:50]
                        hits.append(np.any(ranked == sent2report[i]))
                    recall_at_50 = np.mean(hits)
                    s2i_recall_at_50 = str(recall_at_50)
                    print("Sentence to volume retrieval: ")
                    print("Recall@50: ", recall_at_50)

                    
                    with open(f"{plotdir}/log.txt", "a") as file:
                        file.write("\n\n\n")
                        file.write(f"image to report Recall@50: {i2t_recall_at_50}, report to image Recall@50: {t2i_recall_at_50}, ")
                        file.write(f"image to sentence Recall@50: {i2s_recall_at_50}, sentence to image Recall@50: {s2i_recall_at_50}")
        
        
        self.steps += 1
        return logs




    def infer(self, log_fn=noop):
        device = next(self.CTClip.parameters()).device
        device=torch.device('cuda')
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('Inference complete')
