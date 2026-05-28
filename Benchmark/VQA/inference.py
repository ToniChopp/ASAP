'''
Author: xm_cmic
Date: 2024-05-03 10:28:14
LastEditors: xm_cmic
LastEditTime: 2024-06-01 10:45:37
FilePath: /src-0515/global_inference.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

import os
import re
import csv 
import json 
import numpy as np
import pandas as pd 
import nibabel as nib

import torch
import transformers
import torch.nn.functional as F
from transformers import Trainer 
from transformers import GPT2Tokenizer
from safetensors.torch import save_file, load_file

from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
from peft import PeftModel
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model,AutoPeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig


from model.global_model import Global_VQA_Model

from monai.transforms import (
    Resized,
    Compose
)
from monai.transforms.utility.dictionary import AddChannelD, ToTensorD
from monai.transforms.spatial.dictionary import OrientationD, SpacingD, RandRotate90D, RandFlipD
from monai.transforms.croppad.dictionary import RandSpatialCropD, SpatialPadD, RandScaleCropD, CenterSpatialCropD
from monai.transforms.intensity.dictionary import RandShiftIntensityD, ScaleIntensityRangeD, RandScaleIntensityD, RandGaussianNoiseD

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from bert_score import BERTScorer

import ipdb


@dataclass
class DataArguments:
    # grounded_report_csv,qa_csv,qa_size_csv,mask_root_dir,anatomy_mask_root_dir
    Mode: Optional[str] = field(default="Train")
    train_image_path_csv: Optional[str] = field(default='./radgenome_files/subset_2000/train_image_path.csv')
    test_image_path_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_image_path.csv')
    
    train_regioned_report_csv: Optional[str] = field(default='./radgenome_files/subset_2000/train_region_report.csv')
    test_regioned_report_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv')
    
    train_qa_abnormality_csv: Optional[str] = field(default='./radgenome_files/subset_2000/train_vqa_abnormality.csv')
    test_qa_abnormality_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_vqa_abnormality.csv')
    
    train_qa_location_csv: Optional[str] = field(default='./radgenome_files/subset_2000/train_vqa_location.csv')
    test_qa_location_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_vqa_location.csv')
    
    train_qa_presence_csv: Optional[str] = field(default='./radgenome_files/subset_2000/train_vqa_presence.csv')
    test_qa_presence_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_vqa_presence.csv')
    
    train_qa_size_csv: Optional[str] = field(default='./radgenome_files/subset_2000/train_vqa_size.csv')
    test_qa_size_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_vqa_size.csv')
    
    train_disorders_csv: Optional[str] = field(default='./radgenome_files/subset_2000/train_case_disorders.csv')
    test_disorders_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_case_disorders.csv')
    
    train_mask_root_dir: Optional[str] = field(default='./RadGenome-ChestCT/dataset//train_region_mask')
    test_mask_root_dir: Optional[str] = field(default='./RadGenome-ChestCT/dataset//valid_region_mask')
    train_anatomy_mask_root_dir: Optional[str] = field(default='./RadGenome-ChestCT/dataset//train_anatomy_mask')
    test_anatomy_mask_root_dir: Optional[str] = field(default='./RadGenome-ChestCT/dataset//valid_anatomy_mask')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    per_device_train_batch_size: int = field(default = 1)
    per_device_eval_batch_size: int = field(default = 1)
    gradient_accumulation_steps: int = field(default = 1)
    # eval_accumulation_steps: int = field(default = 8)
    output_dir: Optional[str] = field(default="./results/llama3_global")
    num_train_epochs: int = field(default = 10)
    save_total_limit: int = field(default = 3)
    evaluation_strategy: Optional[str] = field(default="no")
    save_strategy: Optional[str] = field(default="steps")
    task: Optional[str] = field(default="report")
    save_steps: int = field(default = 100)
    logging_steps: int = field(default = 1)
    lora_rank: int = field(default = 8)
    warmup_steps: int = field(default = 500)
    weight_ratio: float = field(default = 0.03)
    weight_decay: float = field(default = 0.00)
    learning_rate: float = field(default = 2e-5)
    optim: str = field(default="adamw_torch")
    lr_scheduler_type: str = field(default="cosine")
    gradient_checkpointing: bool = field(default=False)
    save_on_each_node: bool = field(default = True)
    vision_learnable: bool = field(default = True)
    
@dataclass
class ModelArguments:
    llm_max_length: int = field(default = 384)
    text_dim: int = field(default = 2560)
    language_backbone: str = field(default = 'Qwen3')
    vision_backbone: str = field(default = 'ViT')
    vision_pretrained: Optional[str] = field(default=None)
    
    
def find_max_checkpoint_folder(path):
    # 获取路径下所有的文件夹和文件
    entries = os.listdir(path)
    
    # 过滤出所有以 "checkpoint-" 开头的文件夹
    checkpoint_folders = [folder for folder in entries if folder.startswith("checkpoint-") and os.path.isdir(os.path.join(path, folder))]
    
    # 提取文件夹名称中的数字，并找出最大的数字
    max_num = -1
    max_folder = None
    for folder in checkpoint_folders:
        try:
            num = int(folder.split("-")[1])  # 分割字符串并转换成整数
            if num > max_num:
                max_num = num
                max_folder = folder
        except ValueError:
            # 如果转换失败，忽略这个文件夹
            continue
    
    return os.path.join(path,max_folder)


def zscore_process(image):
    '''
        zscore normalization
    '''
    lower_bound, upper_bound = -1000, 1000
    image = np.clip(image, lower_bound, upper_bound)
    image = image.astype(np.float32)
    b_min, b_max = -1, 1
    i_min, i_max = image.min(), image.max()
    if i_max - i_min > 0:
        image = (image - i_min) / (i_max - i_min) * (b_max - b_min) + b_min
    else:
        image = np.zeros_like(image)
    return image


def read_nii_img(nii_path):
    try:
        # nii_path = "./" + nii_path
        img = nib.load(nii_path)
        img_array = np.asanyarray(img.dataobj)
    except FileNotFoundError:
        print(f"Warning: File not found at {nii_path}. Returning a random array instead.")
        img_array = np.random.rand(224, 224, 112)  # Generates random floats
    except Exception as e:
        print(f"An error occurred while loading the NIfTI file: {e}")
        img_array = np.random.rand(224, 224, 112)  # Backup in case of other exceptions
    return img_array


def parse_abnormality(text):
    match = re.search(r"<Abnormality>\s*(.*)", str(text))
    if match:
        text = match.group(1)
    else:
        text = str(text)
    return text


def normalize(text):

    text = text.lower().strip()

    text = text.replace(".", "")
    text = text.replace(",", "")

    return text


def parse_presence(text):
    match = re.search(r"<Presence>\s*(.*?)\.", str(text))
    if match:
        return match.group(1).strip().lower()
    return ""


def parse_location(text):
    match = re.search(r"<Location>\s*(.*?)\.", str(text))
    if match:
        return match.group(1).strip().lower()
    return ""


def hitscore(pred, gt):

    pred = pred.lower()
    gt = gt.lower()

    if pred == gt:
        return 1

    pred_parts = pred.split("/")
    gt_parts = gt.split("/")

    overlap = len(set(pred_parts) & set(gt_parts))

    return overlap / max(len(pred_parts), len(gt_parts))


def parse_size(size_str):
    """
    Examples:
    "6x4 mm" -> [6, 4]
    "16 mm" -> [16]
    "1.2 x 0.8 cm" -> [12, 8]
    "0.5 m" -> [500]
    """

    # 统一格式（避免大小写、空格问题）
    s = size_str.lower().replace(" ", "")

    # 匹配：数字 + 可选单位
    matches = re.findall(r"(\d+\.?\d*)(mm|cm|m)?", s)

    # 默认单位（如果只写一次，比如 6x4 mm）
    default_unit_match = re.search(r"(mm|cm|m)", s)
    default_unit = default_unit_match.group(1) if default_unit_match else "mm"

    unit_scale = {
        "mm": 1,
        "cm": 10,
        "m": 1000,
        None: None  # 占位，后面处理
    }

    if not matches:
        return [0]

    result = []
    for value, unit in matches:
        value = float(value)

        # 如果当前数字没有单位，用默认单位
        if unit is None:
            unit = default_unit

        scale = unit_scale.get(unit, 1)
        result.append(value * scale)

    return result


def l1_size(pred, gt):
    """
    计算 size L1
    """
    pred_vals = parse_size(pred)
    gt_vals = parse_size(gt)

    # 如果维度不同，取最大值作为代表
    if len(pred_vals) != len(gt_vals):
        pred_vals = [max(pred_vals)]
        gt_vals = [max(gt_vals)]

    pred_vals = np.array(pred_vals)
    gt_vals = np.array(gt_vals)

    return np.mean(np.abs(pred_vals - gt_vals))

    

if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments,DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    torch.autograd.set_detect_anomaly(True)
    
    print("Setup Model")
    if model_args.language_backbone == 'LLaMA3':
        tokenizer = AutoTokenizer.from_pretrained('./Model/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif model_args.language_backbone == 'Qwen3':
        tokenizer = AutoTokenizer.from_pretrained('./model/Qwen3-4B-Instruct-2507', use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained('./Model/MMedLM2-1_8B', use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.language_backbone == 'LLaMA3':
        language_model = AutoModelForCausalLM.from_pretrained('./Model/Meta-Llama-3-8B-Instruct', device_map="cpu",torch_dtype=torch.bfloat16)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=True,
            r=training_args.lora_rank, # Lora 秩
            lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.1# Dropout 比例
        )
        language_model = get_peft_model(language_model, config)
    elif model_args.language_backbone == "Qwen3":
        language_model = AutoModelForCausalLM.from_pretrained('./model/Qwen3-4B-Instruct-2507', device_map="cpu",torch_dtype=torch.bfloat16)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # target_modules=["q_proj", "v_proj"],
            inference_mode=True,
            r=training_args.lora_rank, # Lora 秩
            lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.0# Dropout 比例
        )
        language_model = get_peft_model(language_model, config)
        for param in language_model.parameters():
            param.requires_grad = False
    else:
        language_model = AutoModelForCausalLM.from_pretrained('./Model/MMedLM2-1_8B', device_map="cpu",torch_dtype=torch.bfloat16,trust_remote_code=True)
        for param in language_model.parameters():
            param.requires_grad = False
        

    print("Start Load Model")
    # ckp_dir = find_max_checkpoint_folder(training_args.output_dir) 
    ckp = training_args.output_dir + '/pytorch_model.bin'

    model = Global_VQA_Model(tokenizer,
                             language_model, 
                             vision_backbone=model_args.vision_backbone, 
                             vision_pretrained=model_args.vision_pretrained, 
                             vision_learnable=training_args.vision_learnable,
                             is_train=False
                            )
    checkpoint = torch.load(ckp, map_location="cpu")

    model.load_state_dict(checkpoint)#,strict=False)
          
    model = model.to('cuda')
    model.eval()
    print("Finished Load Model")
    
    # image_path_csv,regioned_report_csv,qa_abnormality_csv,qa_location_csv,qa_presence_csv,qa_size_csv,disorders_csv,mask_root_dir,anatomy_mask_root_dir 
    
    print("Load Test Data")
    report_df = pd.read_csv(data_args.test_image_path_csv)
    volume_to_nii_path_dict = dict(zip(report_df['Volumename'], report_df['nii_path']))
    
    # region_report_df = pd.read_csv(data_args.test_regioned_report_csv)
    # # 检查并填充NaN值
    # region_report_df['Anatomy'].fillna('whole scan', inplace=True)
    # # 提取region列的值
    # region_report_df['Region'] = region_report_df['Anatomy'].apply(lambda x: x.split('/')[0].strip().lower())
    
    qa_df_with_abnormalities = pd.read_csv(data_args.test_qa_abnormality_csv)
    qa_df_with_location = pd.read_csv(data_args.test_qa_location_csv)
    qa_df_with_presence = pd.read_csv(data_args.test_qa_presence_csv)
    qa_size_df = pd.read_csv(data_args.test_qa_size_csv)
    case_disorder_df = pd.read_csv(data_args.test_disorders_csv)
    
    abnormality_template_json = './data/ctrate/generate_data/qa_templates/abnormality_template.json'
    with open(abnormality_template_json, 'r') as file:
        abnormality_template = json.load(file)['abnormality']
    
    location_template_json = './data/ctrate/generate_data/qa_templates/location_template.json'
    with open(location_template_json, 'r') as file:
        location_template = json.load(file)['location']
    
    presence_template_json = './data/ctrate/generate_data/qa_templates/presence_template.json'
    with open(presence_template_json, 'r') as file:
        presence_template = json.load(file)['presence']
    
    size_template_json = './data/ctrate/generate_data/qa_templates/size_template.json'
    with open(size_template_json, 'r') as file:
        size_template = json.load(file)['size']
        
    disorder_template_json = './data/ctrate/generate_data/qa_templates/disorders_template.json'
    with open(disorder_template_json, 'r') as file:
        disorders_template = json.load(file)['disorders']
    
    augmentator = Compose(
            [
                AddChannelD(keys=["image"]),
                OrientationD(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangeD(
                    keys=["image"],
                    a_min=-1000,
                    a_max=1000,
                    b_min=-1,
                    b_max=1,
                    clip=True,
                ),
                CenterSpatialCropD(
                    keys=["image"],
                    roi_size=(224, 224, 112),
                ),
                SpatialPadD(
                    keys=["image"],
                    spatial_size=(224, 224, 112),
                    mode="reflect",
                ),
                ToTensorD(keys=["image"])
            ]
        )
    
    print('Start Testing')
    num_img_patches = 64
    
    
    if training_args.task == "abnormality":
        volume_list = qa_df_with_abnormalities['Volumename'].tolist()
        anatomy_list = qa_df_with_abnormalities['Anatomy'].tolist()
        abnormality_list = qa_df_with_abnormalities['Abnormality'].tolist()
        file_path = os.path.join(training_args.output_dir, 'subset_test_results_abnormality.csv')

        bertscorer = BERTScorer(
            model_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            num_layers=12,
            all_layers=True
        )

        if os.path.exists(file_path):
            # Read the existing file to determine where to start appending
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_data = list(reader)
                start_idx = len(existing_data) - 1  # Assuming the first row is the header

            if start_idx == len(qa_df_with_abnormalities):
                df = pd.read_csv(file_path)
                output_list = df['Output_result'].tolist()
                gt_list = df['GT_result'].tolist()
                gts = {}
                res = {}
                groundt_list = []
                pred_list = []
                hit_scores = []
                
                
                for i in tqdm(range(len(output_list))):
                    pred = normalize(parse_abnormality(output_list[i]))
                    gt = normalize(parse_abnormality(gt_list[i]))

                    gts[i] = gt
                    res[i] = pred

                    pred_list.append("".join(pred).strip())
                    groundt_list.append("".join(gt).strip())

                    hit_score = hitscore(pred, gt)
                    hit_scores.append(hit_score)

                bleu_scorer = Bleu(4)
                rouge_scorer = Rouge()
                bleu_score, bleu_scores = bleu_scorer.compute_score(
                    {k: [v] for k, v in gts.items()},
                    {k: [v] for k, v in res.items()},
                    verbose=0
                )
                rouge_score, rouge_scores = rouge_scorer.compute_score(
                    {k: [v] for k, v in gts.items()},
                    {k: [v] for k, v in res.items()})
                _, _, bert_score = bertscorer.score(groundt_list, pred_list)
                bert_score = bert_score[-1].mean().item()

                print("BLEU Score for Abnormality Prediction: ", bleu_score)
                print("ROUGE Score for Abnormality Prediction: ", rouge_score)
                print("BERTScore for Abnormality Prediction: ", bert_score)
                print("Average Hit Score for Abnormality Prediction: ", np.mean(hit_scores))
                exit()

        else:
            start_idx = 0  # Start from the beginning if the file does not exist

        
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if start_idx == 0:
                # Write the header if the file is new
                writer.writerow(['Volumename', 'Anatomy', 'Abnormality', 'nii_path', 'Output_result', 'GT_result'])
        
            for idx in tqdm(range(start_idx, int(len(qa_df_with_abnormalities)))):
                volume = volume_list[idx]
                anatomy = anatomy_list[idx]
                abnormality = abnormality_list[idx]
                is_nan = pd.isna(abnormality)
                if is_nan:
                    abnormality = 'No Findings'
                
                # process_image 
                image_path = volume_to_nii_path_dict[volume]
                img_array = read_nii_img(image_path)
                aug_data_dict = augmentator({'image': img_array})
                image = aug_data_dict['image']
                image = image.to('cuda')
                
                node_anatomy = anatomy.split('/')[-1]
                query_text = f'<Grounded abnormality detection>  What are the abnormalities in the {node_anatomy}?'
                input_text = f'<Grounded abnormality detection>  What are the abnormalities in the {node_anatomy}? <Abnormality> {abnormality}'# <|end_of_text|>'

                # Tokenize input text without adding special tokens
                tokenizer_output = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                input_ids = tokenizer_output['input_ids']

                # Copy input_ids to label and set eos_token_id locations to -100
                label = input_ids.clone()

                # Process query text if provided
                if query_text:
                    query_tokenizer_output = tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
                    query_input_ids = query_tokenizer_output['input_ids']
                    # Set the labels for the length of the query input ids to -100
                    label[:, :query_input_ids.size(1)] = -100

                # Convert input_ids to list and append eos_token_id, then pad to 960 (1024 - 64) as per your requirement
                input_ids = input_ids.squeeze().tolist() + [tokenizer.eos_token_id]
                input_ids = input_ids + [tokenizer.pad_token_id] * (model_args.llm_max_length - num_img_patches - len(input_ids)) # Padding input_ids

                # Convert label to list, append eos_token_id, and pad both to 1024
                label = label.squeeze().tolist() + [tokenizer.eos_token_id]
                label = [-100]*num_img_patches + label + [-100] * (model_args.llm_max_length - num_img_patches - len(label))  # Padding labels

                
                label = torch.tensor(np.array(label)).unsqueeze(0).to('cuda')
                input_ids = torch.tensor(np.array(input_ids)).unsqueeze(0).to('cuda')
                
                with torch.no_grad():
                    output_sentence = model.generate(input_sentence = query_text, input_image=image, input_ids=input_ids, labels=label)
                    
                    print('input_sentence:',query_text)
                    print('output_sentence:',output_sentence)
                    print('grounded_sentence:',input_text[len(query_text):])
                    gt_sentence = input_text[len(query_text):]
                    
                    # Write the current row to the CSV file
                    writer.writerow([volume, anatomy, abnormality, image_path, output_sentence, gt_sentence])
                    file.flush()
                
        df = pd.read_csv(file_path)
        output_list = df['Output_result'].tolist()
        gt_list = df['GT_result'].tolist()
        gts = {}
        res = {}
        groundt_list = []
        pred_list = []
        hit_scores = []
                
                
        for i in tqdm(range(len(output_list))):
            pred = normalize(parse_abnormality(output_list[i]))
            gt = normalize(parse_abnormality(gt_list[i]))
            gts[i] = gt
            res[i] = pred

            pred_list.append("".join(pred).strip())
            groundt_list.append("".join(gt).strip())

            hit_score = hitscore(pred, gt)
            hit_scores.append(hit_score)

        bleu_scorer = Bleu(4)
        rouge_scorer = Rouge()
        bleu_score, bleu_scores = bleu_scorer.compute_score(
            {k: [v] for k, v in gts.items()},
            {k: [v] for k, v in res.items()},
            verbose=0
        )
        rouge_score, rouge_scores = rouge_scorer.compute_score(
            {k: [v] for k, v in gts.items()},
            {k: [v] for k, v in res.items()})
        _, _, bert_score = bertscorer.score(groundt_list, pred_list)
        bert_score = bert_score[-1].mean().item()

        print("BLEU Score for Abnormality Prediction: ", bleu_score)
        print("ROUGE Score for Abnormality Prediction: ", rouge_score)
        print("BERTScore for Abnormality Prediction: ", bert_score)
        print("Average Hit Score for Abnormality Prediction: ", np.mean(hit_scores))

    
    
    elif training_args.task == "presence":
        abnormality_volume_list = qa_df_with_presence['Volumename'].tolist()
        abnormality_anatomy_list = qa_df_with_presence['Anatomy'].tolist()
        abnormality_abnormality_list = qa_df_with_presence['Finding'].tolist()
        abnormality_answer_list = qa_df_with_presence['Presence'].tolist()
        
        file_path = os.path.join(training_args.output_dir, 'subset_test_results_presence.csv')

        if os.path.exists(file_path):
            # Read the existing file to determine where to start appending
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_data = list(reader)
                start_idx = len(existing_data) - 1  # Assuming the first row is the header

            if start_idx == len(qa_df_with_presence):
                df = pd.read_csv(file_path)
                preds = df["Output_result"].apply(parse_presence)
                gts = df["GT_result"].apply(parse_presence)

                accuracy = (preds == gts).mean()
                print("Accuracy for Presence Prediction:", accuracy)
                exit()

        else:
            start_idx = 0  # Start from the beginning if the file does not exist


        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if start_idx == 0:
                # Write the header if the file is new
                writer.writerow(['Volumename', 'Anatomy', 'Abnormality', 'nii_path', 'Output_result', 'GT_result'])
        
            gt_list = []
            pred_list = []
            for idx in tqdm(range(start_idx, len(abnormality_volume_list))):
                volume = abnormality_volume_list[idx]
                anatomy = abnormality_anatomy_list[idx]
                abnormality = abnormality_abnormality_list[idx]
                answer = abnormality_answer_list[idx]
                
                # process_image 
                image_path = volume_to_nii_path_dict[volume]
                img_array = read_nii_img(image_path)
                aug_data_dict = augmentator({'image': img_array})
                image = aug_data_dict['image']
                image = image.to('cuda')
                
                node_anatomy = anatomy.split('/')[-1]
                query_text = f'<Grounded abnormality presence>  Is there any sign of {abnormality} in the {node_anatomy}?'
                input_text = f'<Grounded abnormality presence>  Is there any sign of {abnormality} in the {node_anatomy}? <Presence> {answer}'# <|end_of_text|>'

                # Tokenize input text without adding special tokens
                tokenizer_output = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                input_ids = tokenizer_output['input_ids']

                # Copy input_ids to label and set eos_token_id locations to -100
                label = input_ids.clone()

                # Process query text if provided
                if query_text:
                    query_tokenizer_output = tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
                    query_input_ids = query_tokenizer_output['input_ids']
                    # Set the labels for the length of the query input ids to -100
                    label[:, :query_input_ids.size(1)] = -100

                # Convert input_ids to list and append eos_token_id, then pad to 960 (1024 - 64) as per your requirement
                input_ids = input_ids.squeeze().tolist() + [tokenizer.eos_token_id]
                input_ids = input_ids + [tokenizer.pad_token_id] * (model_args.llm_max_length - num_img_patches - len(input_ids)) # Padding input_ids

                # Convert label to list, append eos_token_id, and pad both to 1024
                label = label.squeeze().tolist() + [tokenizer.eos_token_id]
                label = [-100]*num_img_patches + label + [-100] * (model_args.llm_max_length - num_img_patches - len(label))  # Padding labels
                
                label = torch.tensor(np.array(label)).unsqueeze(0).to('cuda')
                input_ids = torch.tensor(np.array(input_ids)).unsqueeze(0).to('cuda')
                
                with torch.no_grad():
                    output_sentence = model.generate(input_sentence = query_text, input_image=image, input_ids=input_ids, labels=label)
                    
                    # print('input_sentence:',query_text)
                    # print('output_sentence:',output_sentence)
                    # print('grounded_sentence:',input_text[len(query_text):])
                    gt_sentence = input_text[len(query_text):]
                    
                    # Write the current row to the CSV file
                    writer.writerow([volume, anatomy, abnormality, image_path, output_sentence, gt_sentence])
                    file.flush()

                    pred = parse_presence(output_sentence)
                    gt = parse_presence(gt_sentence)
                    pred_list.append(pred)
                    gt_list.append(gt)
            
            accuracy = (np.array(pred_list) == np.array(gt_list)).mean()
            print("Accuracy for Presence Prediction:", accuracy)
            
    
    elif training_args.task == "location":
        abnormality_volume_list = qa_df_with_location['Volumename'].tolist()
        abnormality_anatomy_list = qa_df_with_location['Anatomy'].tolist()
        abnormality_abnormality_list = qa_df_with_location['Abnormality'].tolist()
        
        bertscorer = BERTScorer(
            model_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            num_layers=12,
            all_layers=True
        )

        file_path = os.path.join(training_args.output_dir, 'subset_test_results_location.csv')

        if os.path.exists(file_path):
            # Read the existing file to determine where to start appending
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_data = list(reader)
                start_idx = len(existing_data) - 1  # Assuming the first row is the header
            
            if start_idx == len(qa_df_with_location):
                df = pd.read_csv(file_path)
                output_list = df['Output_result'].tolist()
                gt_list = df['GT_result'].tolist()
                gts = {}
                res = {}
                groundt_list = []
                pred_list = []
                hit_scores = []
                
                
                for i in tqdm(range(len(output_list))):
                    pred = normalize(parse_location(output_list[i]))
                    gt = normalize(parse_location(gt_list[i]))

                    gts[i] = gt
                    res[i] = pred

                    pred_list.append("".join(pred).strip())
                    groundt_list.append("".join(gt).strip())

                    hit_score = hitscore(pred, gt)
                    hit_scores.append(hit_score)

                bleu_scorer = Bleu(4)
                rouge_scorer = Rouge()
                bleu_score, bleu_scores = bleu_scorer.compute_score(
                    {k: [v] for k, v in gts.items()},
                    {k: [v] for k, v in res.items()},
                    verbose=0
                )
                rouge_score, rouge_scores = rouge_scorer.compute_score(
                    {k: [v] for k, v in gts.items()},
                    {k: [v] for k, v in res.items()})
                _, _, bert_score = bertscorer.score(groundt_list, pred_list)
                bert_score = bert_score[-1].mean().item()

                print("BLEU Score for Location Prediction: ", bleu_score)
                print("ROUGE Score for Location Prediction: ", rouge_score)
                print("BERTScore for Location Prediction: ", bert_score)
                print("Average Hit Score for Location Prediction: ", np.mean(hit_scores))
                exit()

        else:
            start_idx = 0  # Start from the beginning if the file does not exist


        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if start_idx == 0:
                # Write the header if the file is new
                writer.writerow(['Volumename', 'Anatomy', 'Abnormality', 'nii_path', 'Output_result', 'GT_result'])
        
            for idx in tqdm(range(start_idx, len(qa_df_with_location))):
                volume = abnormality_volume_list[idx]
                anatomy = abnormality_anatomy_list[idx]
                abnormality = abnormality_abnormality_list[idx]
                
                # process_image 
                image_path = volume_to_nii_path_dict[volume]
                img_array = read_nii_img(image_path)
                aug_data_dict = augmentator({'image': img_array})
                image = aug_data_dict['image']
                image = image.to('cuda')
                
                node_anatomy = anatomy.split('/')[-1]
                query_text = f'<Abnormality location>  Where is the {abnormality} located in the image?'
                input_text = f'<Abnormality location>  Where is the {abnormality} located in the image? <Location> {anatomy.capitalize()}.'

                # Tokenize input text without adding special tokens
                tokenizer_output = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                input_ids = tokenizer_output['input_ids']

                # Copy input_ids to label and set eos_token_id locations to -100
                label = input_ids.clone()

                # Process query text if provided
                if query_text:
                    query_tokenizer_output = tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
                    query_input_ids = query_tokenizer_output['input_ids']
                    # Set the labels for the length of the query input ids to -100
                    label[:, :query_input_ids.size(1)] = -100

                # Convert input_ids to list and append eos_token_id, then pad to 960 (1024 - 64) as per your requirement
                input_ids = input_ids.squeeze().tolist() + [tokenizer.eos_token_id]
                input_ids = input_ids + [tokenizer.pad_token_id] * (model_args.llm_max_length - num_img_patches - len(input_ids)) # Padding input_ids

                # Convert label to list, append eos_token_id, and pad both to 1024
                label = label.squeeze().tolist() + [tokenizer.eos_token_id]
                label = [-100]*num_img_patches + label + [-100] * (model_args.llm_max_length - num_img_patches - len(label))  # Padding labels
                
                label = torch.tensor(np.array(label)).unsqueeze(0).to('cuda')
                input_ids = torch.tensor(np.array(input_ids)).unsqueeze(0).to('cuda')
                
                with torch.no_grad():
                    output_sentence = model.generate(input_sentence = query_text, input_image=image, input_ids=input_ids, labels=label)
                    
                    print('input_sentence:',query_text)
                    print('output_sentence:',output_sentence)
                    print('grounded_sentence:',input_text[len(query_text):])
                    gt_sentence = input_text[len(query_text):]
                    
                    # Write the current row to the CSV file
                    writer.writerow([volume, anatomy, abnormality, image_path, output_sentence, gt_sentence])
                    file.flush()

            df = pd.read_csv(file_path)
            output_list = df['Output_result'].tolist()
            gt_list = df['GT_result'].tolist()
            gts = {}
            res = {}
            groundt_list = []
            pred_list = []
            hit_scores = []
                
                
            for i in tqdm(range(len(output_list))):
                pred = normalize(parse_location(output_list[i]))
                gt = normalize(parse_location(gt_list[i]))

                gts[i] = gt
                res[i] = pred

                pred_list.append("".join(pred).strip())
                groundt_list.append("".join(gt).strip())

                hit_score = hitscore(pred, gt)
                hit_scores.append(hit_score)

            bleu_scorer = Bleu(4)
            rouge_scorer = Rouge()
            bleu_score, bleu_scores = bleu_scorer.compute_score(
                {k: [v] for k, v in gts.items()},
                {k: [v] for k, v in res.items()},
                verbose=0
            )
            rouge_score, rouge_scores = rouge_scorer.compute_score(
                {k: [v] for k, v in gts.items()},
                {k: [v] for k, v in res.items()})
            _, _, bert_score = bertscorer.score(groundt_list, pred_list)
            bert_score = bert_score[-1].mean().item()

            print("BLEU Score for Location Prediction: ", bleu_score)
            print("ROUGE Score for Location Prediction: ", rouge_score)
            print("BERTScore for Location Prediction: ", bert_score)
            print("Average Hit Score for Location Prediction: ", np.mean(hit_scores))

 
    
    elif training_args.task == "size":
        volume_list = qa_size_df['Volumename'].tolist()
        anatomy_list = qa_size_df['Anatomy'].tolist()
        abnormality_list = qa_size_df['Abnormality'].tolist()
        size_list = qa_size_df['Size'].tolist()
        
        file_path = os.path.join(training_args.output_dir, 'subset_test_results_size.csv')

        if os.path.exists(file_path):
            # Read the existing file to determine where to start appending
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_data = list(reader)
                start_idx = len(existing_data) - 1  # Assuming the first row is the header

            if start_idx == len(qa_size_df):
                df = pd.read_csv(file_path)
                output_list = df['Output_result'].tolist()
                gt_list = df['GT_result'].tolist()
                L1_scores = []
                for i in tqdm(range(len(output_list))):
                    L1_score = l1_size(output_list[i], gt_list[i])
                    L1_scores.append(L1_score)
    
                print("Average L1 Score for Size Prediction: ", np.mean(L1_scores))
                rmse = np.sqrt(np.mean(np.array(L1_scores)**2))
                print("RMSE for Size Prediction: ", rmse)
                exit()

        else:
            start_idx = 0  # Start from the beginning if the file does not exist


        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if start_idx == 0:
                # Write the header if the file is new
                writer.writerow(['Volumename', 'Anatomy', 'Abnormality', 'nii_path', 'Output_result', 'GT_result'])
            L1_scores = []
            for idx in tqdm(range(start_idx, len(qa_size_df))):
                volume = volume_list[idx]
                anatomy = anatomy_list[idx]
                abnormality = abnormality_list[idx]
                size = size_list[idx]
                
                # process_image 
                image_path = volume_to_nii_path_dict[volume]
                img_array = read_nii_img(image_path)
                aug_data_dict = augmentator({'image': img_array})
                image = aug_data_dict['image']
                image = image.to('cuda')
                
                node_anatomy = anatomy.split('/')[-1]
                query_text = f'<Abnormality size prediction>  What is the approximate size of the {abnormality} in the {anatomy}?'
                input_text = f'<Abnormality size prediction>  What is the approximate size of the {abnormality} in the {anatomy}? <Size> {size}.'# <|end_of_text|>'

                # Tokenize input text without adding special tokens
                tokenizer_output = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                input_ids = tokenizer_output['input_ids']

                # Copy input_ids to label and set eos_token_id locations to -100
                label = input_ids.clone()

                # Process query text if provided
                if query_text:
                    query_tokenizer_output = tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
                    query_input_ids = query_tokenizer_output['input_ids']
                    # Set the labels for the length of the query input ids to -100
                    label[:, :query_input_ids.size(1)] = -100

                # Convert input_ids to list and append eos_token_id, then pad to 960 (1024 - 64) as per your requirement
                input_ids = input_ids.squeeze().tolist() + [tokenizer.eos_token_id]
                input_ids = input_ids + [tokenizer.pad_token_id] * (model_args.llm_max_length - num_img_patches - len(input_ids)) # Padding input_ids

                # Convert label to list, append eos_token_id, and pad both to 1024
                label = label.squeeze().tolist() + [tokenizer.eos_token_id]
                label = [-100]*num_img_patches + label + [-100] * (model_args.llm_max_length - num_img_patches - len(label))  # Padding labels
                
                label = torch.tensor(np.array(label)).unsqueeze(0).to('cuda')
                input_ids = torch.tensor(np.array(input_ids)).unsqueeze(0).to('cuda')
                
                with torch.no_grad():
                    output_sentence = model.generate(input_sentence = query_text, input_image=image, input_ids=input_ids, labels=label)
                    
                    print('input_sentence:',query_text)
                    print('output_sentence:',output_sentence)
                    print('grounded_sentence:',input_text[len(query_text):])
                    gt_sentence = input_text[len(query_text):]
                    
                    # Write the current row to the CSV file
                    writer.writerow([volume, anatomy, abnormality, image_path, output_sentence, gt_sentence])
                    file.flush()
                
                L1_score = l1_size(output_sentence, gt_sentence)
                L1_scores.append(L1_score)
            
        print("Average L1 Score for Size Prediction: ", np.mean(L1_scores))
        rmse = np.sqrt(np.mean(np.array(L1_scores)**2))
        print("RMSE for Size Prediction: ", rmse)
    
    
    elif training_args.task == "disorder":
        volume_list = case_disorder_df['Volumename'].tolist()
        disorders_list = case_disorder_df['Disorders'].tolist()
        # Open the CSV file for writing
        file_path = os.path.join(training_args.output_dir, 'subset_test_results_disorder.csv')

        bertscorer = BERTScorer(
            model_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            num_layers=12,
            all_layers=True
        )

        if os.path.exists(file_path):
            # Read the existing file to determine where to start appending
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_data = list(reader)
                start_idx = len(existing_data) - 1  # Assuming the first row is the header

            if start_idx == len(case_disorder_df):
                df = pd.read_csv(file_path)
                output_list = df['Output_result'].tolist()
                gt_list = df['GT_result'].tolist()
                gts = {}
                res = {}
                groundt_list = []
                pred_list = []
                hit_scores = []
                
                
                for i in tqdm(range(len(output_list))):
                    pred = normalize(parse_abnormality(output_list[i]))
                    gt = normalize(parse_abnormality(gt_list[i]))

                    gts[i] = gt
                    res[i] = pred

                    pred_list.append("".join(pred).strip())
                    groundt_list.append("".join(gt).strip())

                    hit_score = hitscore(pred, gt)
                    hit_scores.append(hit_score)

                bleu_scorer = Bleu(4)
                rouge_scorer = Rouge()
                bleu_score, bleu_scores = bleu_scorer.compute_score(
                    {k: [v] for k, v in gts.items()},
                    {k: [v] for k, v in res.items()},
                    verbose=0
                )
                rouge_score, rouge_scores = rouge_scorer.compute_score(
                    {k: [v] for k, v in gts.items()},
                    {k: [v] for k, v in res.items()})
                _, _, bert_score = bertscorer.score(groundt_list, pred_list)
                bert_score = bert_score[-1].mean().item()

                print("BLEU Score for Case Disorder Prediction: ", bleu_score)
                print("ROUGE Score for Case Disorder Prediction: ", rouge_score)
                print("BERTScore for Case Disorder Prediction: ", bert_score)
                print("Average Hit Score for Case Disorder Prediction: ", np.mean(hit_scores))
                exit()

        else:
            start_idx = 0  # Start from the beginning if the file does not exist


        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if start_idx == 0:
                # Write the header if the file is new
                writer.writerow(['Volumename', 'nii_path', 'Output_result', 'GT_result'])
            
            gts = {}
            res = {}
            groundt_list = []
            pred_list = []
            hit_scores = []
            for idx in tqdm(range(start_idx, len(case_disorder_df))):
                volume = volume_list[idx]
                disorders = disorders_list[idx]
                
                # process_image 
                image_path = volume_to_nii_path_dict[volume]
                img_array = read_nii_img(image_path)
                aug_data_dict = augmentator({'image': img_array})
                image = aug_data_dict['image']
                image = image.to('cuda')
                
                query_text = f'<Disorder prediction> What are the abnormalities in the scan?'
                input_text = f'<Disorder prediction> What are the abnormalities in the scan? <Disorder> {disorders.capitalize()}'# <|end_of_text|>'

                # Tokenize input text without adding special tokens
                tokenizer_output = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                input_ids = tokenizer_output['input_ids']

                # Copy input_ids to label and set eos_token_id locations to -100
                label = input_ids.clone()

                # Process query text if provided
                if query_text:
                    query_tokenizer_output = tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
                    query_input_ids = query_tokenizer_output['input_ids']
                    # Set the labels for the length of the query input ids to -100
                    label[:, :query_input_ids.size(1)] = -100

                # Convert input_ids to list and append eos_token_id, then pad to 960 (1024 - 64) as per your requirement
                input_ids = input_ids.squeeze().tolist() + [tokenizer.eos_token_id]
                input_ids = input_ids + [tokenizer.pad_token_id] * (model_args.llm_max_length - num_img_patches - len(input_ids)) # Padding input_ids

                # Convert label to list, append eos_token_id, and pad both to 1024
                label = label.squeeze().tolist() + [tokenizer.eos_token_id]
                label = [-100]*num_img_patches + label + [-100] * (model_args.llm_max_length - num_img_patches - len(label))  # Padding labels
                
                label = torch.tensor(np.array(label)).unsqueeze(0).to('cuda')
                input_ids = torch.tensor(np.array(input_ids)).unsqueeze(0).to('cuda')
                
                with torch.no_grad():
                    output_sentence = model.generate(input_sentence = query_text, input_image=image, input_ids=input_ids, labels=label)
                    
                    print('input_sentence:',query_text)
                    print('output_sentence:',output_sentence)
                    print('grounded_sentence:',input_text[len(query_text):])
                    gt_sentence = input_text[len(query_text):]
                    
                    # Write the current row to the CSV file
                    writer.writerow([volume, image_path, output_sentence, gt_sentence])
                    file.flush()

        df = pd.read_csv(file_path)
        output_list = df['Output_result'].tolist()
        gt_list = df['GT_result'].tolist()
        gts = {}
        res = {}
        groundt_list = []
        pred_list = []
        hit_scores = []
                
                
        for i in tqdm(range(len(output_list))):
            pred = normalize(parse_abnormality(output_list[i]))
            gt = normalize(parse_abnormality(gt_list[i]))

            gts[i] = gt
            res[i] = pred

            pred_list.append("".join(pred).strip())
            groundt_list.append("".join(gt).strip())

            hit_score = hitscore(pred, gt)
            hit_scores.append(hit_score)

        bleu_scorer = Bleu(4)
        rouge_scorer = Rouge()
        bleu_score, bleu_scores = bleu_scorer.compute_score(
            {k: [v] for k, v in gts.items()},
            {k: [v] for k, v in res.items()},
            verbose=0
        )
        rouge_score, rouge_scores = rouge_scorer.compute_score(
            {k: [v] for k, v in gts.items()},
            {k: [v] for k, v in res.items()})
        _, _, bert_score = bertscorer.score(groundt_list, pred_list)
        bert_score = bert_score[-1].mean().item()

        print("BLEU Score for Case Disorder Prediction: ", bleu_score)
        print("ROUGE Score for Case Disorder Prediction: ", rouge_score)
        print("BERTScore for Case Disorder Prediction: ", bert_score)
        print("Average Hit Score for Case Disorder Prediction: ", np.mean(hit_scores))