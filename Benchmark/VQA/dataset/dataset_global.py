'''
Author: xm_cmic
Date: 2024-04-18 22:15:46
LastEditors: xm_cmic
LastEditTime: 2024-05-25 20:56:12
FilePath: /src-0515/dataset/dataset_global.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
import os 
import csv 
import json 
import numpy as np 
import pandas as pd
import random
import copy
import nibabel as nib

import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from monai.transforms import (
    Resized,
    Compose
)
from monai.transforms.utility.dictionary import AddChannelD, ToTensorD
from monai.transforms.spatial.dictionary import OrientationD, SpacingD, RandRotate90D, RandFlipD
from monai.transforms.croppad.dictionary import RandSpatialCropD, SpatialPadD, RandScaleCropD
from monai.transforms.intensity.dictionary import RandShiftIntensityD, ScaleIntensityRangeD, RandScaleIntensityD, RandGaussianNoiseD
import ipdb

# def zscore_process(image):
#     '''
#         zscore normalization
#     '''
#     lower_bound, upper_bound = -1000, 1000
#     image = np.clip(image, lower_bound, upper_bound)
#     image = image.astype(np.float32)
#     b_min, b_max = -1, 1
#     i_min, i_max = image.min(), image.max()
#     if i_max - i_min > 0:
#         image = (image - i_min) / (i_max - i_min) * (b_max - b_min) + b_min
#     else:
#         image = np.zeros_like(image)
#     return image


def assign_sample_weights(csv_file, column_name):
    # assign weight of each column
    df = pd.read_csv(csv_file)
    freq_dict = df[column_name].value_counts().to_dict()
    max_freq = max(freq_dict.values())
    weights = {value: max_freq / freq for value, freq in freq_dict.items()}
    df['sample_weight'] = df[column_name].map(weights)
    return df

def assign_sample_weights_df(df, column_name):
    freq_dict = df[column_name].value_counts().to_dict()
    max_freq = max(freq_dict.values())
    weights = {value: max_freq / freq for value, freq in freq_dict.items()}
    df['sample_weight'] = df[column_name].map(weights)
    return df
 
 

class CTRATE_Dataset(Dataset):
    # CTRATE_Dataset(data_args.train_image_path_csv,data_args.train_regioned_report_csv,data_args.train_qa_abnormality_csv,data_args.train_qa_location_csv,data_args.train_qa_presence_csv,data_args.train_qa_size_csv,data_args.train_disorders_csv,data_args.train_mask_root_dir,data_args.train_anatomy_mask_root_dir,tokenizer=tokenizer,train=True)
    def __init__(self, image_path_csv, regioned_report_csv, qa_abnormality_csv, qa_location_csv, qa_presence_csv, qa_size_csv,
                 disorders_csv, mask_root_dir, anatomy_mask_root_dir, tokenizer, crop_size=(224,224,112), train=True, total_length=1024):
        # mask_root_dir big region
        self.mask_root_dir = mask_root_dir
        self.anatomy_mask_root_dir = anatomy_mask_root_dir

        self.task_list = ["abnormality", "presence", "size", "location", "disorder"]
        # self.task_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        # self.task_weights = [0.3, 0.1, 0.1, 0.2, 0.3]
        self.task_weights = [0.25, 0.15, 0.15, 0.2, 0.25]
        
        report_df = pd.read_csv(image_path_csv)
        self.volume_to_nii_path_dict = dict(zip(report_df['Volumename'], report_df['nii_path']))
        
        regioned_report_df = pd.read_csv(regioned_report_csv)
        # 检查并填充NaN值
        regioned_report_df['Anatomy'].fillna('whole scan', inplace=True)
        # 提取region列的值
        regioned_report_df['Region'] = regioned_report_df['Anatomy'].apply(lambda x: str(x).split('/')[0].strip().lower())
        self.region_report_df = regioned_report_df #assign_sample_weights_df(regioned_report_df, 'Region')
        
        # Volumename,Anatomy,Abnormality,Non-Abnormality
        qa_df_with_abnormalities = pd.read_csv(qa_abnormality_csv)
        # Volumename,Anatomy,Abnormality
        self.qa_df_with_abnormalities = assign_sample_weights_df(qa_df_with_abnormalities,'Abnormality')
        
        qa_df_with_location = pd.read_csv(qa_location_csv)
        self.qa_df_with_location = qa_df_with_location #assign_sample_weights_df(qa_df_with_location,'Anatomy')
        
        self.qa_df_with_presnece = pd.read_csv(qa_presence_csv)

        self.qa_size_df = pd.read_csv(qa_size_csv) #assign_sample_weights(qa_size_csv, 'Abnormality')
        # self.case_disorder_df = pd.read_csv(disorders_csv) #assign_sample_weights(disorders_csv, 'Disorders') 
        self.case_disorder_df = assign_sample_weights(disorders_csv, 'Disorders') 

        self.total_length = total_length
        
        abnormality_template_json = './data/ctrate/generate_data/qa_templates/abnormality_template.json'
        with open(abnormality_template_json, 'r') as file:
            self.abnormality_template = json.load(file)['abnormality']
        
        location_template_json = './data/ctrate/generate_data/qa_templates/location_template.json'
        with open(location_template_json, 'r') as file:
            self.location_template = json.load(file)['location']
        
        presence_template_json = './data/ctrate/generate_data/qa_templates/presence_template.json'
        with open(presence_template_json, 'r') as file:
            self.presence_template = json.load(file)['presence']
        
        size_template_json = './data/ctrate/generate_data/qa_templates/size_template.json'
        with open(size_template_json, 'r') as file:
            self.size_template = json.load(file)['size']
            
        disorder_template_json = './data/ctrate/generate_data/qa_templates/disorders_template.json'
        with open(disorder_template_json, 'r') as file:
            self.disorders_template = json.load(file)['disorders']
        
        self.crop_size = crop_size
        self.train = train
        self.img_padding = [-100]
        # self.img_token_num = int((crop_size[0]//16)*(crop_size[1]//16)*(crop_size[2]//8) / 8)
        # self.img_token_num = int((crop_size[0]//16)*(crop_size[1]//16)*(crop_size[2]//8))
        self.img_token_num = 64
        self.seq_length = self.total_length - (self.img_token_num)
        
        # self.augmentator = Resized(
        #         spatial_size = [self.crop_size[0],self.crop_size[1],self.crop_size[2]],
        #         keys=["image"],
        #         mode=['area']
        #     )

        self.augmentator = Compose(
            [
                AddChannelD(keys=["image"]),
                OrientationD(keys=["image"], axcodes="RAS"),
                # SpacingD(
                #     keys=["image"],
                #     pixdim=(1.0, 1.0, 1.0),
                #     mode="bilinear",
                # ),
                ScaleIntensityRangeD(
                    keys=["image"],
                    a_min=-1000.0,
                    a_max=1000.0,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                ),
                RandScaleCropD(
                    keys=["image"],
                    roi_scale=(0.8, 0.8, 0.8),
                    max_roi_scale=(1.0, 1.0, 1.0),
                    random_center=True,
                    random_size=True,
                ),
                RandSpatialCropD(
                    keys=["image"],
                    roi_size=(self.crop_size[0], self.crop_size[1], self.crop_size[2]),
                    random_size=False,
                    random_center=True,
                ),
                SpatialPadD(
                    keys=["image"],
                    spatial_size=(self.crop_size[0], self.crop_size[1], self.crop_size[2]),
                    mode="reflect",
                ),
                RandFlipD(
                    keys=["image"],
                    prob=0.2,
                    spatial_axis=0,
                ),
                RandFlipD(
                    keys=["image"],
                    prob=0.2,
                    spatial_axis=1,
                ),
                RandFlipD(
                    keys=["image"],
                    prob=0.2,
                    spatial_axis=2,
                ),
                RandShiftIntensityD(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.1,
                ),
                ToTensorD(keys=["image"])
            ]
        )

        
        self.tokenizer = tokenizer
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        
    def __len__(self):
        if self.train:
            return len(self.region_report_df)*10
        else:
            return 500

    # def sample_row(self,df):
    #     return df.sample(1, weights=df['sample_weight']) 
    
    def sample_row(self, df):
        return df.sample(1)
    
    def read_nii_img(self, nii_path):
        try:
            # nii_path = "../" + nii_path
            img = nib.load(nii_path)
            img_array = np.asanyarray(img.dataobj)
        except FileNotFoundError:
            print(f"Warning: File not found at {nii_path}. Returning a random array instead.")
            img_array = np.random.rand(224, 224, 112)  # Generates random floats
        except Exception as e:
            print(f"An error occurred while loading the NIfTI file: {e}")
            img_array = np.random.rand(224, 224, 112)  # Backup in case of other exceptions
        return img_array
                    
    def get_abnormality_data(self):
        select_abnormality_template = random.choice(self.abnormality_template)
        
        sample_row = self.sample_row(self.qa_df_with_abnormalities)
        volume = sample_row['Volumename'].values[0]
        anatomy = sample_row['Anatomy'].values[0]
        abnormality = sample_row['Abnormality'].values[0]
        
        node_anatomy = anatomy.split('/')[-1]
        head_anatomy = anatomy.split('/')[0]
        # is_nan = pd.isna(abnormality)
        
        image_path = self.volume_to_nii_path_dict[volume]
        img_array = self.read_nii_img(image_path)
        # img_array = zscore_process(img_array)
        # img_array = img_array[np.newaxis,:,:]

        if node_anatomy == "others":
            input_question = select_abnormality_template.replace('{region}',node_anatomy)
            query_text = f'<Grounded abnormality detection> {input_question}'
            input_text = f'<Grounded abnormality detection> {input_question} <Abnormality> {abnormality.capitalize()}.'
        if random.random() < 0.5:
            input_question = select_abnormality_template.replace('{region}',node_anatomy)
            query_text = f'<Grounded abnormality detection> {input_question}'
            input_text = f'<Grounded abnormality detection> {input_question} <Abnormality> {abnormality.capitalize()}.'
        else:
            input_question = select_abnormality_template.replace('{region}','given region')
            query_text = f'<Grounded abnormality detection> {input_question}'
            input_text = f'<Grounded abnormality detection> {input_question} <Region> The given region is {node_anatomy}. This is belong to {head_anatomy}. <Abnormality> {abnormality.capitalize()}.'

        return img_array, query_text, input_text 
        
    def get_presence_data(self):
        select_presence_template = random.choice(self.presence_template)
        
        sample_row = self.sample_row(self.qa_df_with_presnece)
        # Volumename,Anatomy,Finding,Presence
        volume = sample_row['Volumename'].values[0]
        anatomy = sample_row['Anatomy'].values[0]
        select_abnormality = sample_row['Finding'].values[0]
        answer = sample_row['Presence'].values[0]
        
        node_anatomy = anatomy.split('/')[-1]
        # is_abnormality_nan = pd.isna(abnormality)
        # is_nonabnormality_nan = pd.isna(nonabnormality)
        
        image_path = self.volume_to_nii_path_dict[volume]
        img_array = self.read_nii_img(image_path)
        # img_array = zscore_process(img_array)
        # img_array = img_array[np.newaxis,:,:]
        
            
        if random.random() < 0.5:
            input_question = select_presence_template.replace('{region}',node_anatomy).replace('{abnormality}',select_abnormality)
        else:
            input_question = select_presence_template.replace('{region}','given region').replace('{abnormality}',select_abnormality)
        query_text = f'<Grounded abnormality presence> {input_question}'
        input_text = f'<Grounded abnormality presence> {input_question} <Presence> {answer}'
        return img_array, query_text, input_text 
      
    def get_location_data(self):
        select_location_template = random.choice(self.location_template)
        # select_idx = random.choice(self.qa_indices_with_abnormalities)
        sample_row = self.sample_row(self.qa_df_with_location)
        volume = sample_row['Volumename'].values[0]
        anatomy = sample_row['Anatomy'].values[0]
        abnormality = sample_row['Abnormality'].values[0]
       
        
        node_anatomy = anatomy.split('/')[-1]
        abnormalities = abnormality.split(',')
        select_abnormality = random.choice(abnormalities).strip()

        # process_image 
        image_path = self.volume_to_nii_path_dict[volume]
        img_array = self.read_nii_img(image_path)
        # img_array = zscore_process(img_array)
        # img_array = img_array[np.newaxis,:,:]
        
        input_question = select_location_template.replace('{abnormality}',select_abnormality)
        query_text = f'<Abnormality location> {input_question}'
        input_text = f'<Abnormality location> {input_question} <Location> {anatomy.capitalize()}.'
            
        return img_array, query_text, input_text 
      
    def get_size_data(self):
        select_size_template = random.choice(self.size_template)
        # select_idx = random.randint(0,len(self.qa_size_volume_list)-1)
        sample_row = self.sample_row(self.qa_size_df)
        volume = sample_row['Volumename'].values[0]
        anatomy = sample_row['Anatomy'].values[0]
        abnormality = sample_row['Abnormality'].values[0]
        size = sample_row['Size'].values[0]
        
        # volume = self.qa_size_volume_list[select_idx]
        # anatomy = self.qa_size_anatomy_list[select_idx]
        # abnormality = self.qa_size_abnormality_list[select_idx]
        # size = self.qa_size_size_list[select_idx]
        
        node_anatomy = anatomy.split('/')[-1]
        abnormalities = abnormality.split(',')
        select_abnormality = random.choice(abnormalities).strip()

        # process_image 
        image_path = self.volume_to_nii_path_dict[volume]
        img_array = self.read_nii_img(image_path)
        # img_array = zscore_process(img_array)
        # img_array = img_array[np.newaxis,:,:]
        
        if random.random() < 0.5:
            input_question = select_size_template.replace('{abnormality}',select_abnormality).replace('{region}',node_anatomy)
        else:
            input_question = select_size_template.replace('{abnormality}',select_abnormality).replace('{region}','given region')
            
        query_text = f'<Abnormality size prediction> {input_question}'
        input_text = f'<Abnormality size prediction> {input_question} <Size> {size}.'
            
        return img_array, query_text, input_text 
               
    def get_report_data(self):
        sample_row = self.sample_row(self.region_report_df)
        # sample_row = self.region_report_df.iloc[idx]
        volume = sample_row['Volumename'].values[0]
        anatomy = sample_row['Anatomy'].values[0]
        region = sample_row['Region'].values[0]
        report = sample_row['Sentence'].values[0]
        image_path = self.volume_to_nii_path_dict[volume]
        # select_idx = random.randint(0,len(self.volume_list)-1)
        # volume = self.volume_list[select_idx]
        # anatomy = self.anatomy_list[select_idx]
        # report = self.report_list[select_idx]
        # image_path = self.nii_path_list[select_idx]
        
        # process_image 
        img_array = self.read_nii_img(image_path)
        # img_array = zscore_process(img_array)
        # img_array = img_array[np.newaxis,:,:]
        
        node_anatomy = anatomy.split('/')[-1]
        head_anatomy = anatomy.split('/')[0]
        
        if node_anatomy == "others" or node_anatomy == 'whole scan':
            query_text = f'<Grounded report generation> Please describe the detailed finding of the {anatomy} region.'
            input_text = f'<Grounded report generation> Please describe the detailed finding of the {anatomy} region. <Report> {report}'
        elif random.random() < 0.5:
            query_text = f'<Grounded report generation> Please describe the detailed finding of the {anatomy} region.'
            input_text = f'<Grounded report generation> Please describe the detailed finding of the {anatomy} region. <Report> {report}'
        else:
            query_text = f'<Grounded report generation> Please describe the detailed finding of the given region.'
            input_text = f'<Grounded report generation> Please describe the detailed finding of the given region. <Region> The given region is {node_anatomy}. This is belong to {head_anatomy}. <Report> {report}'
        
        return img_array, query_text, input_text 
    
    def get_disorder_data(self):
        select_disorder_template = random.choice(self.disorders_template)
        
        sample_row = self.sample_row(self.case_disorder_df)
        volume = sample_row['Volumename'].values[0]
        disorders = sample_row['Disorders'].values[0]

        # process_image 
        image_path = self.volume_to_nii_path_dict[volume]
        img_array = self.read_nii_img(image_path)
        # img_array = zscore_process(img_array)
        # img_array = img_array[np.newaxis,:,:]

        input_question = select_disorder_template
        query_text = f'<Disorder prediction> {input_question}'
        input_text = f'<Disorder prediction> {input_question} <Disorder> {disorders.capitalize()}.'
            
        return img_array, query_text, input_text 
    
    def __getitem__(self, idx):
        # img_array, mask_array, query_text, input_text = self.get_report_data(idx)
        
        task_idx = random.choices(self.task_list, weights=self.task_weights, k=1)[0] #random.choice(self.task_list)
        is_neg = False

        if task_idx == "report":
            img_array, query_text, input_text = self.get_report_data()
        elif task_idx == "abnormality":
            img_array, query_text, input_text = self.get_abnormality_data()
            gt = input_text.split("<Abnormality> ")[-1].strip('.').lower()
            if gt == "no findings":
                is_neg = True
        elif task_idx == "presence":
            img_array, query_text, input_text = self.get_presence_data()
        elif task_idx == "location":
            img_array, query_text, input_text = self.get_location_data()
        elif task_idx == "size":
            img_array, query_text, input_text = self.get_size_data()
        elif task_idx == "disorder":
            img_array, query_text, input_text = self.get_disorder_data()
            gt = input_text.split("<Disorder> ")[-1].strip('.').lower()
            if gt == "no findings":
                is_neg = True

        aug_data_dict = self.augmentator({'image': img_array})
        image = aug_data_dict['image']
        
        # Tokenize input text without adding special tokens
        tokenizer_output = self.tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
        input_ids = tokenizer_output['input_ids']
    
        # Copy input_ids to label and set eos_token_id locations to -100
        label = input_ids.clone()

        # Process query text if provided
        if query_text:
            query_tokenizer_output = self.tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
            query_input_ids = query_tokenizer_output['input_ids']
            # Set the labels for the length of the query input ids to -100
            label[:, :query_input_ids.size(1)] = -100

        # Convert input_ids to list and append eos_token_id, then pad
        input_ids = input_ids.squeeze().tolist() + [self.tokenizer.eos_token_id]
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.total_length - self.img_token_num - len(input_ids))  # Padding input_ids

        # Convert label to list, append eos_token_id, and pad both to 4096
        label = label.squeeze().tolist() + [self.tokenizer.eos_token_id]
        label = [-100]*self.img_token_num + label + [-100] * (self.total_length - self.img_token_num - len(label))  # Padding labels

        task_idx_ids = []
        neg_signals = []
        task_idx_id = self.task_list.index(task_idx)   # 0 and 4 are abnormality and disorder
        task_idx_ids.append(task_idx_id)
        neg_signals.append(is_neg)
        
        item = {
                "input_image": image,
                'input_ids': np.array(input_ids),
                'task_idx': np.array(task_idx_ids),
                'neg_signal': np.array(neg_signals),
                'labels': np.array(label)
            }
        return item 
    



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('../model/Qwen3-4B-Instruct-2507', use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = CTRATE_Dataset(
        image_path_csv="../RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_image_path.csv",
        regioned_report_csv="../RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_region_report.csv",
        qa_abnormality_csv="../RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_vqa_abnormality.csv",
        qa_location_csv="../RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_vqa_location.csv",
        qa_presence_csv="../RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_vqa_presence.csv",
        qa_size_csv="../RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_vqa_size.csv",
        disorders_csv="../RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_case_disorders.csv",
        mask_root_dir="../RadGenome-ChestCT/dataset/train_region_mask",
        anatomy_mask_root_dir="../RadGenome-ChestCT/dataset/train_anatomy_mask",
        tokenizer=tokenizer,
        crop_size=(224,224,112),
        train=True,
        total_length=768
    )
    # import ipdb; ipdb.set_trace()
    for i in range(100):
        print(dataset[i]['task_idx'], dataset[i]['neg_signal'])