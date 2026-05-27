import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import tqdm
# import osqp
import nibabel as nib
from monai.transforms import *
import re


pathologies = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
                    'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema',
                    'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
                    'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
                    'Interlobular septal thickening']


class CTReportDatasetinfer(Dataset):
    def __init__(self, args, data_folder, csv_file, mode="train", resize_dim=500, labels="labels.csv"):
        self.data_folder = data_folder
        self.labels = labels
        self.accession_to_text = self.load_accession_text(csv_file)
        self.mode = mode
        self.paths=[]
        self.samples = self.prepare_samples()

        transform_train = Compose([
            AddChannel(),
            Orientation(axcodes="RAS",image_only=True),
            Spacing(
                pixdim=(args.spacing_x, args.spacing_y, args.spacing_z),
                mode="bilinear",
                image_only=True,
            ),
            ScaleIntensityRange(
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            RandScaleCrop(
                roi_scale=(0.8, 0.8, 0.8),
                max_roi_scale=(1.0, 1.0, 1.0),
                random_center=True,
                random_size=True,
            ),
            RandSpatialCrop(
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                random_size=False,
                random_center=True,
            ),
            SpatialPad(
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="reflect"
            ),
            RandFlip(
                prob=args.RandFlip_prob,
                spatial_axis=0,
            ),
            RandFlip(
                prob=args.RandFlip_prob,
                spatial_axis=1,
            ),
            RandFlip(
                prob=args.RandFlip_prob,
                spatial_axis=2,
            ),
            RandShiftIntensity(
                offsets=0.10,
                prob=args.RandShiftIntensity_prob,
            ),
            RandGaussianNoise(
                prob=args.RandGaussianNoise_prob,
            ),
            ToTensor()
        ])

        transform_test = Compose([
            AddChannel(),
            Orientation(axcodes="RAS",image_only=True),
            Spacing(
                pixdim=(args.spacing_x, args.spacing_y, args.spacing_z),
                mode="bilinear",
                image_only=True,
            ),
            ScaleIntensityRange(
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CenterSpatialCrop(
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
            ),
            SpatialPad(
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                # mode="constant",
                # constant_values=-1,
                mode="reflect"
            ),
            ToTensor()
        ])

        if self.mode == "train":
            self.transform = transform_train
        else:
            self.transform = transform_test



    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        # df = df.drop_duplicates(subset='finding', keep='first')
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['img_path']] = row["finding"],row['impression']
        return accession_to_text


    def prepare_samples_ori(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))
        # print(patient_folders)
##        with open('/home/csexuefeng/CT-CLIP/scripts/patient_folders_train.json', 'w') as f:
##        with open('/home/csexuefeng/CT-CLIP/scripts/patient_folders_valid.json', 'w') as f:
##            json.dump(patient_folders, f)
##        with open('/home/csexuefeng/CT-CLIP/scripts/patient_folders_train.json', 'r') as file:
##        with open('/home/csexuefeng/CT-CLIP/scripts/patient_folders_valid.json', 'r') as file:
##            patient_folders = json.load(file)
        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        for patient_folder in tqdm.tqdm(patient_folders):
##            print(patient_folder)
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz.npz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

##                    accession_number = accession_number.replace(".npz", ".nii.gz")
                    accession_number = accession_number.replace(".npz", "")
                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]
                    text_final = ""
                    for text in list(impression_text):
                        text = str(text)
                        if text == "Not given.":
                            text = ""

                        text_final = text_final + text

                    onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
                    if len(onehotlabels) > 0:
                        samples.append((nii_file, text_final, onehotlabels[0]))
                        self.paths.append(nii_file)
        return samples


    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # with open('./ct_rate_preprocess_val.json', 'r') as file:
        #     nii_files = json.load(file)
        # Read labels once outside the loop
        df = pd.read_csv(self.labels)
        label_cols = list(df.columns[1:])
        df['one_hot_labels'] = list(df[label_cols].values)

        nii_files = list(self.accession_to_text.keys())
        for nii_file in nii_files:
            img_name = nii_file.split("/")[-1]
            impression_text = self.accession_to_text[nii_file]
            text_final = ""

            for text in list(impression_text):
                text = str(text)
                if text == "Not given.":
                    text = ""

                text_final = text_final + text

            onehotlabels = df[df["img_list"] == img_name]["one_hot_labels"].values
            if len(onehotlabels) > 0:
                samples.append((nii_file, text_final, onehotlabels[0]))
                self.paths.append(nii_file)

        return samples
    '''
    def prepare_samples(self):
        samples = []
        import json

        ann_path = '/project/medimgfmod/CTRG/label/chest/chest_ctrg_r2gen_nitfy_ori_split.json'
        ann = json.loads(open(ann_path, 'r').read())
        example = ann['val']
        for i in range(len(example)):
            nii_file = '/scratch/medimgfmod/CT/CTRG/Chest_New_nitfy_2/' + example[i]['id']
            input_text_concat = example[i]['report']
            samples.append((nii_file, input_text_concat))
            self.paths.append(nii_file)
        return samples
    '''
    

    def __len__(self):
        return len(self.samples)
    

    def _split_sentences_(self, report):
        report = report.replace('\n', ' ')
        sents = re.split(r'[.;]\s*', report)
        sents = [s.strip() for s in sents if len(s.strip()) > 0]
        return sents


    def __getitem__(self, index):
        nii_file, input_text, onehotlabels = self.samples[index]
        nii_file = nii_file.split("/")[-1]
        if nii_file.startswith("train") or nii_file.startswith("val"):
            paths = nii_file.split("_")
            image_path = paths[0] + "_" + paths[1] + "/"  + paths[0] + "_" + paths[1] + paths[2] + "/" + nii_file
            image_path = os.path.join(self.data_folder + "/" + paths[0] + "_preprocessed", image_path)
        else:
            image_path = os.path.join(self.data_folder, nii_file)
        image_data = nib.load(image_path)
        image_data = np.asanyarray(image_data.dataobj)

        if self.transform != None:
            image_data = self.transform(image_data)

        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')


        # input_sentences = self._split_sentences_(input_text)
        
        
        # text_prompts = []
        # for l in range(len(onehotlabels)):
        #     text_yes = ""
        #     text_no = ""
        #     if onehotlabels[l] == 1.0:
        #         text_yes = text_yes + f"{pathologies[l]}. "
        #         text_no = text_no + f"not {pathologies[l]}. "
        #     elif onehotlabels[l] == 0.0:
        #         text_yes = text_yes + f"not {pathologies[l]}. "
        #         text_no = text_no + f"{pathologies[l]}. "
        #     text = [text_yes, text_no]
        #     text_prompts.extend(text)

        acc_name = nii_file.replace(".nii.gz", "")

        return image_data, input_text, onehotlabels, acc_name




if __name__ == "__main__":
    data_folder = "../../../Data/CT-Rate"
    reports_file = "./dataset/CT-Rate_val_reports.csv"
    labels = "./dataset/CT-Rate_val_labels.csv"
    import argparse
    from monai.utils import set_determinism

    set_determinism(seed=42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vit_tiny_patch16', 'vit_base_patch16', 'vit_large_patch16', 'vit_large_patch32'],
                        default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # Required parameters

    parser.add_argument("--stage", type=str, default="train", help="train or test?")
    
    parser.add_argument("--task", choices=["CT-Rate", "RadChestCT", "CC-CCII", "LUNA16"],
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
    parser.add_argument("--RandFlip_prob", type=float, default=0.2, help="probability of RandFlip")
    parser.add_argument("--RandRotate_prob", type=float, default=0.8, help="probability of RandRotate 180°")
    parser.add_argument("--RandScaleIntensity_prob", type=float, default=0.1, help="probability of RandScaleIntensity")
    parser.add_argument("--RandShiftIntensity_prob", type=float, default=0.1, help="probability of RandShiftIntensity")
    parser.add_argument("--RandGaussianNoise_prob", type=float, default=0.2, help="probability of RandGaussianNoise")
    args = parser.parse_args()
    ds = CTReportDatasetinfer(args=args, data_folder=data_folder, csv_file=reports_file, labels=labels, mode="val")
    
    print(ds[0])