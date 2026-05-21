from copy import deepcopy
import os
from typing import List, Tuple, Optional, Union
from PIL import Image
import pandas as pd
import numpy as np
import tokenizers
import random

import torch
from torch.utils.data import Dataset
import nibabel as nib
from monai.transforms.compose import Compose
from monai.transforms.utility.array import AddChannel, ToTensor
from monai.transforms.spatial.array import Orientation, Spacing, RandRotate90, RandFlip
from monai.transforms.croppad.array import RandSpatialCrop, SpatialPad, RandScaleCrop
from monai.transforms.intensity.array import RandShiftIntensity, ScaleIntensityRange, RandScaleIntensity
from monai.transforms.utility.dictionary import AddChannelD, ToTensorD
from monai.transforms.spatial.dictionary import OrientationD, SpacingD, RandRotate90D, RandFlipD
from monai.transforms.croppad.dictionary import RandSpatialCropD, SpatialPadD, RandScaleCropD
from monai.transforms.intensity.dictionary import RandShiftIntensityD, ScaleIntensityRangeD, RandScaleIntensityD, RandGaussianNoiseD


import ipdb


class ASAPDataset(Dataset):
    def __init__(
        self,
        args
    ):
        self.max_caption_length = args.max_caption_length
        self.data_root = args.data_path
        self.images_list, self.finding_list, self.reports_list = self._read_csv()

        self.dataset_path = args.dataset_path
        self.tokenizer = tokenizers.Tokenizer.from_file(os.path.join(self.data_root, "cxrbert_wordpiece.json"))
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.tokenizer.enable_truncation(max_length=self.max_caption_length)
        self.tokenizer.enable_padding(length=self.max_caption_length)

        self.transform = Compose(
            [
                AddChannelD(keys=["image", "label"]),
                OrientationD(keys=["image", "label"], axcodes="RAS"),
                SpacingD(
                    keys=["image", "label"],
                    pixdim=(args.spacing_x, args.spacing_y, args.spacing_z),
                    mode="bilinear",
                ),
                ScaleIntensityRangeD(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                RandScaleCropD(
                    keys=["image", "label"],
                    roi_scale=(args.roi_scale, args.roi_scale, args.roi_scale),
                    max_roi_scale=(1.0, 1.0, 1.0),
                    random_center=True,
                    random_size=True,
                ),
                RandSpatialCropD(
                    keys=["image", "label"],
                    roi_size=(args.roi_x, args.roi_y, args.roi_z),
                    random_size=False,
                    random_center=True,
                ),
                SpatialPadD(
                    keys=["image", "label"],
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode="reflect",
                ),
                RandFlipD(
                    keys=["image", "label"],
                    prob=args.RandFlip_prob,
                    spatial_axis=0,
                ),
                RandFlipD(
                    keys=["image", "label"],
                    prob=args.RandFlip_prob,
                    spatial_axis=1,
                ),
                RandFlipD(
                    keys=["image", "label"],
                    prob=args.RandFlip_prob,
                    spatial_axis=2,
                ),
                RandShiftIntensityD(
                    keys=["image"],
                    offsets=0.10,
                    prob=args.RandShiftIntensity_prob,
                ),
                ToTensorD(keys=["image", "label"])
            ]
        )


    def __len__(self):
        return len(self.images_list)


    def __getitem__(self, index):
        # load image
        volume_name = str(self.images_list[index])
        paths = volume_name.split("_")
        volume_path = paths[0] + "_" + paths[1] + "/"  + paths[0] + "_" + paths[1] + paths[2] + "/" + volume_name
        volume_path = os.path.join(self.dataset_path + "/train_preprocessed_int", volume_path)
        mask_path = os.path.join(self.dataset_path + "/mask_preprocessed/train", volume_name)

        image = nib.load(volume_path)
        image = np.asanyarray(image.dataobj)
        mask = nib.load(mask_path)
        mask = np.asanyarray(mask.dataobj)

        transformed = self.transform({"image": image, "label": mask})
        image = transformed["image"]    # type: ignore
        mask = transformed["label"]     # type: ignore
        
        ids_f, attention_mask_f, type_ids_f, sentence_mask_f = self._encode_text(self.finding_list[index], sent_mask=True)  # type: ignore
        if self.reports_list[index].endswith("Not given.") or self.reports_list[index].endswith("Not given"):
            self.reports_list[index] = self.finding_list[index]
        ids, attention_mask, type_ids = self._encode_text(self.reports_list[index], sent_mask=False)    # type: ignore
        masked_ids = self._random_mask(ids)

        return image, mask, ids_f, attention_mask_f, type_ids_f, sentence_mask_f, ids, attention_mask, type_ids, masked_ids

 
    def _read_csv(self):
        train_csv_path = os.path.join(self.data_root, "train_reports_ctrate.csv")
        df_train = pd.read_csv(train_csv_path, encoding='utf-8')
        train_img = np.array(df_train.iloc[:, 0])
        train_finding = np.array(df_train.iloc[:, 1])
        train_impression = np.array(df_train.iloc[:, 2])
        train_report = [str(s1) + str(s2) for s1, s2 in zip(train_finding, train_impression)]

        # -- only CT RATE --
        img_list = train_img
        finding_list = train_finding
        report_list = train_report

        return img_list, finding_list, report_list
    

    def _encode_text(self, text, sent_mask=False):
        sent = ""
        sent += str(text)
        sent = '[CLS] '+ sent
        sent = sent.replace("?", "").replace("(", "( ").replace(")", " ) ").replace("  ", " ").replace(":", ",").replace(";", ",")

        if sent_mask:
            encoded = self.tokenizer.encode(sent)

            ids = torch.tensor(encoded.ids).unsqueeze(0)
            attention_mask = torch.tensor(encoded.attention_mask).unsqueeze(0)
            type_ids = torch.tensor(encoded.type_ids).unsqueeze(0)
            sentence_mask = deepcopy(type_ids)
            sentence_cnt = 1
            temp = 0
            for index in range(len(ids[0])-1):
                if ids[0][index] == 18 and not self._isNumber(ids[0][index-1]) and not self._isNumber(ids[0][index+1]):     # 18 is '.', make sure not number
                    sentence_mask[0][temp:(index+1)] = sentence_cnt
                    sentence_cnt += 1
                    temp = index + 1
            
            return ids, attention_mask, type_ids, sentence_mask


        encoded = self.tokenizer.encode(sent)

        ids = torch.tensor(encoded.ids).unsqueeze(0)
        attention_mask = torch.tensor(encoded.attention_mask).unsqueeze(0)
        type_ids = torch.tensor(encoded.type_ids).unsqueeze(0)

        return ids, attention_mask, type_ids


    def _isNumber(self, id):
        if id >= 20 and id <= 29:
            return True
        return False


    def _random_mask(self,tokens):
        masked_tokens = deepcopy(tokens)
        
        for i in range(1, masked_tokens.shape[1]-1):                # random mask on 75% prob
            prob = random.random()
            if masked_tokens[0][i] != 3:         # not [SEP]
                if prob < 0.75:
                    masked_tokens[0][i] = 4      # [MASK] token id is 4


        return masked_tokens


    def collate_fn(self, instances: List[Tuple]):
        ids_f_list, attention_mask_f_list, type_ids_f_list, sentence_mask_f_list = [], [], [], []
        image_list, ids_list, attention_mask_list, type_ids_list, masked_ids_list = [], [], [], [], []
        mask_list = []
        # flatten
        for b in instances:
            image, mask, ids_f, attention_mask_f, type_ids_f, sentence_mask_f, ids, attention_mask, type_ids, masked_ids = b

            image_list.append(image)
            mask_list.append(mask)
            ids_f_list.append(ids_f)
            attention_mask_f_list.append(attention_mask_f)
            type_ids_f_list.append(type_ids_f)
            sentence_mask_f_list.append(sentence_mask_f)
            ids_list.append(ids)
            attention_mask_list.append(attention_mask)
            type_ids_list.append(type_ids)
            masked_ids_list.append(masked_ids)

        # stack
        image_stack = torch.stack(image_list)
        mask_stack = torch.stack(mask_list)
        ids_f_stack = torch.stack(ids_f_list).squeeze()
        attention_mask_f_stack = torch.stack(attention_mask_f_list).squeeze()
        type_ids_f_stack = torch.stack(type_ids_f_list).squeeze()
        sentence_mask_f_stack = torch.stack(sentence_mask_f_list).squeeze()
        ids_stack = torch.stack(ids_list).squeeze()
        attention_mask_stack = torch.stack(attention_mask_list).squeeze()
        type_ids_stack = torch.stack(type_ids_list).squeeze()
        masked_ids_stack = torch.stack(masked_ids_list).squeeze()

        # sort and add to dictionary
        return_dict = {
            "image": image_stack,
            "mask": mask_stack,
            "ids_f": ids_f_stack,
            "attention_mask_f": attention_mask_f_stack,
            "type_ids_f": type_ids_f_stack,
            "sentence_mask_f": sentence_mask_f_stack,
            "ids": ids_stack,
            "attention_mask": attention_mask_stack,
            "type_ids": type_ids_stack,
            "masked_ids": masked_ids_stack
        }

        return return_dict



import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../dataset', type=str, help='dataset path')
    parser.add_argument('--dataset_path', default='../../../../Data/CT-Rate', type=str, help='image dataset path')
    parser.add_argument('--checkpoint_path', default='../checkpoints', type=str, help='checkpoint path')
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
    parser.add_argument("--RandRotate90_prob", type=float, default=0.2, help="probability of RandRotate90")
    parser.add_argument("--RandShiftIntensity_prob", type=float, default=0.1, help="probability of RandShiftIntensity")
    args = parser.parse_args()


    dataset = ASAPDataset(args)
    i = 0
    for data in dataset:
        # ipdb.set_trace()
        i += 1

    print(i)
        
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)
    # for data in dataloader:
    #     print(data)
    #     break

    # for data in dataloader:
    #     print(data)
    #     break
    # print(len(dataset))
    # print(len(dataloader)