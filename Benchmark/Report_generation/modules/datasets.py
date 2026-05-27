import os
import json
import pandas as pd
import torch
from PIL import Image
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from monai.transforms import *

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        # self.num_slices = args.num_slices
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[
                :self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(IuxrayMultiImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)

    def _load_data(self, label_path):
        label_dict = {}

        data = pd.read_csv(label_path)
        for index, row in data.iterrows():
            idx = row['id']
            label = row[1:].to_list()

            label_dict[idx] = list(map(lambda x: 1 if x == 1.0 else 0, label))

        return label_dict

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        pid = image_id.split('_')[0][3:]
        try:
            labels = torch.tensor(self.label[int(pid)], dtype=torch.float32)
        except:
            # print('Except id ', pid)
            labels = torch.tensor([0 for _ in range(14)], dtype=torch.float32)

        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class CTRG_MultiImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(CTRG_MultiImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)
        self.args = args
        self.transform = transform


    def _load_data(self, label_path):
        label_dict = {}

        data = pd.read_csv(label_path)
        for index, row in data.iterrows():
            idx = row['id']
            label = row[1:].to_list()

            label_dict[idx] = list(map(lambda x: 1 if x == 1.0 else 0, label))

        return label_dict

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_paths = os.path.join(self.args.image_dir, image_id)
        
        '''
        # select num_slices images from image_paths
        num_slices = self.num_slices
        step = len(image_paths) // num_slices

        if len(image_paths) > num_slices:
            image_paths = [image_paths[i]
                           for i in range(0, len(image_paths), step)]
            image_paths = image_paths[:num_slices]
        else:
            image_paths = image_paths + \
                [image_paths[-1]] * (num_slices - len(image_paths))

        assert len(image_paths) == num_slices

        images = []
        for image_path in image_paths:
            image = Image.open(os.path.join(
                self.image_dir, image_path)).convert('L')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

##        image = torch.stack(images, 0)
##        labels = torch.tensor(self.label[int(image_id)], dtype=torch.float32)
        '''

        imageData = nib.load(image_paths)
        imageData = np.asanyarray(imageData.dataobj)
        transformed_image = self.transform(imageData)  if self.transform is not None else None
        # image = transformed_image['image'].permute(0,3,2,1).contiguous()
        image = transformed_image
        labels = torch.tensor(self.label[int(image_id[:-7])], dtype=torch.float32)
        '''
        images = []
        for image_path in image_paths:
            image = Image.open(os.path.join(self.image_dir, image_path)).convert('L')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

##        image = torch.stack(images, 0)
        image = torch.stack(images, -1)
##        image = F.interpolate(image, size=(128, 64)).permute(0,3,1,2).contiguous()
        image = F.interpolate(image, size=(96, 96)).permute(0,3,1,2).contiguous()
        labels = torch.tensor(self.label[int(image_id)], dtype=torch.float32)
        '''
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class CTRate_MultiImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(CTRate_MultiImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)
        self.args = args
        self.transform = transform


    def _load_data(self, label_path):
        label_dict = {}

        data = pd.read_csv(label_path)
        for index, row in data.iterrows():
            idx = row['img_list']
            label = row[1:].to_list()

            label_dict[idx] = list(map(lambda x: 1 if x == 1.0 else 0, label))

        return label_dict

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        image_id = example['img_path']
        paths = image_id.split("_")
        image_path = paths[0] + "_" + paths[1] + "/"  + paths[0] + "_" + paths[1] + paths[2] + "/" + image_id
        image_paths = os.path.join(self.args.image_dir, image_path)
        
        '''
        # select num_slices images from image_paths
        num_slices = self.num_slices
        step = len(image_paths) // num_slices

        if len(image_paths) > num_slices:
            image_paths = [image_paths[i]
                           for i in range(0, len(image_paths), step)]
            image_paths = image_paths[:num_slices]
        else:
            image_paths = image_paths + \
                [image_paths[-1]] * (num_slices - len(image_paths))

        assert len(image_paths) == num_slices

        images = []
        for image_path in image_paths:
            image = Image.open(os.path.join(
                self.image_dir, image_path)).convert('L')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

##        image = torch.stack(images, 0)
##        labels = torch.tensor(self.label[int(image_id)], dtype=torch.float32)
        '''

        imageData = nib.load(image_paths)
        imageData = np.asanyarray(imageData.dataobj)
        transformed_image = self.transform(imageData)  if self.transform is not None else None
        # image = transformed_image['image'].permute(0,3,2,1).contiguous()
        image = transformed_image
        labels = torch.tensor(self.label[image_id], dtype=torch.float32)
        '''
        images = []
        for image_path in image_paths:
            image = Image.open(os.path.join(self.image_dir, image_path)).convert('L')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

##        image = torch.stack(images, 0)
        image = torch.stack(images, -1)
##        image = F.interpolate(image, size=(128, 64)).permute(0,3,1,2).contiguous()
        image = F.interpolate(image, size=(96, 96)).permute(0,3,1,2).contiguous()
        labels = torch.tensor(self.label[int(image_id)], dtype=torch.float32)
        '''
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(MimiccxrSingleImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = pd.read_csv(args.label_path)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        d = self.label[self.label['dicom_id'] == image_id]
        labels = torch.tensor(d.values.tolist()[0][8:], dtype=torch.float32)
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class CovidSingleImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(CovidSingleImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)

    def _load_data(self, label_file):
        labels = {}

        # print(f"Loading data from {label_file}")

        data = pd.read_csv(label_file)
        # data = data[data['split'] == self.subset]
        for index, row in data.iterrows():
            idx = row['idx']
            label = [1, 0] if row['label'] == '轻型' else [0, 1]
            labels[idx] = label

        return labels

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        labels = torch.tensor(self.label[image_id], dtype=torch.float32)
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class CovidAllImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        labels = torch.tensor(example['label'], dtype=torch.float32)
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample
    


if __name__ == "__main__":
    import argparse
    from tokenizers import Tokenizer
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--image_dir', type=str, default='../../../../Data/CT-Rate/valid_preprocessed/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='../CT-Rate_split.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--label_path', type=str, default='../data_csv/CT-Rate_labels.csv',
                        help='the path to the directory containing the label.')
    parser.add_argument('--image_size', type=int, default=96, help='')
    parser.add_argument('--crop_size', type=int, default=96, help='')
    parser.add_argument('--dataset_name', type=str, default='CT-Rate',
                        choices=['iu_xray', 'mimic_cxr', 'covid', 'covidall', 'ctrg', 'CT-Rate'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=150, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=4, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=4, help='the number of samples for a batch')
    args = parser.parse_args()
    tokenizer = Tokenizer(args)
    ds = CTRate_MultiImageDataset(args, tokenizer=tokenizer, split='train')
    print(ds[0])
