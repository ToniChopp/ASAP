import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset, CovidSingleImageDataset, CovidAllImageDataset, CTRG_MultiImageDataset, CTRate_MultiImageDataset

from monai.transforms.compose import Compose
from monai.transforms.io.array import LoadImage
from monai.transforms.utility.array import AddChannel, ToTensor, EnsureChannelFirst
from monai.transforms.spatial.array import Orientation, Spacing, RandRotate90, RandFlip, RandRotate
from monai.transforms.croppad.array import RandSpatialCrop, SpatialPad, CenterSpatialCrop, RandScaleCrop
from monai.transforms.intensity.array import RandShiftIntensity, ScaleIntensityRange, RandScaleIntensity, RandGaussianNoise


class LADataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

##        normalize = transforms.Normalize(mean=[0.500, 0.500, 0.500],
##                                         std=[0.275, 0.275, 0.275])
        
        # normalize = transforms.Normalize(mean=[0.500],
        #                                  std=[0.275])

        
        if split == 'train':
            self.transform = Compose([
                AddChannel(),
                Orientation(axcodes="RAS", image_only=True),
                ScaleIntensityRange(
                    a_min=0.0,
                    a_max=255.0,
                    b_min=-1.0,
                    b_max=1.0,
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
        else:
            self.transform = Compose([
                AddChannel(),
                Orientation(axcodes="RAS", image_only=True),
                ScaleIntensityRange(
                    a_min=0.0,
                    a_max=255.0,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                ),
                CenterSpatialCrop(
                    roi_size=(args.roi_x, args.roi_y, args.roi_z),
                ),
                SpatialPad(
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode="reflect"
                ),
                ToTensor()
            ])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'covid':
            self.dataset = CovidSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'covidall':
            self.dataset = CovidAllImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'ctrg':
            self.dataset = CTRG_MultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'CT-Rate':
            self.dataset = CTRate_MultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'pin_memory': True
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths, labels = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        labels = torch.stack(labels, 0)

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), labels

