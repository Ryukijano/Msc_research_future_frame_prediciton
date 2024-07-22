from typing import Union, List
from numpy.core.fromnumeric import clip, searchsorted
import torch
from torch import select
from torch.utils import data
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch import Tensor

import numpy as np
from PIL import Image
from pathlib import Path
import os
import copy
from typing import List
from tqdm import tqdm
import random

import cv2

def get_dataloader(data_set_name, batch_size, data_set_dir, num_past_frames=10, num_future_frames=10, test_past_frames = 10, test_future_frames = 10, ngpus = 1, num_workers = 1):
    if data_set_name == 'Suturing':
        # Define transformations (adjust as needed)
        norm_transform = VidNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        renorm_transform = VidReNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            VidResize((128, 128)),
            VidRandomHorizontalFlip(0.5),
            VidRandomVerticalFlip(0.5),
            VidToTensor(),
            VidNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([VidResize((128, 128)), VidToTensor(), norm_transform])

        # Load train and test datasets directly
        train_set = JIGSAWSDataset(Path(data_set_dir).joinpath('train'), train_transform,
                                    num_past_frames=num_past_frames, num_future_frames=num_future_frames)
        test_set = JIGSAWSDataset(Path(data_set_dir).joinpath('test'), test_transform,
                                num_past_frames=test_past_frames, num_future_frames=test_future_frames)
        N = batch_size
        train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=N, shuffle=False, num_workers=num_workers, drop_last=False)

        # Return loaders for the single split
        return train_loader, test_loader, renorm_transform

class JIGSAWSDataset(Dataset):
    """JIGSAWS Suturing dataset class."""

    def __init__(self, data_path, transform, num_past_frames=10, num_future_frames=10):
        self.data_path = Path(data_path)
        self.transform = transform
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.clip_length = num_past_frames + num_future_frames
        self.clips = self.load_data()

    def load_data(self):
        """Loads data clips from pre-split sequences."""
        clips = []
        for sequence_folder in self.data_path.glob('*'):
            frame_files = sorted(sequence_folder.glob('*.png'))
            clips.append(frame_files)
        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        images = [Image.open(f).convert('RGB') for f in clip]

        if self.transform:
            transformed = self.transform(images)
            if isinstance(transformed, torch.Tensor):
                #if the transform returns a single tensor, return it as is
                tensor = transformed
            else:
                #if it's still a list, we need to stack it
                tensor = torch.stack(transformed, dim=0)
        else:
            #if no transform, convert to tensor manually
            tensor = torch.stack([transforms.ToTensor()(img) for img in images], dim=0)
            
        return tensor[:self.num_past_frames], tensor[self.num_future_frames:]  # Return past and future frames

class VidResize(object):
    def __init__(self, *args, **resize_kwargs):
        self.resize_kwargs = resize_kwargs
        self.args = args

    def __call__(self, clip):
        if isinstance(clip, list):
            return [transforms.Resize(*self.args, **self.resize_kwargs)(img) for img in clip]
        else:
            return transforms.Resize(*self.args, **self.resize_kwargs)(clip)

class VidCenterCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip):
        if isinstance(clip, list):
            return [transforms.CenterCrop(*self.args, **self.kwargs)(img) for img in clip]
        else:
            return transforms.CenterCrop(*self.args, **self.kwargs)(clip)

class VidCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: Union[Image.Image, List[Image.Image]]):
        if isinstance(clip, list):
            return [transforms.functional.crop(img, *self.args, **self.kwargs) for img in clip]
        else:
            return transforms.functional.crop(clip, *self.args, **self.kwargs)
        
class VidRandomHorizontalFlip(object):
    def __init__(self, p: float):
        assert 0 <= p <= 1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: Union[Image.Image, List[Image.Image]]):
        if np.random.rand() < self.p:
            if isinstance(clip, list):
                return [transforms.functional.hflip(img) for img in clip]
            else:
                return transforms.functional.hflip(clip)
        return clip

class VidRandomVerticalFlip(object):
    def __init__(self, p: float):
        assert 0 <= p <= 1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: Union[Image.Image, List[Image.Image]]):
        if np.random.rand() < self.p:
            if isinstance(clip, list):
                return [transforms.functional.vflip(img) for img in clip]
            else:
                return transforms.functional.vflip(clip)
        return clip

class VidToTensor(object):
    def __call__(self, clip: Union[Image.Image, List[Image.Image]]):
        """
        Return: clip --- Tensor with shape (T, C, H, W) for list input or (C, H, W) for single image
        """
        if isinstance(clip, list):
            return torch.stack([transforms.ToTensor()(img) for img in clip], dim=0)
        else:
            return transforms.ToTensor()(clip)

class VidNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip: Union[Tensor, List[Tensor]]):
        """
        Return: clip --- Tensor with shape (T, C, H, W) for list input or (C, H, W) for single image
        """
        if isinstance(clip, list):
            return [transforms.Normalize(self.mean, self.std)(img) for img in clip]
        elif clip.dim() == 4:  # (T, C, H, W)
            return torch.stack([transforms.Normalize(self.mean, self.std)(img) for img in clip])
        else:  # (C, H, W)
            return transforms.Normalize(self.mean, self.std)(clip)

class VidReNormalize(object):
    def __init__(self, mean, std):
        try:
            self.inv_std = [1.0/s for s in std]
            self.inv_mean = [-m for m in mean]
            self.renorm = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.], std=self.inv_std),
                transforms.Normalize(mean=self.inv_mean, std=[1., 1., 1.])
            ])
        except TypeError:
            # For grayscale images
            self.inv_std = 1.0/std
            self.inv_mean = -mean
            self.renorm = transforms.Compose([
                transforms.Normalize(mean=0., std=self.inv_std),
                transforms.Normalize(mean=self.inv_mean, std=1.)
            ])

    def __call__(self, clip: Union[Tensor, List[Tensor]]):
        """
        Return: clip --- Tensor with shape (T, C, H, W) for list input or (C, H, W) for single image
        """
        if isinstance(clip, list):
            return [self.renorm(img) for img in clip]
        elif clip.dim() == 4:  # (T, C, H, W)
            return torch.stack([self.renorm(img) for img in clip])
        else:  # (C, H, W)
            return self.renorm(clip)

class VidPad(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: Union[Image.Image, List[Image.Image]]):
        if isinstance(clip, list):
            return [transforms.Pad(*self.args, **self.kwargs)(img) for img in clip]
        else:
            return transforms.Pad(*self.args, **self.kwargs)(clip)

def mean_std_compute(dataset, device, color_mode = 'RGB'):
    """
    arguments:
        dataset: pytorch dataloader
        device: torch.device('cuda:0') or torch.device('cpu') for computation
    return:
        mean and std of each image channel.
        std = sqrt(E(x^2) - (E(X))^2)
    """
    data_iter= iter(dataset)
    sum_img = None
    square_sum_img = None
    N = 0

    pgbar = tqdm(desc = 'summarizing...', total = len(dataset))
    for idx, sample in enumerate(data_iter):
        past, future = sample
        clip = torch.cat([past, future], dim = 0)
        N += clip.shape[0]

        img = torch.sum(clip, axis = 0)

        if idx == 0:
            sum_img = img
            square_sum_img = torch.square(img)
            sum_img = sum_img.to(torch.device(device))
            square_sum_img = square_sum_img.to(torch.device(device))
        else:
            img = img.to(device)
            sum_img = sum_img + img
            square_sum_img = square_sum_img + torch.square(img)
        
        pgbar.update(1)
    
    pgbar.close()

    mean_img = sum_img/N
    mean_square_img = square_sum_img/N
    if color_mode == 'RGB':
        mean_r, mean_g, mean_b = torch.mean(mean_img[0, :, :]), torch.mean(mean_img[1, :, :]), torch.mean(mean_img[2, :, :])
        mean_r2, mean_g2, mean_b2 = torch.mean(mean_square_img[0,:,:]), torch.mean(mean_square_img[1,:,:]), torch.mean(mean_square_img[2,:,:])
        std_r, std_g, std_b = torch.sqrt(mean_r2 - torch.square(mean_r)), torch.sqrt(mean_g2 - torch.square(mean_g)), torch.sqrt(mean_b2 - torch.square(mean_b))

        return ([mean_r.cpu().numpy(), mean_g.data.cpu().numpy(), mean_b.cpu().numpy()], [std_r.cpu().numpy(), std_g.cpu().numpy(), std_b.cpu().numpy()])
    else:
        mean = torch.mean(mean_img)
        mean_2 = torch.mean(mean_square_img)
        std = torch.sqrt(mean_2 - torch.square(mean))

        return (mean.cpu().numpy(), std.cpu().numpy())
