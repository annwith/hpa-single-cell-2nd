import numpy as np
import os
import pandas as pd
from skimage.io import imread
from pathlib import Path
import cv2

from torch.utils.data import Dataset
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)
import random
import torch
import math


def a_ordinary_collect_method(batch):
    '''
    I am a collect method for User Dataset
    '''
    img, pe, exp, msk, cnt = [], [], [], [], []
    # debug
    study_id = []
    weight = []
    # debug end
    if len(batch[0]) == 5:
        for i, p, e, m, l in batch:
            img.append(i)
            pe.append(p)
            exp.append(e)
            msk.append(m)
            cnt.append(l)
        return (torch.cat(img), torch.tensor(np.concatenate(pe)).long(),
                torch.tensor(np.concatenate(exp)).float(), torch.tensor(np.concatenate(msk)).float(), cnt[0])

def normwidth(size, margin=32):
    outsize = size // margin * margin
    outsize = max(outsize, margin)
    return outsize


def resize_short(img, target_size):
    """ resize_short """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(math.ceil(img.shape[1] * percent))
    resized_height = int(math.ceil(img.shape[0] * percent))

    # resized_width = normwidth(resized_width)
    # resized_height = normwidth(resized_height)
    resized = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
    return resized


class RANZERDataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None, mode='train', file_dict=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = tfms
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
        ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.file_dict = file_dict
        self.cols = ['class{}'.format(i) for i in range(19)]
        if cfg.data.cell == 'none':
            self.cell_path = 'notebooks/pad_resized_cell_four'
        else:
            self.cell_path = cfg.data.cell

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.mode == 'train':
            row = self.df.loc[index]
            cnt = self.cfg.experiment.count
            if row['idx'] > cnt:
                selected = random.sample([i for i in range(row['idx'])], cnt)
            else:
                selected = [i for i in range(row['idx'])]
            batch = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
            mask = np.zeros((cnt))
            label = np.zeros((cnt, 19))
            for idx, s in enumerate(selected):
                path = self.path / f'../../{self.cell_path}/{row["ID"]}_{s+1}.png'
                img = imread(path)
                if self.transform is not None:
                    res = self.transform(image=img)
                    img = res['image']
                if not img.shape[0] == self.cfg.transform.size:
                    img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
                img = self.tensor_tfms(img)
                batch[idx, :, :, :] = img
                mask[idx] = 1
                label[idx] = row[self.cols].values.astype(np.float)
            # img = self.tensor_tfms(img)
            if self.cfg.experiment.smoothing == 0:
                return batch, mask, label, row[self.cols].values.astype(np.float)
            else:
                return batch, mask, 0.9*label + 0.1/19, 0.9 * row[self.cols].values.astype(np.float) + 0.1/19

            # return batch, mask, label, row[self.cols].values.astype(np.float)
        
        if self.mode == 'valid':
            row = self.df.loc[index]
            selected = [i for i in range(row['idx'])]
            cnt = row['idx']
            batch = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
            mask = np.zeros((cnt))
            label = np.zeros((cnt, 19))
            for idx, s in enumerate(selected):
                path = self.path / f'../../{self.cell_path}/{row["ID"]}_{s+1}.png'
                img = imread(path)
                if self.transform is not None:
                    res = self.transform(image=img)
                    img = res['image']
                if not img.shape[0] == self.cfg.transform.size:
                    img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
                img = self.tensor_tfms(img)
                batch[idx, :, :, :] = img
                mask[idx] = 1
                label[idx] = row[self.cols].values.astype(np.float)

            return batch, mask, label, row[self.cols].values.astype(np.float), cnt
        

class ConfAwareRANZERDataset(Dataset):  # Inherits from PyTorch's Dataset class
    def __init__(self, df, tfms=None, cfg=None, mode='train', file_dict=None):
        # Store the dataframe with sample metadata
        self.df = df.reset_index(drop=True)

        if cfg.train.conf_aware:
            self.conf_df = pd.read_csv(cfg.train.conf_csv)
            self.conf_df = self.conf_df.reset_index(drop=True)
            self.conf_df = self.conf_df.set_index('filename')
            self.conf_cols = ['prob_{}'.format(i) for i in range(19)]
            print("ConfAwareRANZERDataset (init) conf_df dataframe:", self.conf_df.head())
        else:
            self.conf_df = None
        
        self.mode = mode  # Either 'train' or 'valid'
        self.transform = tfms  # Optional image transforms (e.g. from Albumentations)
        self.cfg = cfg  # Configuration object with settings like input size, paths, etc.

        # Normalization and conversion to tensor
        self.tensor_tfms = Compose([
            ToTensor(),  # Converts image to PyTorch tensor (C x H x W)
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),  # Normalizes each channel
        ])

        # Path to the current file's directory
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))

        self.file_dict = file_dict  # Optional dictionary mapping IDs to files
        self.cols = ['class{}'.format(i) for i in range(19)]  # Target label column names

        # Define where to find cell images
        if cfg.data.cell == 'none':
            self.cell_path = 'notebooks/pad_resized_cell_four'  # default path
        else:
            self.cell_path = cfg.data.cell  # custom path from config

        # Print dataframe
        print("(ConfAwareRANZERDataset (init) df dataframe:", self.df.head())

    def __len__(self):
        # Return the number of rows (samples) in the DataFrame
        return len(self.df)

    def __getitem__(self, index):
        # Get the sample row
        row = self.df.loc[index]

        # -------- TRAIN MODE --------
        if self.mode == 'train':
            cnt = self.cfg.experiment.count  # number of cells per image to sample

            # If more cells than count, sample a subset
            if row['idx'] > cnt:
                selected = random.sample([i for i in range(row['idx'])], cnt)
            else:
                # Otherwise use all available cells
                selected = [i for i in range(row['idx'])]

            # Allocate empty tensors for images, masks, labels and confidence scores
            batch = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))  # 4-channel images
            mask = np.zeros((cnt))  # 1 if cell exists, 0 if padded
            label = np.zeros((cnt, 19))  # one-hot/multilabel target vector for each cell
            conf = np.zeros((cnt, 19)) # confidence scores for each cell

            # Load and process each selected cell image
            for idx, s in enumerate(selected):
                path = self.path / f'../../{self.cell_path}/{row["ID"]}_{s+1}.png'
                img = imread(path)

                # Apply optional image augmentations
                if self.transform is not None:
                    res = self.transform(image=img)
                    img = res['image']

                # Ensure image has the correct size
                if not img.shape[0] == self.cfg.transform.size:
                    img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))

                # Apply tensor conversion and normalization
                img = self.tensor_tfms(img)

                # Store processed image and metadata
                batch[idx, :, :, :] = img
                mask[idx] = 1  # this cell exists
                label[idx] = row[self.cols].values.astype(np.float64)  # target values for that cell
                if self.cfg.train.conf_aware:
                    # print("rowID:", row['ID'])
                    conf_row = self.conf_df.loc[row['ID']+f'_{s+1}']
                    conf[idx] = conf_row[self.conf_cols].values.astype(np.float64)

            img_label = row[self.cols].values.astype(np.float64)

            if self.cfg.train.conf_aware:
                img_conf = self.conf_df.loc[row['ID']]
                img_conf = img_conf[self.conf_cols].values.astype(np.float64)

            # Apply label smoothing if configured
            if self.cfg.experiment.smoothing == 0:
                return batch, mask, label, img_label, conf, img_conf
            else:
                raise NotImplementedError("Label smoothing not implemented for confidence scores")

        # -------- VALIDATION MODE --------
        if self.mode == 'valid':
            selected = [i for i in range(row['idx'])]  # use all cells for validation
            cnt = row['idx']  # number of cells

            # Allocate tensors
            batch = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
            mask = np.zeros((cnt))
            label = np.zeros((cnt, 19))

            for idx, s in enumerate(selected):
                path = self.path / f'../../{self.cell_path}/{row["ID"]}_{s+1}.png'
                img = imread(path)

                if self.transform is not None:
                    res = self.transform(image=img)
                    img = res['image']

                if not img.shape[0] == self.cfg.transform.size:
                    img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))

                img = self.tensor_tfms(img)

                batch[idx, :, :, :] = img
                mask[idx] = 1
                label[idx] = row[self.cols].values.astype(np.float64)

            # Returns full batch, mask, labels, and count of cells
            return batch, mask, label, row[self.cols].values.astype(np.float64), cnt
        

class GetPredictionsDataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None):
        print('[ i ] GetPredictionsDataset')

        self.df = df.reset_index(drop=True)
        self.transform = tfms
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
        ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.cols = ['class{}'.format(i) for i in range(19)]
        if cfg.data.cell == 'none':
            self.cell_path = 'notebooks/pad_resized_cell_four'
        else:
            self.cell_path = cfg.data.cell

        print('self.cell_path: {}'.format(self.cell_path))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        selected = [i for i in range(row['idx'])]
        cnt = row['idx']
        filename = row['ID']

        batch = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
        mask = np.zeros((cnt))
        label = np.zeros((cnt, 19))
        for idx, s in enumerate(selected):
            path = self.path / f'../../{self.cell_path}/{row["ID"]}_{s+1}.png'
            img = imread(path)
            if self.transform is not None:
                res = self.transform(image=img)
                img = res['image']
            if not img.shape[0] == self.cfg.transform.size:
                img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
            img = self.tensor_tfms(img)
            batch[idx, :, :, :] = img
            mask[idx] = 1
            label[idx] = row[self.cols].values.astype(np.float64)

        return batch, mask, label, row[self.cols].values.astype(np.float64), cnt, filename
