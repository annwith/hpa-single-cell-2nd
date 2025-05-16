import numpy as np
import os
import pandas as pd
from skimage.io import imread
from pathlib import Path
import cv2

from torch.utils.data import Dataset
from torchvision.transforms import (
    ToTensor, Normalize, Compose)
import random
import torch
import math


def collect_changeable_number_of_cells(batch):
    # Desempacota o batch
    ipts, lbls, img_lbls, conf_lbls, conf_img_lbls, cnts = zip(*batch)

    # Concatena células (ex: ipt = [tensor(C_i) for i in batch] -> tensor(C_total, ...))
    ipts = torch.cat(ipts, dim=0)
    lbls = torch.cat(lbls, dim=0)
    conf_lbls = torch.cat(conf_lbls, dim=0)

    # lbls geralmente são rótulos da imagem inteira (1 por imagem), então pode ser empilhado
    img_lbls = torch.stack(img_lbls, dim=0)
    conf_img_lbls = torch.stack(conf_img_lbls, dim=0)

    # cnts indica quantas células por imagem — ex: [12, 8, 10] — mantido como tensor
    cnts = torch.tensor(cnts)

    return ipts, lbls, img_lbls, conf_lbls, conf_img_lbls, cnts


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
                label[idx] = row[self.cols].values.astype(np.float64)
            if self.cfg.experiment.smoothing == 0:
                return batch, mask, label, row[self.cols].values.astype(np.float64)
            else:
                return batch, mask, 0.9*label + 0.1/19, 0.9 * row[self.cols].values.astype(np.float64) + 0.1/19

        
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
                label[idx] = row[self.cols].values.astype(np.float64)

            return batch, mask, label, row[self.cols].values.astype(np.float64), cnt
        

class ConfAwareRANZERDataset(Dataset):  # Inherits from PyTorch's Dataset class
    def __init__(self, df, tfms=None, cfg=None, mode='train', file_dict=None):
        # Store the dataframe with sample metadata
        self.df = df.reset_index(drop=True)

        if cfg.train.conf_aware:
            self.conf_df = pd.read_csv(cfg.train.conf_csv)
            self.conf_df = self.conf_df.reset_index(drop=True)
            self.conf_df = self.conf_df.set_index('filename')
            self.conf_cols = ['prob_{}'.format(i) for i in range(19)]
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]

        # -------- TRAIN MODE --------
        if self.mode == 'train':
            if self.cfg.experiment.count == -1:
                cnt = row['idx']
            else:
                cnt = self.cfg.experiment.count

            # If more cells than count, sample a subset
            if row['idx'] > cnt:
                selected = random.sample([i for i in range(row['idx'])], cnt)
            else:
                # Otherwise use all available cells
                selected = [i for i in range(row['idx'])]

            # Allocate empty tensors for images, masks, labels and confidence scores
            batch = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
            label = np.zeros((cnt, 19))
            img_label = np.zeros((19))
            conf = np.zeros((cnt, 19))
            img_conf = np.zeros((19))

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
                label[idx] = row[self.cols].values.astype(np.float64)
                if self.cfg.train.conf_aware:
                    conf_row = self.conf_df.loc[row['ID']+f'_{s+1}']
                    conf[idx] = conf_row[self.conf_cols].values.astype(np.float64)

            img_label = row[self.cols].values.astype(np.float64)

            if self.cfg.train.conf_aware:
                img_conf = self.conf_df.loc[row['ID']]
                img_conf = img_conf[self.conf_cols].values.astype(np.float64)

            # Convert values to torch tensors
            batch = torch.tensor(batch)
            label = torch.tensor(label)
            img_label = torch.tensor(img_label)
            conf = torch.tensor(conf)
            img_conf = torch.tensor(img_conf)
            cnt = torch.tensor(cnt)

            # print("batch", batch.shape)
            # print("label", label.shape)
            # print("img_label", img_label.shape)
            # print("conf", conf.shape)
            # print("img_conf", img_conf.shape)
            # print("cnt", cnt)

            # Apply label smoothing if configured
            if self.cfg.experiment.smoothing == 0:
                return batch, label, img_label, conf, img_conf, cnt
            else:
                print("smooooooooooothing")
                label = 0.9 * label + 0.1 / 19
                img_label = 0.9 * img_label + 0.1 / 19
                return batch, label, img_label, conf, img_conf, cnt

        # -------- VALIDATION MODE --------
        if self.mode == 'valid':
            selected = [i for i in range(row['idx'])]  # use all cells for validation
            cnt = row['idx']  # number of cells

            batch = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
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
                label[idx] = row[self.cols].values.astype(np.float64)

            return batch, label, row[self.cols].values.astype(np.float64), cnt
        

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
