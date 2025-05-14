from utils import Config
import pandas as pd
from path import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataloaders.datasets import ConfAwareRANZERDataset, collect_changeable_number_of_cells
from dataloaders.transform_loader import get_tfms
import os
import numpy as np
from dataloaders.sampler import RandomBatchSampler


class RandomKTrainTestSplit:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        if cfg.experiment.file == 'none':
            csv_file = 'exp_with_idx_max.csv'
        else:
            csv_file = cfg.experiment.file
        train = pd.read_csv(path / 'split' / csv_file)

        self.train_meta, self.valid_meta = (train[train.fold != cfg.experiment.run_fold],
                                            train[train.fold == cfg.experiment.run_fold])
        if cfg.basic.debug:
            print('[ W ] Debug Mode!, down sample')
            self.train_meta = self.train_meta.sample(frac=0.05)
            self.valid_meta = self.valid_meta.sample(frac=0.05)

    def get_dataloader(self, test_only=False, train_shuffle=True, infer=False, tta=-1, tta_tfms=None):
        if test_only:
            raise NotImplementedError('Test only mode is not implemented!')

        print('[ √ ] Using transformation: {} & {}, image size: {}'.format(
            self.cfg.transform.name, self.cfg.transform.val_name, self.cfg.transform.size
        ))
        if self.cfg.transform.name == 'None':
            train_tfms = None
        else:
            train_tfms = get_tfms(self.cfg.transform.name)
        if tta_tfms:
            val_tfms = tta_tfms
        elif self.cfg.transform.val_name == 'None':
            val_tfms = None
        else:
            val_tfms = get_tfms(self.cfg.transform.val_name)

        print('[ i ] Use confidence aware dataset (ConfAwareRANZERDataset)')
        
        train_ds = ConfAwareRANZERDataset(
            df=self.train_meta, tfms=train_tfms, cfg=self.cfg, mode='train')
        valid_ds = ConfAwareRANZERDataset(
            df=self.valid_meta, tfms=val_tfms, cfg=self.cfg, mode='valid')

        if self.cfg.experiment.count == -1:
            train_dl = DataLoader(dataset=train_ds, batch_size=self.cfg.train.batch_size,
                                  num_workers=self.cfg.transform.num_preprocessor,
                                  collate_fn=collect_changeable_number_of_cells, 
                                  shuffle=train_shuffle, drop_last=True, pin_memory=True)
        else:
            train_dl = DataLoader(dataset=train_ds, batch_size=self.cfg.train.batch_size,
                                  num_workers=self.cfg.transform.num_preprocessor,
                                  shuffle=train_shuffle, drop_last=True, pin_memory=True)
        if tta == -1:
            tta = 1

        valid_dl = DataLoader(dataset=valid_ds, batch_size=self.cfg.eval.batch_size, drop_last=True,
                              num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)
        
        return train_dl, valid_dl, None
