from utils import Config
import pandas as pd
from path import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataloaders.datasets import ConfAwareRANZERDataset, RANZERDataset
from dataloaders.transform_loader import get_tfms
import os
import numpy as np
from dataloaders.sampler import RandomBatchSampler
from dataloaders.datasets import a_ordinary_collect_method


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

        if self.cfg.train.conf_aware:
            print('[ i ] Use confidence aware training')
            train_ds = ConfAwareRANZERDataset(
                df=self.train_meta, tfms=train_tfms, cfg=self.cfg, mode='train')
            valid_ds = ConfAwareRANZERDataset(
                df=self.valid_meta, tfms=val_tfms, cfg=self.cfg, mode='valid')
        else:
            print('[ i ] Use normal training')
            train_ds = RANZERDataset(
                df=self.train_meta, tfms=train_tfms, cfg=self.cfg, mode='train')
            valid_ds = RANZERDataset(
                df=self.valid_meta, tfms=val_tfms, cfg=self.cfg, mode='valid')

        if self.cfg.experiment.weight and train_shuffle:
            train = self.train_meta.copy()
            method_dict = {
                'sqrt': np.sqrt,
                'log2': np.log2,
                'log1p': np.log1p,
                'log10': np.log10,
                'as_it_is': lambda w: w
            }
            if self.cfg.experiment.method in method_dict:
                print('[ √ ] Use weighted sampler, method: {}'.format(self.cfg.experiment.method))
                cw = (1 / method_dict[self.cfg.experiment.method](train.iloc[:, :19].sum(0))).values
                print(train.head(2))
                weight = (train.iloc[:, :19] * cw).max(1).values
                print(weight)
            elif 'pow' in self.cfg.experiment.method:
                p = float(self.cfg.experiment.method.replace('pow_', ''))
                print('[ √ ] Use weighted sampler, method: Power of {}'.format(p))
                for x in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
                    train['{}_p'.format(x)] = (1 / np.power(
                        train[[x, 'fold']].groupby(x).transform('count')['fold'].values, p)
                                               ) / len(train[x].value_counts())
                weight = train[['grapheme_root_p', 'vowel_diacritic_p', 'consonant_diacritic_p']].max(1).values
            else:
                raise Exception('Unknown weighting method!')
            rs = WeightedRandomSampler(weights=weight, num_samples=len(weight))
            train_dl = DataLoader(train_ds, sampler=rs, batch_size=self.cfg.train.batch_size,
                                  num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)
        elif self.cfg.experiment.batch_sampler:
            print('[ i ] Batch Sampler!')
            bs = RandomBatchSampler(train_ds.df, self.cfg.train.batch_size, cfg=self.cfg)
            train_dl = DataLoader(dataset=train_ds, batch_sampler=bs,
                                  num_workers=self.cfg.transform.num_preprocessor)
        else:
            train_dl = DataLoader(dataset=train_ds, batch_size=self.cfg.train.batch_size,
                                  num_workers=self.cfg.transform.num_preprocessor,
                                  shuffle=train_shuffle, drop_last=True, pin_memory=True)
        if tta == -1:
            tta = 1

        valid_dl = DataLoader(dataset=valid_ds, batch_size=self.cfg.eval.batch_size,
                              collate_fn=a_ordinary_collect_method, drop_last=True,
                              num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)
        
        return train_dl, valid_dl, None
