import os

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from utils import parse_args
from models import get_model
from dataloaders.datasets import RANZERDataset
from dataloaders.transform_loader import get_tfms

import warnings
warnings.filterwarnings('ignore')


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def basic_validate(model, dl, cfg, val_report_txt, model_path, mode):
    print('[ √ ] On validation! mode={}'.format(mode))

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        results = []
        losses, predicted, truth = [], [], []
        for i, (ipt, mask, lbl, img_lbl, n_cell) in enumerate(tqdm(dl)):

            ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
            exp_label = img_lbl.view(-1, 19)
            ipt, exp_label = ipt.to(DEVICE), exp_label.to(DEVICE)

            # Get logits and loss
            if cfg.basic.amp == 'Native':
                with torch.cuda.amp.autocast():
                    _, output = model(ipt, n_cell)
                    loss = F.binary_cross_entropy_with_logits(
                        output, exp_label,
                        reduction='none')
                    if not len(loss.shape) == 0:
                        loss = loss.mean()
                    output = output.float()
            else:
                raise NotImplementedError("cfg.basic.amp is not Native.")
            
            losses.append(loss.item())

            # Predictions
            pred = torch.sigmoid(output.cpu()).numpy()

            # Append to lists
            predicted.append(pred)
            truth.append(exp_label.cpu().numpy())
            
            results.append({
                'step': i,
                'loss': loss.item(),
            })
        
        # Concatenate results and calculate validation loss
        predicted = np.concatenate(predicted)
        truth = np.concatenate(truth)
        val_loss = np.array(losses).mean()
        
        # Classification report
        predicted_binary = (predicted > 0.5).astype(int)
        report = classification_report(
            truth, 
            predicted_binary, 
            output_dict=True)

        # Convert to DataFrame for nicer formatting
        report_df = pd.DataFrame(report).transpose()

        # Round for readability
        report_df = report_df.round(4)
        print(report_df)

        # Save report to txt
        if val_report_txt:
            with open(val_report_txt, 'a') as f:
                f.write(f'Mode: {mode}\n')
                f.write(f'Val Fold: {cfg.experiment.run_fold}\n')
                f.write(f'Model: {model_path}\n')
                f.write(f'Loss: {val_loss:.4f}\n')
                f.write(report_df.to_string())
                f.write('\n\n')
    

if __name__ == '__main__':
    print('[ √ ] Landmark!')

    args, cfg = parse_args()

    print('[ √ ] cfg.experiment.file: {}'.format(cfg.experiment.file))
    print('[ √ ] cfg.experiment.run_fold: {}'.format(cfg.experiment.run_fold))
    train_fold = (cfg.experiment.run_fold + 1) % 5
    print('[ √ ] train_fold: {}'.format(train_fold))

    if DEVICE.type == 'cuda':
        print(f"[ i ] Using GPU: {torch.cuda.get_device_name(DEVICE)}")
    else:
        print("[ i ] Using CPU")

    # Get csv file
    csv_file = cfg.experiment.file
    df = pd.read_csv('dataloaders/split/'+csv_file)

    df_train = df[df.fold == train_fold].copy()
    df_val = df[df.fold == cfg.experiment.run_fold].copy()

    print('[ √ ] Using transformation: {}, image size: {}'.format(
        cfg.transform.val_name, 
        cfg.transform.size))
    
    # if tta_tfms:
    #     val_tfms = tta_tfms
    if cfg.transform.val_name == 'None':
        val_tfms = None
    else:
        val_tfms = get_tfms(cfg.transform.val_name)

    # Get dataset
    ds_train = RANZERDataset(
        df=df_train, tfms=val_tfms, cfg=cfg, mode='valid')
    ds_val = RANZERDataset(
        df=df_val, tfms=val_tfms, cfg=cfg, mode='valid')
    
    # Get dataloader
    dl_train = DataLoader(
        dataset=ds_train, 
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.transform.num_preprocessor, 
        pin_memory=False)
    dl_val = DataLoader(
        dataset=ds_val, 
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.transform.num_preprocessor, 
        pin_memory=False)
    print('[ i ] train dataset size: {}'.format(len(ds_train)))
    print('[ i ] val dataset size: {}'.format(len(ds_val)))
    print('[ i ] train dataloader size: {}'.format(len(dl_train)))
    print('[ i ] val dataloader size: {}'.format(len(dl_val)))
    print('[ i ] batch size: {}'.format(cfg.eval.batch_size))

    # Get weights path
    weights_path = args.predict_weights_path
    print('[ i ] weights_path: {}'.format(weights_path))

    # Get the files from the weights path
    if os.path.isdir(weights_path):
        files = os.listdir(weights_path)
        files = [f for f in files if f.endswith('.pth')]
        files.sort()
    else:
        raise FileNotFoundError(f"Directory {weights_path} does not exist.")

    epoch = 0
    while True:
        # Get the file name
        f = 'f{fold}_epoch-{epoch}.pth'.format(
            fold=cfg.experiment.run_fold, 
            epoch=epoch)
        print('[ i ] file: {}'.format(f))

        if f not in files:
            print('No more files of this fold to validate.')
            break

        # Load model
        model_path = os.path.join(weights_path, f)
        model = get_model(cfg)
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE)

        # Validate
        basic_validate(
            model, dl_train, cfg, args.val_report_txt, model_path, mode='train')
        basic_validate(
            model, dl_val, cfg, args.val_report_txt, model_path, mode='valid')

        # Next epoch
        epoch += 1

    print('[ √ ] Validated!')
    