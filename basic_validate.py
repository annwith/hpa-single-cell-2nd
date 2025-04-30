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


def basic_validate(model, dl, cfg, val_report_txt, model_path):
    print('[ √ ] Validation')
    print('model_path: {}'.format(model_path))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model.to(device)
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        results = []
        losses, predicted, truth = [], [], []
        for i, (ipt, mask, lbl, img_lbl, n_cell) in enumerate(tqdm(dl)):

            ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
            exp_label = img_lbl.view(-1, 19)
            ipt, exp_label = ipt.to(device), exp_label.to(device)

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
        # Convert to binary predictions
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
                f.write(f'Model: {model_path}\n')
                f.write(f'Validation Loss: {val_loss:.4f}\n')
                f.write(report_df.to_string())
                f.write('\n\n')
    

if __name__ == '__main__':
    print('[ √ ] Landmark!')
    args, cfg = parse_args()

    # Get csv file
    csv_file = cfg.experiment.file
    print('cfg.experiment.file: {}'.format(cfg.experiment.file))

    df = pd.read_csv('dataloaders/split/'+csv_file)
    print('df shape: {}'.format(df.shape))
    print('df head: {}'.format(df.head()))

    df = df[df.fold == cfg.experiment.run_fold]
    print('df shape: {}'.format(df.shape))

    print('[ √ ] Using transformation: {}, image size: {}'.format(
        cfg.transform.val_name, 
        cfg.transform.size
    ))
    # if tta_tfms:
    #     val_tfms = tta_tfms
    if cfg.transform.val_name == 'None':
        val_tfms = None
    else:
        val_tfms = get_tfms(cfg.transform.val_name)

    # Get dataset
    ds = RANZERDataset(
        df=df, tfms=val_tfms, cfg=cfg, mode='valid')
    
    # Get dataloader
    dl = DataLoader(
        dataset=ds, 
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.transform.num_preprocessor, 
        pin_memory=False)
    print('dataset size: {}'.format(len(ds)))
    print('dataloader size: {}'.format(len(dl)))
    print('batch size: {}'.format(cfg.eval.batch_size))

    # Get weights path
    weights_path = args.predict_weights_path
    print('weights_path: {}'.format(weights_path))

    # Get the files from the weights path
    if os.path.isdir(weights_path):
        files = os.listdir(weights_path)
        files = [f for f in files if f.endswith('.pth')]
        files.sort()

    for f in files:
        print('file: {}'.format(f))
        if f.endswith('.pth'):
            model_path = os.path.join(weights_path, f)
            print('model_path: {}'.format(model_path))
    
        # loading model
        model = get_model(cfg)
        model.load_state_dict(torch.load(model_path,
            map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu', 'cuda:2': 'cpu', 'cuda:3': 'cpu'}
        ))
        model = model.cpu()
        print('model loaded')
        
        if len(cfg.basic.GPU) == 1:
            print('[ W ] single gpu prediction the gpus is {}'.format(cfg.basic.GPU))
            # torch.cuda.set_device(cfg.basic.GPU)
            model = model.cuda()
        else:
            print('[ W ] dp prediction the gpus is {}'.format(cfg.basic.GPU))
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=[int(x) for x in cfg.basic.GPU])

        # predict
        basic_validate(model, dl, cfg, args.val_report_txt, model_path)

    print('validated')
    