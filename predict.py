import os
from utils import parse_args
from models import get_model
import torch
import pandas as pd
import tqdm

from dataloaders.datasets import GetPredictionsDataset
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')


def basic_predict(mdl, dl, cfg, tune=None):
    print("[ √ ] Basic predict")

    # Save to CSV
    save_path = 'predictions/' + cfg.model.name + '_fold_' + str(cfg.experiment.run_fold) + '.csv'

    # Check if file exists
    if os.path.exists(save_path):
        raise FileExistsError(f"File {save_path} already exists.")

    if not tune:
        print("Not tune.")
        tq = tqdm.tqdm(dl, disable=False)
    else:
        print("Yes tune.")
        tq = dl

    mdl.eval()
    results = []
    with torch.no_grad():
        for i, (ipt, mask, lbl, image_lbl, n_cell, filename) in enumerate(tq):

            # DEBUG: Print each value and its shape and type
            # print(f'ipt: {ipt.shape}, {ipt.dtype}')
            # print(f'mask: {mask.shape}, {mask.dtype}')
            # print(f'lbl: {lbl.shape}, {lbl.dtype}')
            # print(f'image_lbl: {image_lbl.shape}, {image_lbl.dtype}')
            # print(f'n_cell: {n_cell.shape}, {n_cell.dtype}')
            # print(f'filename: {type(filename)}')

            n_cell = n_cell[0]
            filename = filename[0]

            # print(f'n_cell: {n_cell}')
            # print(f'filename: {filename}')

            # DEBUG: Break
            # if i == 10:
            #     break

            ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
            lbl = lbl.view(-1, lbl.shape[-1])
            
            if torch.cuda.is_available():
                ipt, lbl = ipt.cuda(), lbl.cuda()

            if cfg.basic.amp == 'Native':
                with torch.cuda.amp.autocast():
                    if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                        print("Model arc or cos.")
                        raise NotImplementedError()
                    else:
                        cell_logits, image_logits = mdl(ipt, n_cell)
                        cell_probs = torch.sigmoid(cell_logits).cpu().numpy()
                        image_probs = torch.sigmoid(image_logits).cpu().numpy()
                        cell_logits = cell_logits.cpu().numpy()
                        image_logits = image_logits.cpu().numpy()
            else:
                print("cfg.basic.amp is not Native.")
                raise NotImplementedError()
            
            # Add cell-level output
            for j in range(int(n_cell)):
                results.append({
                    'filename': f'{filename}_{j+1}',
                    **{f'logit_{k}': cell_logits[j, k] for k in range(cell_logits.shape[1])},
                    **{f'prob_{k}': cell_probs[j, k] for k in range(cell_probs.shape[1])},
                    'type': 'cell'
                })

            # Add image-level output
            results.append({
                'filename': filename,
                **{f'logit_{k}': image_logits[0, k] for k in range(image_logits.shape[1])},
                **{f'prob_{k}': image_probs[0, k] for k in range(image_probs.shape[1])},
                'type': 'image'
            })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Saved validation outputs to {save_path}")


if __name__ == '__main__':
    print('[ √ ] Landmark!')
    args, cfg = parse_args()

    # Get csv file
    csv_file = cfg.experiment.file
    print('cfg.experiment.file: {}'.format(cfg.experiment.file))

    df = pd.read_csv('dataloaders/split/'+csv_file)
    print('df shape: {}'.format(df.shape))
    print('df head: {}'.format(df.head()))

    df = df[df['fold'] != cfg.experiment.run_fold]
    print('df shape: {}'.format(df.shape))
    print('df length: {}'.format(len(df)))

    # Get dataset
    ds = GetPredictionsDataset(
        df=df, 
        tfms=None, 
        cfg=cfg)
    
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

    # loading model
    model = get_model(cfg)
    model.load_state_dict(torch.load(weights_path,
        map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu', 'cuda:2': 'cpu', 'cuda:3': 'cpu'}
    ))
    print('[ i ] Model loaded')
    
    if not torch.cuda.is_available():
        print('[ W ] cpu prediction')
        model = model.cpu()
    elif len(cfg.basic.GPU) == 1:
        print('[ W ] single gpu prediction the gpus is {}'.format(cfg.basic.GPU))
        model = model.cuda()
    else:
        print('[ W ] dp prediction the gpus is {}'.format(cfg.basic.GPU))
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[int(x) for x in cfg.basic.GPU])

    # predict
    basic_predict(model, dl, cfg)
    print('[ √ ] Predict done.')
    
