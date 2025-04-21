from utils import parse_args, prepare_for_result
from dataloaders import get_dataloader
from models import get_model
from optimizers import get_optimizer
from basic_train_cp import basic_train_conf_aware
from scheduler import get_scheduler
from utils import load_matched_state
from torch.utils.tensorboard import SummaryWriter
import torch
try:
    from apex import amp
except:
    pass

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    print('[ √ ] Landmark!')
    args, cfg = parse_args()
    
    result_path = prepare_for_result(cfg)
    writer = SummaryWriter(log_dir=result_path)
    cfg.dump_json(result_path / 'config.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[ i ] Using device: {}'.format(device))

    # Multiple fold training
    if cfg.experiment.run_fold == -1:
        for i in range(cfg.experiment.fold):
            torch.cuda.empty_cache()
            
            print('[ ! ] Full fold coverage training! for fold: {}'.format(i))
            cfg.experiment.run_fold = i
            
            train_dl, valid_dl, test_dl = get_dataloader(cfg)(cfg).get_dataloader()
            print('[ i ] The length of train_dl is {}, valid dl is {}'.format(len(train_dl), len(valid_dl)))
            
            model = get_model(cfg)
            model = model.to(device)
            if not cfg.model.from_checkpoint == 'none':
                print('[ ! ] loading model from checkpoint: {}'.format(cfg.model.from_checkpoint))
                load_matched_state(model, torch.load(cfg.model.from_checkpoint))
            
            optimizer = get_optimizer(model, cfg)
            print('[ i ] Model: {}, loss_func: {}, optimizer: {}'.format(
                cfg.model.name, cfg.loss.name, cfg.optimizer.name))
            
            if not cfg.basic.amp == 'None' and not cfg.basic.amp == 'Native':
                print('[ i ] Call apex\'s initialize')
                model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.basic.amp)
            
            if not cfg.scheduler.name == 'none':
                scheduler = get_scheduler(cfg, optimizer, len(train_dl))
            else:
                scheduler = None
            
            if len(cfg.basic.GPU) > 1:
                model = torch.nn.DataParallel(model)
        
        if cfg.train.conf_aware:
            print('[ ! ] Use confidence aware training')
            basic_train_conf_aware(
                cfg, model, train_dl, valid_dl, optimizer, result_path, scheduler, writer)
        else:
            raise NotImplementedError('Normal training is not implemented on this script.')
    
    # Single fold training
    else:
        print('[ ! ] Single fold training! for fold: {}'.format(cfg.experiment.run_fold))

        train_dl, valid_dl, test_dl = get_dataloader(cfg)(cfg).get_dataloader()
        print('[ i ] The length of train_dl is {}, valid dl is {}'.format(len(train_dl), len(valid_dl)))

        model = get_model(cfg)
        model = model.to(device)    
        if not cfg.model.from_checkpoint == 'none':
            print('[ ! ] loading model from checkpoint: {}'.format(cfg.model.from_checkpoint))
            load_matched_state(model, torch.load(cfg.model.from_checkpoint, map_location='cpu'))
        
        optimizer = get_optimizer(model, cfg)
        print('[ i ] Model: {}, loss_func: {}, optimizer: {}'.format(cfg.model.name, cfg.loss.name, cfg.optimizer.name))

        if not cfg.basic.amp == 'None' and not cfg.basic.amp == 'Native':
            print('[ i ] Call apex\'s initialize')
            model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.basic.amp)

        if not cfg.scheduler.name == 'none':
            scheduler = get_scheduler(cfg, optimizer, len(train_dl))
        else:
            scheduler = None

        if len(cfg.basic.GPU) > 1:
            model = torch.nn.DataParallel(model)

        if cfg.train.conf_aware:
            print('[ ! ] Use confidence aware training')
            basic_train_conf_aware(
                cfg, model, train_dl, valid_dl, optimizer, result_path, scheduler, writer)
        else:
            raise NotImplementedError('Normal training is not implemented on this script.')
