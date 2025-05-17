from utils import *
import tqdm
from configs import Config
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
import gc

try:
    from apex import amp
except:
    pass


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(
    cfg: Config, 
    model, 
    train_dl, 
    valid_dl, 
    optimizer, 
    save_path, 
    scheduler, 
    writer, 
    tune=None):

    # Positive weight
    pos_weight = torch.ones(19) / cfg.loss.pos_weight
    pos_weight = pos_weight.to(DEVICE)

    print('[ √ ] Training')
    
    # Start training
    try:
        optimizer.zero_grad()
        
        for epoch in range(cfg.train.num_epochs):
            # First we update batch sampler if exist
            if cfg.experiment.batch_sampler:
                train_dl.batch_sampler.update_miu(
                    cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor
                )
                print('[ W ] set miu to {}'.format(cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor))

            # Update scheduler if StepLR
            if scheduler and cfg.scheduler.name in ['StepLR']:
                scheduler.step(epoch)

            # Progress bar (tune is for tuning)
            if not tune:
                tq = tqdm.tqdm(train_dl)
            else:
                tq = train_dl
            
            basic_lr = optimizer.param_groups[0]['lr']
            losses = []
            
            # Scaler for mixed precision training
            if cfg.basic.amp == 'Native':
                scaler = torch.cuda.amp.GradScaler()
            
            model.train() # Set model to training mode

            for i, (ipt, lbl, img_lbl, conf_lbl, conf_img_lbl, cnt) in enumerate(tq):

                # DEBUG: Print each value and its shape and type
                if cfg.basic.debug:
                    print(f'ipt: {ipt.shape}, {ipt.dtype}')
                    print(f'lbl: {lbl.shape}, {lbl.dtype}')
                    print(f'image_lbl: {img_lbl.shape}, {img_lbl.dtype}')
                    print(f'conf_lbl: {conf_lbl.shape}, {conf_lbl.dtype}')
                    print(f'conf_img_lbl: {conf_img_lbl.shape}, {conf_img_lbl.dtype}')
                    print(f'cnt: {cnt.shape}, {cnt.dtype}')
                
                # DEBUG:
                if cfg.basic.debug and i == 10:
                    break

                if cfg.experiment.count > 0:
                    ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
                    lbl = lbl.view(-1, lbl.shape[-1])
                    conf_lbl = conf_lbl.view(-1, conf_lbl.shape[-1])
                
                # DEBUG: Print each value and its shape and type
                if cfg.basic.debug:
                    print(f'ipt: {ipt.shape}, {ipt.dtype}')
                    print(f'lbl: {lbl.shape}, {lbl.dtype}')
                    print(f'image_lbl: {img_lbl.shape}, {img_lbl.dtype}')
                    print(f'conf_lbl: {conf_lbl.shape}, {conf_lbl.dtype}')
                    print(f'conf_img_lbl: {conf_img_lbl.shape}, {conf_img_lbl.dtype}')
                    print(f'cnt: {cnt.shape}, {cnt.dtype}')

                # Warm up lr initial
                if cfg.scheduler.warm_up and epoch == 0:
                    length = len(train_dl)
                    initial_lr = basic_lr / length
                    optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)
                
                # Move data to device
                ipt, lbl, img_lbl, conf_lbl, conf_img_lbl = [
                    ipt.to(DEVICE), 
                    lbl.to(DEVICE), 
                    img_lbl.to(DEVICE), 
                    conf_lbl.to(DEVICE), 
                    conf_img_lbl.to(DEVICE)]

                r = np.random.rand(1) # Why is this needed?

                # DEBUG: Print each value
                if cfg.basic.debug:
                    print(f"r: {r}")
                    print(f"cfg.train.cutmix: {cfg.train.cutmix}")
                    print(f"cfg.train.beta: {cfg.train.beta}")
                    print(f"cfg.train.cutmix_prob: {cfg.train.cutmix_prob}")

                if cfg.train.cutmix and cfg.train.beta > 0 and r < cfg.train.cutmix_prob:
                    raise NotImplementedError("cfg.train.cutmix is not implemented.")
                else:
                    if cfg.basic.amp == 'Native':
                        with torch.cuda.amp.autocast():
                            if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                                raise NotImplementedError("Model arc or cos is not implemented.")
                            else:
                                cell, img = model(ipt, cnt)

                            # Cell loss
                            if cfg.train.cell_pred_as_labels:
                                loss_cell = F.binary_cross_entropy_with_logits(
                                    cell, conf_lbl,
                                    pos_weight=pos_weight,
                                    reduction='none')
                            else:
                                loss_cell = F.binary_cross_entropy_with_logits(
                                    cell, lbl,
                                    pos_weight=pos_weight,
                                    reduction='none')

                            # Image loss
                            loss_img = F.binary_cross_entropy_with_logits(
                                img, img_lbl,
                                reduction='none')

                            # DEBUG: Print each value and its shape and type
                            if cfg.basic.debug:
                                print(f"loss_cell: {loss_cell.shape}")
                                print(f"loss_img: {loss_img.shape}")

                            if cfg.train.conf_aware:
                                # Cell conformity
                                conformity = 1 - torch.abs(lbl - conf_lbl)
                                w = cfg.train.conf_alpha * conformity ** cfg.train.conf_gamma

                                # Image conformity
                                img_conformity = 1 - torch.abs(img_lbl - conf_img_lbl)
                                img_w = cfg.train.conf_alpha * img_conformity ** cfg.train.conf_gamma

                                # DEBUG: Print each value and its shape and type
                                if cfg.basic.debug:
                                    print(f"conformity shape: {conformity.shape}")
                                    print(f"w shape: {w.shape}")
                                    print(f"img_conformity shape: {img_conformity.shape}")
                                    print(f"img_w shape: {img_w.shape}")

                                weighted_loss_cell = loss_cell * w
                                weighted_loss_img = loss_img * img_w

                                del conformity, w, img_conformity, img_w
                                gc.collect()
                            else:
                                weighted_loss_cell = loss_cell
                                weighted_loss_img = loss_img

                            # DEBUG: Print each value and its shape and type
                            if cfg.basic.debug:
                                print(f"weighted_loss_cell: {weighted_loss_cell.shape}")
                                print(f"weighted_loss_img: {weighted_loss_img.shape}")
                            
                            # Calculate mean if needed
                            if not len(weighted_loss_cell.shape) == 0:
                                weighted_loss_cell = weighted_loss_cell.mean()
                            if not len(weighted_loss_img.shape) == 0:
                                weighted_loss_img = weighted_loss_img.mean()
                            
                            # Calculate total loss
                            loss = cfg.loss.cellweight * weighted_loss_cell + weighted_loss_img
                            losses.append(loss.item())

                            del ipt, lbl, img_lbl, conf_lbl, conf_img_lbl, cnt
                            del cell, img, loss_cell, loss_img, weighted_loss_cell, weighted_loss_img
                            gc.collect()
                    else:
                        raise NotImplementedError("cfg.basic.amp is not Native.")

                # Backward pass
                if cfg.basic.amp == 'Native':
                    scaler.scale(loss).backward()
                elif not cfg.basic.amp == 'None':
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                del loss
                gc.collect()

                # Optimizer step
                # print(f"cfg.train.clip: {cfg.train.clip}")
                if i % cfg.optimizer.step == 0:
                    if cfg.basic.amp == 'Native':
                        if cfg.train.clip:
                            scaler.unscale_(optimizer)
                            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        if cfg.train.clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        optimizer.step()
                        optimizer.zero_grad()

                # If CyclicLR, OneCycleLR, or CosineAnnealingLR, step the scheduler
                if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR']:
                    if epoch == 0 and cfg.scheduler.warm_up:
                        pass
                    else:
                        scheduler.step()

                if not tune:
                    # tq.set_postfix(loss=np.array(losses).mean(), lr=optimizer.param_groups[0]['lr'])

                    # Get CPU and RAM usage
                    cpu_mem = psutil.virtual_memory()
                    cpu_mem_used = cpu_mem.used / (1024 ** 3)  # Convert to GB
                    cpu_mem_free = cpu_mem.available / (1024 ** 3)  # Convert to GB

                    # Get GPU usage
                    gpu_mem_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    gpu_mem_free = (
                        torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
                    ) / (1024 ** 3)
                    
                    # Update tqdm description
                    tq.set_description(
                        f"[Epoch {epoch}] | Loss: {np.array(losses).mean():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                        f"GPU: {gpu_mem_used:.2f}GB / {gpu_mem_free:.2f}GB | "
                        f"CPU: {cpu_mem_used:.2f}GB / {cpu_mem_free:.2f}GB"
                    )

                    del cpu_mem, cpu_mem_used, cpu_mem_free, gpu_mem_used, gpu_mem_free
                    gc.collect()

            # Validation
            validate_loss = validate(model, valid_dl, cfg, writer)
            print(('[ √ ] epochs: {}, train loss: {:.4f}, valid loss: {:.4f}').format(
                epoch, np.array(losses).mean(), validate_loss))
            
            del validate_loss
            gc.collect()
            
            # Write to tensorboard
            writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)
            writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)

            # Write to log
            with open(save_path / 'train.log', 'a') as fp:
                fp.write('{}\t{:.8f}\t{:.4f}\t{:.4f}\n'.format(
                    epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean(), validate_loss))
            
            # Save model
            torch.save(model.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                cfg.experiment.run_fold, epoch))
            
            # If using ReduceLROnPlateau, step the scheduler
            if scheduler and cfg.scheduler.name in ['ReduceLROnPlateau']:
                scheduler.step(validate_loss)
    
    except KeyboardInterrupt:
        print('[ X ] Ctrl + c, QUIT')
        torch.save(model.state_dict(), save_path / 'checkpoints/quit_f{}.pth'.format(cfg.experiment.run_fold))


def validate(
    model, 
    valid_dl, 
    cfg,
    writer):
    
    print('[ √ ] Validation')

    # Move model to device
    model.to(DEVICE)

    # Set model to evaluation mode
    model.eval()

    # Set tqdm progress bar
    tq = tqdm.tqdm(valid_dl)

    with torch.no_grad():
        results = []
        losses, predicted, truth = [], [], []
        for i, (ipt, lbl, img_lbl, n_cell) in enumerate(tq):

            # DEBUG:
            if cfg.basic.debug and i == 10:
                break

            ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
            img_lbl = img_lbl.view(-1, 19)
            ipt, img_lbl = ipt.to(DEVICE), img_lbl.to(DEVICE)

            # DEBUG: Print each value and its shape and type
            if cfg.basic.debug:
                print(f"ipt: {ipt.shape}, {ipt.dtype}")
                print(f"img_lbl: {img_lbl.shape}, {img_lbl.dtype}")

            # Get logits and loss
            if cfg.basic.amp == 'Native':
                with torch.cuda.amp.autocast():
                    _, output = model(ipt, n_cell)
                    loss = F.binary_cross_entropy_with_logits(
                        output, img_lbl,
                        reduction='none')
                    if not len(loss.shape) == 0:
                        loss = loss.mean()
                    output = output.float()
            else:
                raise NotImplementedError("cfg.basic.amp is not Native.")
            
            # Append loss to list
            losses.append(loss.item())

            # Predictions
            pred = torch.sigmoid(output.cpu()).numpy()

            # DEBUG: Print each value and its shape and type
            if cfg.basic.debug:
                print(f"predicted: {pred.shape}, {pred.dtype}")
                print(f"truth: {img_lbl.shape}, {img_lbl.dtype}")

            # Append to lists
            predicted.append(pred)
            truth.append(img_lbl.cpu().numpy())
            
            results.append({
                'step': i,
                'loss': loss.item(),
            })
        
        # Concatenate results and calculate validation loss
        predicted = np.concatenate(predicted)
        truth = np.concatenate(truth)
        val_loss = np.array(losses).mean()

        # DEBUG: Print each value and its shape and type
        if cfg.basic.debug:
            print(f"val_loss: {val_loss}")
            print(f"predicted: {predicted}")
            print(f"predicted: {predicted.shape}, {predicted.dtype}")
            print(f"truth: {truth}")
            print(f"truth: {truth.shape}, {truth.dtype}")
        
        # Classification report
        predicted_binary = (predicted > 0.5).astype(int)
        report = classification_report(
            truth, 
            predicted_binary, 
            output_dict=True)

        # Convert to DataFrame for nicer formatting
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(4)
        print(report_df)

        # Write to tensorboard
        writer.add_scalar('valid_f{}/loss'.format(cfg.experiment.run_fold), val_loss, cfg.train.num_epochs)

        return val_loss

