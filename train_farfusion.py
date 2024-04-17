# python train_farfusion.py config=config/1_belfusion_vae.yaml name=TransformerVAE arch.args.online=False
# python train_farfusion.py config=config/2_belfusion_ldm.yaml name=LatentMLPMatcher arch.args.online=False
# python evaluate.py  --resume ./results/LatentMLPMatcher/checkpoint_best.pth  --gpu-ids 0  --split val
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import logging
import model as module_arch
from metric import *
import model.losses as module_loss
from functools import partial
from utils import load_config, store_config, AverageMeter
from dataset import get_dataloader
import wandb
from datetime import datetime
import random

from accelerate import Accelerator
accelerator = Accelerator()

def pprint(f, msg):
    print(msg)
    f.write(msg + "\n")


def evaluate(cfg, pred_list_em, speaker_em, listener_em, epoch):
    assert listener_em.shape[0] == speaker_em.shape[0], "speaker and listener emotion must have the same shape"
    assert listener_em.shape[0] == pred_list_em.shape[0], "predictions and listener emotion must have the same shape"

    # only the fast diversity metrics ploted often
    metrics = {
        # APPROPRIATENESS METRICS
        #"FRDist": compute_FRD(data_path, pred_list_em[:,0], listener_em), # FRDist (1) --> slow, ~3 mins
        #"FRCorr": compute_FRC(data_path, pred_list_em[:,0], listener_em), # FRCorr (2) --> slow, ~3 mins

        # DIVERSITY METRICS --> all very fast, compatible with validation in training loop
        "FRVar": compute_FRVar(pred_list_em), # FRVar (1) --> intra-variance (among all frames in a prediction),
        "FRDiv": compute_s_mse(pred_list_em), # FRDiv (2) --> inter-variance (among all predictions for the same speaker),
        "FRDvs": compute_FRDvs(pred_list_em), # FRDvs (3) --> diversity among reactions generated from different speaker behaviours
        
        # OTHER METRICS
        # FRRea (realism)
        #"FRSyn": compute_TLCC(pred_list_em, speaker_em), # FRSyn (synchrony) --> EXTREMELY slow, ~1.5h
    }
    return metrics


def update_averagemeter_from_dict(results, meters):
    # if meters is empty, it will be initialized. If not, it will be updated
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            value = value.item()

        if key in meters:
            meters[key].update(value)
        else:
            meters[key] = AverageMeter()
            meters[key].update(value)

# Train
def train(cfg, model, train_loader, optimizer, criterion, device):
    losses_meters = {}

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()
    for batch_idx, (s_emotion, s_3dmm, l_emotion, l_3dmm, l_reference, _, _) in enumerate(tqdm(train_loader)):

        optimizer.zero_grad()

        prediction = model(listener_3dmm=l_3dmm, listener_emotion=l_emotion, speaker_3dmm=s_3dmm, speaker_emotion=s_emotion)
        prediction["split"] = 'train'

        losses = criterion(**prediction)
        update_averagemeter_from_dict(losses, losses_meters)
        accelerator.backward(losses["loss"])
        optimizer.step()

    return {key: losses_meters[key].avg for key in losses_meters}


def validate(cfg, model, val_loader, criterion, device, epoch):
    num_preds = cfg.trainer.get("num_preds", 10) # number of predictions to make
    losses_meters = {}

    model, val_loader = accelerator.prepare(model, val_loader)
    model.eval()

    with torch.no_grad():
        all_predictions, speaker_emotions, listener_emotions = [], [], []
        for batch_idx, (s_emotion_, s_3dmm_, l_emotion_, l_3dmm_, l_reference, _, _) in enumerate(tqdm(val_loader)):

            
            prediction = model(listener_3dmm=l_3dmm_, listener_emotion=l_emotion_, speaker_3dmm=s_3dmm_, speaker_emotion=s_emotion_) # [B, S, D]
            prediction["split"] = 'val'

            losses = criterion(**prediction)
            update_averagemeter_from_dict(losses, losses_meters)


    return {"val_" + key: losses_meters[key].avg for key in losses_meters}


def main():
    # load yaml config
    cfg = load_config()
    cfg.trainer.out_dir = os.path.join(cfg.trainer.out_dir, cfg["name"])
    os.makedirs(cfg.trainer.out_dir, exist_ok=True)
    store_config(cfg)
    f = open(os.path.join(cfg.trainer.out_dir, "log.txt"), "w")

    start_epoch = 0
    pprint(f, str(cfg.dataset))
    pprint(f, str(cfg.validation_dataset))
    
    train_loader = get_dataloader(cfg.dataset, cfg.dataset.split, load_emotion_s=True, load_emotion_l=True, load_3dmm_s=True, load_3dmm_l=True, load_ref=False, repeat_mirrored=True)

    valid_loader = get_dataloader(cfg.validation_dataset, cfg.validation_dataset.split, load_emotion_s=True, load_emotion_l=True, load_3dmm_s=True, load_3dmm_l=True, load_ref=False, repeat_mirrored=True)

    pprint(f, 'Train dataset: {} samples'.format(len(train_loader.dataset)))
    pprint(f, 'Valid dataset: {} samples'.format(len(valid_loader.dataset)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(module_arch, cfg.arch.type)(cfg.arch.args)

    # model = model.to(device)
    pprint(f, 'Model {} : params: {:4f}M'.format(cfg.arch.type, sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    criterion = partial(getattr(module_loss, cfg.loss.type), **cfg.loss.args)
    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    if cfg.trainer.resume != None:
        checkpoint_path = cfg.trainer.resume
        pprint(f, "Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoints['state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoints['optimizer'])

    last_epoch_stored = 199
    val_loss = 0
    val_metrics = None
    log_dict = {}
    val_loss_max = 999999

    for epoch in range(start_epoch, cfg.trainer.epochs):

        # =================== TRAIN ===================
        train_losses = train(cfg, model, train_loader, optimizer, criterion, device)
        log_dict.update(train_losses)

        # =================== VALIDATION ===================
        if (cfg.trainer.val_period > 0 and (epoch + 1) % cfg.trainer.val_period == 0) or epoch == start_epoch:
            val_losses = validate(cfg, model, valid_loader, criterion, device, epoch)
            log_dict.update(val_losses)

            print(f"-----------------------Updated best model at epoch_{epoch+1} !-----------------------")
            checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }
            os.makedirs(cfg.trainer.out_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(cfg.trainer.out_dir, 'checkpoint_best.pth'))

        # =================== log ===================
        log_message = 'epoch: {}'.format(epoch)
        for key, value in log_dict.items():
            log_message += ", {}: {:.6f}".format(key, value)
        pprint(f, log_message)
        f.flush()

    f.close()

# ---------------------------------------------------------------------------------


if __name__=="__main__":
    torch.manual_seed(6)
    torch.cuda.manual_seed(6)
    np.random.seed(6)
    random.seed(6)
    main()