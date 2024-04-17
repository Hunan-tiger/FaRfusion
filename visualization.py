import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from render import Render
from metric import *
from dataset import get_dataloader
from utils import load_config
import model as module_arch

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--dataset-path', default="/public_bme/data/v-lijm/REACT_2024", type=str, help="dataset path")
    parser.add_argument('--split', type=str, help="split of dataset", choices=["val", "test"], required=True)
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    parser.add_argument('--online', action='store_true', help='online / offline method')
    parser.add_argument('--outdir', default="./results", type=str, help="result dir")
    parser.add_argument('--device', default='cuda', type=str, help="device: cuda / cpu")
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    args = parser.parse_args()
    return args

# Evaluating
def val(args, model, val_loader, render):
    model.eval()

    out_dir = os.path.join(args.outdir, args.split)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for batch_idx, (speaker_emotion, speaker_3dmm, listener_emotion, listener_3dmm, listener_references, speaker_video_data, listener_video_data) in enumerate(tqdm(val_loader)):
        if torch.cuda.is_available():
                speaker_emotion, speaker_3dmm, listener_emotion, listener_3dmm = speaker_emotion.cuda(), speaker_3dmm.cuda(), listener_emotion.cuda(), listener_3dmm.cuda()
                speaker_video_data, listener_video_data, listener_references = speaker_video_data.cuda(), listener_video_data.cuda(), listener_references.cuda()

        with torch.no_grad():
            prediction = model(listener_3dmm=listener_3dmm, listener_emotion=listener_emotion, speaker_3dmm=speaker_3dmm, speaker_emotion=speaker_emotion)

            listener_3dmm_out = prediction["pred_3dmm"]

            B = speaker_video_data.shape[0]
            if (batch_idx % 25) == 0:
                for bs in range(B):
                    render.rendering_for_fid(out_dir, "{}_b{}_ind{}".format(args.split, str(batch_idx + 1), str(bs + 1)),
                            listener_3dmm_out[bs], speaker_video_data[bs], listener_references[bs], listener_video_data[bs,:750])
    return 

def main(args):
    checkpoint_path = args.resume
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    if not os.path.exists(config_path): # args-based loading --> Trans-VAE by default
        pass
    else: # config-based loading --> BeLFusion
        cfg = load_config(config_path)
        dataset_cfg = cfg.validation_dataset if args.split == "val" else cfg.test_dataset
        dataset_cfg.dataset_path = args.dataset_path
        val_loader = get_dataloader(dataset_cfg, args.split, load_emotion_s=True, load_emotion_l=True, load_3dmm_s=True, load_3dmm_l=True, load_ref=True)

        model = getattr(module_arch, cfg.arch.type)(cfg.arch.args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    if args.resume != '': #  resume from a checkpoint
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model = model.cuda()
        render = Render('cuda')
    else:
        render = Render()

    val(args, model, val_loader, render)


if __name__=="__main__":
    args = parse_arg()
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    main(args)

