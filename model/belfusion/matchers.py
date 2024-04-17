import torch
import torch.nn as nn
import os
import numpy as np
from model.belfusion.mlp_diff import MLPSkipNet, Activation
from model.belfusion.unet_diff import GDUnet_Latent
from utils import load_config_from_file
import model as module_arch
from model.belfusion.diffusion import LatentDiffusion
from model.belfusion.resample import UniformSampler

class BaseLatentModel(nn.Module):
    def __init__(self, cfg, emb_size=None, autoencoder_path=None,
                 freeze_emotion_encoder=True, cond_embed_dim=None,
                ):
        super(BaseLatentModel, self).__init__()

        self.diffusion = LatentDiffusion(cfg) # TODO init the diffusion object here
        self.schedule_sampler = UniformSampler(self.diffusion)
        
        self.emb_size = emb_size
        self.cond_embed_dim = cond_embed_dim

        def_dtype = torch.get_default_dtype()
        # load auxiliary model (emotion embedder)
        configpath = os.path.join(os.path.dirname(autoencoder_path), "config.yaml")
        cfg = load_config_from_file(configpath)
        self.embed_ = getattr(module_arch, cfg.arch.type)(cfg.arch.args)

        checkpoint = torch.load(autoencoder_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        self.embed_.load_state_dict(state_dict)

        if freeze_emotion_encoder:
            for para in self.embed_.parameters():
                para.requires_grad = False

        self.linear = nn.Sequential(
            nn.Flatten(-2, -1),
            nn.Linear(emb_size*750, self.cond_embed_dim),
            nn.GroupNorm(32, self.cond_embed_dim),
        )
    
        torch.set_default_dtype(def_dtype) # config loader changes this


    def encode(self, coeff, emo):
        return self.embed_.encode(coeff, emo)

    def decode(self, em_emb):
        return self.embed_.decode(em_emb)
    
    def cond_avg(self, cond):
        '''
        cond: [batch_size, seq_length, features]
        '''
        cond = self.linear(cond)
        return cond

    def get_emb_size(self):
        return self.emb_size

    def forward(self, pred, timesteps, seq_em):
        raise NotImplementedError("This is an abstract class.")
    
    # override checkpointing
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model = self.model.to(device)
        self.embed_ = self.embed_.to(device)
        super().to(device)
        return self
    
    def cuda(self):
        return self.to(torch.device("cuda"))
    
    # override eval and train
    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()


class LatentMLPMatcher(BaseLatentModel):

    def __init__(self, cfg):
        super(LatentMLPMatcher, self).__init__(cfg, 
                emb_size=cfg.emb_length, autoencoder_path=cfg.autoencoder_path, freeze_emotion_encoder=cfg.freeze_emotion_encoder, cond_embed_dim=cfg.cond_embed_dim)

        assert cfg.emb_length is not None, "Embedding length must be specified."
        self.emb_length = cfg.emb_length # TODO multiply by 2 if using speaker + listener embeddings

        self.window_size = cfg.window_size
        self.cond_embed_dim = cfg.cond_embed_dim
        self.model_channels = cfg.model_channels
        self.online = cfg.online
        self.seq_len = cfg.seq_len
        self.k = cfg.get("k", 1)
        

        self.init_params = {
            "in_channels": cfg.get("in_channels", self.window_size),
            "out_channels": cfg.get("out_channels", self.window_size),
            "cond_embed_dim": cfg.get("cond_embed_dim", self.cond_embed_dim),
            "model_channels": cfg.get("model_channels", self.model_channels),
            "dims": cfg.get("dims", 2),
            "use_scale_shift_norm": cfg.get("use_scale_shift_norm", False),
        }
        self.model = GDUnet_Latent(**self.init_params)

    def forward_offline(self, speaker_3dmm=None, speaker_emotion=None, listener_3dmm=None, listener_emotion=None, **kwargs):
        is_training = self.model.training
        batch_size = listener_3dmm.shape[0]
        k_active = self.training

        if is_training:
            window_start = torch.randint(0, self.seq_len-self.window_size+1, (1,), device=listener_3dmm.device)
            window_end = window_start + self.window_size
            # target to be predicted (and forward diffused)

            embed_s = self.encode(speaker_3dmm[:, window_start:window_end], speaker_emotion[:, window_start:window_end])
            embed_l = self.encode(listener_3dmm[:, window_start:window_end], listener_emotion[:, window_start:window_end])

            model_cond = self.cond_avg(embed_s) # [b, s, d] -> [b, d]

            embed_z = embed_l - embed_s  # Zd = Zl - Zs

            x_start = embed_z
            t, _ = self.schedule_sampler.sample(batch_size, listener_3dmm.device) 

            model_output, loss_ldm = self.diffusion.train_(self.model, x_start, t, model_kwargs={"cond": model_cond}, k_active=k_active)
            pred_embed_z = model_output
            model_output += embed_s.repeat_interleave(self.k, dim=0) if k_active else x_start  # Zl^ = Zd + Zs

            pred_3dmm, pred_emotion = self.decode(model_output)
            target_3dmm, target_emotion = self.decode(embed_l)

            results = {                            
              "pred_emotion": pred_emotion,                     
              "target_emotion": target_emotion, 
              "pred_3dmm": pred_3dmm,
              "target_3dmm": target_3dmm,
              "loss_ldm": loss_ldm,
              "embed_z": embed_z,
              "pred_embed_z": pred_embed_z,
            }

            if k_active and self.k>1:
                results = {k: v.view(batch_size, self.k, *results[k].shape[1:]) for k, v in results.items()}
            return results

        else: # iterate over all windows
            diff_batch, seq_len = speaker_3dmm.shape[:2]

            embed_s = self.encode(speaker_3dmm, speaker_emotion)
            model_cond = self.cond_avg(embed_s)
            
            output = self.diffusion.test_(self, self.model, diff_batch, seq_len, model_kwargs={"cond": model_cond})
            output["z_prediction"] += embed_s.repeat_interleave(self.k, dim=0) if k_active else x_start

            pred_3dmm, pred_emotion = self.decode(output["z_prediction"])
            output["pred_3dmm"]=pred_3dmm
            output["target_3dmm"]=listener_3dmm
            output["pred_emotion"]=pred_emotion
            output["target_emotion"]=listener_emotion

            return output


    def forward_online(self, speaker_video=None, speaker_audio=None, listener_video=None, listener_audio=None, **kwargs):
        is_training = self.model.training
        batch_size = speaker_video.shape[0]

        if is_training:
            # same as offline, but speaker emotion must be shifted by the window size
            # in order to only use past information
            speaker_video_shifted = speaker_video[:, :-self.window_size]
            speaker_audio_shifted = speaker_audio[:, :-self.window_size]

            listener_video_shifted = listener_video[:, self.window_size:]
            listener_audio_shifted = listener_audio[:, self.window_size:]

            # for the same listener window to be predicted, the speaker emotion will correspond to the past
            return self.forward_offline(speaker_video_shifted, speaker_audio_shifted, listener_video_shifted, listener_audio_shifted, **kwargs)

        else:
            # shift speaker emotion by window size and fill with zeros on the left
            # TODO an alternative strategy might be filling it with the most common speaker emotion
            speaker_video_shifted = torch.cat([torch.zeros_like(speaker_video[:, :self.window_size]), speaker_video[:, :-self.window_size]], dim=1)
            speaker_audio_shifted = torch.cat([torch.zeros_like(speaker_audio[:, :self.window_size]), speaker_audio[:, :-self.window_size]], dim=1)

            return self.forward_offline(speaker_video_shifted, speaker_audio_shifted, listener_video, listener_audio,**kwargs)

    def forward(self, **kwargs):
        if self.online:
            return self.forward_online(**kwargs)
        else:
            return self.forward_offline(**kwargs)