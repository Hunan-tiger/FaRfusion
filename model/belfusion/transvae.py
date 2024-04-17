import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)



def lengths_to_mask(lengths, device):
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths) # 255
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask



# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask



class Behaviour_features_fusion(nn.Module):
    def __init__(self, coeff_3dmm = 58, emotion = 25, feature_dim=128, depth=2, device = 'cpu'):
        super(Behaviour_features_fusion, self).__init__()
        self.PE = PositionalEncoding(feature_dim)
        self.mlp_3dmm = nn.Sequential(
            nn.Linear(coeff_3dmm, feature_dim*2),
            nn.LayerNorm(feature_dim*2),
            nn.SiLU(),
            nn.Linear(feature_dim*2, feature_dim)
        )
        self.mlp_emotion = nn.Sequential(
            nn.Linear(emotion, feature_dim*2),
            nn.LayerNorm(feature_dim*2),
            nn.SiLU(),
            nn.Linear(feature_dim*2, feature_dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim,
                                                             nhead=4,
                                                             dim_feedforward=feature_dim * 2,
                                                             dropout=0., batch_first=True)
        self.fusion_ = nn.TransformerDecoder(decoder_layer, num_layers=depth)


    def forward(self, coeff_3dmm, emotion):
        coeff_3dmm = self.mlp_3dmm(coeff_3dmm)
        emotion = self.mlp_emotion(emotion)
        emotion = self.PE(emotion)
        coeff_3dmm = self.PE(coeff_3dmm)
        return self.fusion_(emotion, coeff_3dmm)



class Encoder(nn.Module): ## Transformer Encoder
    def __init__(self,
                 in_channels: int,
                 latent_dim: int = 128,
                 depth: int = 2,
                 **kwargs) -> None:
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.depth = depth

        self.linear = nn.Linear(in_channels, latent_dim)
        self.PE = PositionalEncoding(latent_dim)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=4,
                                                             dim_feedforward=latent_dim * 2,
                                                             dropout=0.1,batch_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=self.depth)


    def forward(self, input):
        x = self.linear(input)  # B, seq_len（256）, latent_dim（128）
        B, T, D = input.shape
        x = self.PE(x)

        lengths = [len(item) for item in input]  ## [T, T, ..] B个T
        mask = lengths_to_mask(lengths, input.device)   ## (B, seq_len),全为True
        z = self.seqTransEncoder(x, src_key_padding_mask=~mask) # B, seq_len, latent_dim

        return z



class Decoder(nn.Module):  ## Transformer
    def __init__(self, output_3dmm_dim = 58, output_emotion_dim = 25, feature_dim = 128, device = 'cpu', max_seq_len=751, n_head = 4, seq_len=256, depth=2, **kwargs):
        super(Decoder, self).__init__()

        self.feature_dim = feature_dim
        self.device = device
        self.seq_len = seq_len
        self.depth = depth

        self.PE = PositionalEncoding(feature_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim,
                                                             nhead=n_head,
                                                             dim_feedforward=feature_dim * 2,
                                                             dropout=0.1, batch_first=True)
        self.listener_reaction_decoder = nn.TransformerEncoder(decoder_layer, num_layers=self.depth)

        self.listener_reaction_3dmm_map_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2),
            nn.LayerNorm(feature_dim*2),
            nn.SiLU(),
            nn.Linear(feature_dim*2, output_3dmm_dim)
        )
        self.listener_reaction_emotion_map_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2),
            nn.LayerNorm(feature_dim*2),
            nn.SiLU(),
        )
        self.listener_reaction_emotion_map_layer_au = nn.Linear(feature_dim*2, output_emotion_dim-10)
        self.listener_reaction_emotion_map_layer_va = nn.Linear(feature_dim*2, output_emotion_dim-23)
        self.listener_reaction_emotion_map_layer_fer = nn.Linear(feature_dim*2, output_emotion_dim-17)


    def forward(self, z):
        B = z.shape[0]
        z = self.PE(z)

        listener_reaction = self.listener_reaction_decoder(z)
        listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)

        emotion_all = self.listener_reaction_emotion_map_layer(listener_reaction)
        listener_emotion_out = torch.cat((self.listener_reaction_emotion_map_layer_au(emotion_all), self.listener_reaction_emotion_map_layer_va(emotion_all), self.listener_reaction_emotion_map_layer_fer(emotion_all)), dim=2)

        return listener_3dmm_out, listener_emotion_out



class TransformerVAE(nn.Module):
    def __init__(self, cfg):
        super(TransformerVAE, self).__init__()

        self.feature_dim = cfg.feature_dim
        self.output_3dmm_dim = cfg.output_3dmm_dim
        self.output_emotion_dim = cfg.output_emotion_dim
        self.coeff_3dmm = cfg.coeff_3dmm
        self.emotion = cfg.emotion
        self.seq_len = cfg.seq_len
        self.depth = cfg.depth
        self.device = cfg.device

        self.Behaviour_features_fusion = Behaviour_features_fusion(coeff_3dmm=self.coeff_3dmm, emotion=self.emotion,feature_dim=self.feature_dim, depth=self.depth, device=self.device)

        self.reaction_decoder = Decoder(output_3dmm_dim=self.output_3dmm_dim, output_emotion_dim=self.output_emotion_dim, feature_dim=self.feature_dim, depth=self.depth, device=self.device)


    # ---------self training----------------
    def _encode(self, a, b):
        return self.Behaviour_features_fusion(a, b)
    
    def _decode(self, z):
        pred_3dmm, pred_emotion = self.reaction_decoder(z)
        return pred_3dmm, pred_emotion
    
    
    # -------------------- ldm -------------------
    def encode(self, a, b):
        return self._encode(a, b)
    
    def decode(self, z):
        y_3dmm, y_emotion = self._decode(z)
        return y_3dmm, y_emotion
    

    def forward(self, listener_3dmm=None, listener_emotion=None, **kwargs):

        features_fusion = self._encode(listener_3dmm, listener_emotion)

        pred_3dmm, pred_emotion = self._decode(features_fusion)
        target_3dmm, target_emotion = listener_3dmm, listener_emotion

        result = {'pred_emotion':pred_emotion, 'target_emotion':target_emotion, 'pred_3dmm':pred_3dmm, 'target_3dmm':target_3dmm}
        return result



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    l_video = torch.randn(2, 750, 64+25088+1408)
    l_audio = torch.randn(2, 750, 1536)
    l_emotion = torch.randn(2, 750, 25)
    l_3dmm = torch.randn(2, 750, 58)
    model = TransformerVAE()
    res = model(l_video, l_audio, l_emotion, l_3dmm)
    print(res['prediction'].shape, res['target'].shape, res['coefficients_3dmm'].shape, res['target_coefficients'].shape)
