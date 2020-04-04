import torch
from model.Feature_decoder import Decoder
from model.Glow.Flow import Model
import torch.nn as nn
from model.Model_utils import Extractor

class Flow_based(torch.nn.Module):

    def __init__(self, encoder_h_dim=64, decoder_h_dim=128,
                 mlp_dim=1024, num_layers=1, dropout=0, in_channel=2, length=8, n_flow=8, n_block=1):

        super(Flow_based, self).__init__()
        # Encoder: an RNN encoder to extract data into hidden state
        self.encoder = nn.LSTM(
            in_channel, encoder_h_dim, num_layers, dropout=dropout
        )
        self.Flow = Model(int(in_channel/(2**n_block)), encoder_h_dim, n_flow, n_block, affine=True, conv_lu=True)
        # Decoder: produce prediction paths
        self.decoder = self.decoder = nn.LSTM(
            int(encoder_h_dim / (2**n_block)), decoder_h_dim, num_layers, dropout=dropout
        )
        self.mlp = nn.Linear(decoder_h_dim, length)

    def forward(self, obs_traj):
        output, state = self.encoder(obs_traj)  # extract features
        hidden_x = state[0]
        log_p_sum, logdet, z_outs = self.Flow(hidden_x.permute(1, 0, 2), reverse='False') # obtain latent of pred trajectory
        hidden_y = torch.cat(z_outs, dim=1)  # concat latents into single tensor
        output, state_tuple = self.decoder(hidden_y)  # decode latent into pred traj
        pred = self.mlp(output)
        return log_p_sum, logdet, pred
