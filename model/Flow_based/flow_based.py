import torch
from model.Feature_decoder import Decoder
from model.Glow.Flow import Model
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Flow_based(torch.nn.Module):

    def __init__(self, encoder_h_dim=64, decoder_h_dim=128,
                 mlp_dim=1024, num_layers=1, dropout=0, in_channel=2, length=8, n_flow=8, n_block=1):

        super(Flow_based, self).__init__()
        self.z_shapes = self.calc_z_shapes(int(in_channel/2), encoder_h_dim, n_block)
        # Encoder: an RNN encoder to extract data into hidden state
        self.encoder = nn.LSTM(
            in_channel, encoder_h_dim, num_layers, dropout=dropout
        )
        self.Flow = Model(int(in_channel/(2**n_block)), encoder_h_dim, n_flow, n_block, affine=True, conv_lu=True)

        # Decoder: produce prediction paths
        self.decoder = self.decoder = nn.LSTM(
            int(encoder_h_dim / (2**(n_block-1))), length, num_layers, dropout=dropout
        )

    def forward(self, obs_traj, pred_traj):
        n_bins = 2. ** 5
        output, state = self.encoder(obs_traj)  # extract features
        hidden_x = state[0]
        output, state = self.encoder(pred_traj)
        hidden_y = state[0]
        input = hidden_y.permute(1, 0, 2)
        log_p_sum, logdet, z_outs = self.Flow(input + torch.rand_like(input) / n_bins, reverse='False') # obtain latent of pred trajectory
        z = torch.cat(z_outs, dim=1)  # concat latents into single tensor
        hidden_x = self.squeese(hidden_x.permute(1, 0, 2))
        c = torch.cat((hidden_x, z), dim=2)
        output, state_tuple = self.decoder(c)  # decode latent into pred traj
        return log_p_sum, logdet, output

    def inference(self, obs_traj):
        output, state = self.encoder(obs_traj)
        hidden_x = state[0]
        hidden_x = self.squeese(hidden_x.permute(1, 0, 2))
        z= self.sample(hidden_x.shape[0], self.z_shapes) # sampling from normal gaussian
        z = torch.cat(z, dim=1)
        c = torch.cat((hidden_x, z), dim=2)
        output, _ = self.decoder(c)
       # pred = self.mlp(output)
        return output

    def sample(self, batch, z_shapes):
        z_sample = []
        for z in z_shapes:
            z_new = torch.randn(batch, *z)
            z_sample.append(z_new.to(device))

        return z_sample

    def calc_z_shapes(self, n_channel, input_size, n_block):
        z_shapes = []

        for i in range(n_block - 1):
            input_size //= 2
            n_channel *= 2

            z_shapes.append((n_channel, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 2, input_size))

        return z_shapes

    def squeese(self, input):
        b_size, n_channel, length = input.shape
        squeezed = input.view(b_size, n_channel, length // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 2)
        out = squeezed.contiguous().view(b_size, n_channel * 2, length // 2)
        return out