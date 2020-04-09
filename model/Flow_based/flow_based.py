import torch
from model.Feature_decoder import Decoder
from model.Glow.Flow import Model
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Flow_based(torch.nn.Module):

    def __init__(self, encoder_h_dim=64, decoder_h_dim=128,
                 mlp_dim=1024, num_layers=1, dropout=0, in_channel=2, length=8, n_flow=8, n_block=1):
        super(Flow_based, self).__init__()
        self.z_shapes = self.calc_z_shapes(int(in_channel/(2**n_block)), encoder_h_dim, n_block)
        # Encoder: an RNN encoder to extract data into hidden state
        #self.F1 = nn.Linear(2, encoder_h_dim)
        self.encoder = nn.LSTM(
            in_channel, encoder_h_dim, num_layers, dropout=dropout
        )
        self.Flow = Model(int(in_channel / (2**(n_block))), encoder_h_dim, n_flow, n_block, affine=True, conv_lu=True)
        # Decoder: produce prediction paths
        self.decoder = nn.LSTM(
            int(encoder_h_dim / (2**(n_block-1))), decoder_h_dim, num_layers, dropout=dropout
        )
        self.F2 = nn.Linear(decoder_h_dim, length)

    def forward(self, obs_traj, pred_traj):
        n_bins = 2. ** 5
        hidden_x = self.Encoder(obs_traj)  # extract features
        hidden_y = self.Encoder(pred_traj)
        hidden_x = hidden_x.permute(1, 0, 2)
        input = hidden_y.permute(1, 0, 2)
        log_p_sum, logdet, z_outs = self.Flow(input + torch.rand_like(input) / n_bins, reverse=False) # obtain latent of pred trajectory
        recon_y = self.Flow(z_outs, reverse=True)
        c = torch.cat((hidden_x, recon_y), dim=1)
        out = self.Decoder(c)              # decode latent into pred traj
        return log_p_sum, logdet, out

    def inference(self, obs_traj):
        hidden_x = self.Encoder(obs_traj).permute(1, 0, 2)
        z_lists= self.sample(hidden_x.shape[0], self.z_shapes) # sampling from normal gaussian
        recon_y = self.Flow(z_lists, reverse='True')
        c = torch.cat((hidden_x, recon_y), dim=1)
        out = self.Decoder(c)
        return out

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

    def Encoder(self, input):
        #embedding = self.F1(input)
        output, state = self.encoder(input)
        hidden = state[0]
        return hidden

    def Decoder(self, input):
        output, state = self.decoder(input)
        output = self.F2(output)
        return output