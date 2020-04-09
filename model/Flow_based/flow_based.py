import torch
from model.Feature_encoder import Encoder
from model.Glow.Flow import Model
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Flow_based(torch.nn.Module):

    def __init__(self, encoder_h_dim=64, decoder_h_dim=128, embedding_dim=64, h_dim=128,
                 num_layers=1, dropout=0, in_channel=2, length=8, n_flow=8, n_block=1):
        super(Flow_based, self).__init__()
        self.seq_len = length
        self.encoder_h_dim = encoder_h_dim
        self.num_layers = 1
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.z_shapes = self.calc_z_shapes(int(in_channel/(2**n_block)), encoder_h_dim, n_block)
        # Encoder: an RNN encoder to extract data into hidden state
        self.encoder = Encoder()
        self.Flow = Model(int(in_channel / (2**(n_block))), encoder_h_dim, n_flow, n_block, affine=True, conv_lu=True)

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, obs_traj, obs_traj_rel, pred_traj_rel):
        n_bins = 2. ** 5
        hidden_x = self.encoder(obs_traj_rel)  # extract features
        hidden_y = self.encoder(pred_traj_rel)
        hidden_x = hidden_x.permute(1, 0, 2)
        input = hidden_y.permute(1, 0, 2)
        log_p_sum, logdet, z_outs = self.Flow(input + torch.rand_like(input) / n_bins, reverse=False) # obtain latent of pred trajectory
        recon_y = self.Flow(z_outs, reverse=True)
        hidden_x = hidden_x.view(-1, self.encoder_h_dim)
        hidden_y = recon_y.view(-1, self.encoder_h_dim)
        c = torch.cat((hidden_x, hidden_y), dim=1)
        out = self.Decoder(c, obs_traj, obs_traj_rel)              # decode latent into pred traj
        return log_p_sum, logdet, out

    def inference(self, obs_traj, obs_traj_rel):
        hidden_x = self.encoder(obs_traj_rel)
        z_lists= self.sample(hidden_x.shape[1], self.z_shapes) # sampling from normal gaussian
        recon_y = self.Flow(z_lists, reverse=True)
        hidden_x = hidden_x.view(-1, self.encoder_h_dim)
        hidden_y = recon_y.view(-1, self.encoder_h_dim)
        c = torch.cat((hidden_x, hidden_y), dim=1)
        out = self.Decoder(c, obs_traj, obs_traj_rel)
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

    def Decoder(self, input, obs_traj, obs_traj_rel):
        decoder_h = torch.unsqueeze(input, 0)
        batch = obs_traj.size(1)
        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).to(device)
        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]

        pred_traj = []  # prediction trajectory
        batch = last_pos.size(0)
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj = torch.stack(pred_traj, dim=0)

        return pred_traj