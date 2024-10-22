import torch
from model.Feature_decoder import Decoder
from model.Feature_encoder import Encoder
from model.Model_utils import Extractor, make_mlp
from model.Glow.Flow import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Flow_based(torch.nn.Module):

    def __init__(self, encoder_h_dim=64, decoder_h_dim=128, embedding_dim=64, h_dim=128, mlp_dim=1024,
                 num_layers=1, dropout=0, length=8, in_channel=2, n_flow=8, n_block=1, bottleneck_dim=1024,
                 activation='relu', batch_norm=True, pooling=True, pooling_type='pool_net', pool_every_timestep=True,
                 neighborhood_size=2.0, grid_size=8, cell='LSTM'):
        super(Flow_based, self).__init__()
        self.seq_len = length
        self.encoder_h_dim = encoder_h_dim
        self.num_layers = num_layers
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.pooling = pooling
        self.z_shapes = self.calc_z_shapes(num_layers, encoder_h_dim, n_block)
        # Extractor: an RNN encoder to extract data into hidden state
        self.extractor = Extractor(pooling, embedding_dim, encoder_h_dim,
                                   mlp_dim, bottleneck_dim, activation, batch_norm, cell)
        self.Flow = Model(num_layers, encoder_h_dim, n_flow, n_block, affine=True, conv_lu=True)
        if pooling:
            mlp_decoder_context_dims = [
                (encoder_h_dim+bottleneck_dim), mlp_dim, encoder_h_dim
            ]
            self.mlp = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
        else:
            pass

        # Decoder: produce prediction paths
        self.decoder = Decoder(
            self.seq_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            pooling=pooling,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size,
            cell=cell
        )

    def forward(self, obs_traj, obs_traj_rel, pred_traj, pred_traj_rel, seq_start_end):
        n_bins = 2. ** 6
        hidden_x = self.extractor(obs_traj, obs_traj_rel, seq_start_end, pooling=self.pooling).unsqueeze(0)
        hidden_y = self.extractor(pred_traj, pred_traj_rel, seq_start_end, pooling=False).unsqueeze(0)
        hidden_x = hidden_x.permute(1, 0, 2)
        input = hidden_y.permute(1, 0, 2)
        log_p_sum, logdet, z_outs = self.Flow(input + torch.rand_like(input) / n_bins, reverse=False) # obtain latent of pred trajectory
        z_lists = self.sample(hidden_x.shape[0], self.z_shapes)  # sampling from normal gaussian
        recon_y = self.Flow(z_lists, reverse=True)
        hidden_x, hidden_y = torch.squeeze(hidden_x, 1), torch.squeeze(recon_y, 1)
        if self.pooling:
            hidden_x = self.mlp(hidden_x)
        c = torch.cat((hidden_y, hidden_x), dim=1)
        out, loss = self.Decoder(c, obs_traj, obs_traj_rel, pred_traj_rel, seq_start_end)              # decode latent into pred traj
        return log_p_sum, logdet, out, loss

    def inference(self, obs_traj, obs_traj_rel, seq_start_end):
        hidden_x = self.extractor(obs_traj, obs_traj_rel, seq_start_end, pooling=self.pooling).unsqueeze(0)
        hidden_x = hidden_x.permute(1, 0, 2)
        z_lists = self.sample(hidden_x.shape[0], self.z_shapes) # sampling from normal gaussian
        recon_y = self.Flow(z_lists, reverse=True)
        hidden_x = torch.squeeze(hidden_x, 1)
        hidden_y = torch.squeeze(recon_y, 1)
        if self.pooling:
            hidden_x = self.mlp(hidden_x)
        c = torch.cat((hidden_y, hidden_x), dim=1)
        pred_traj_rel = None
        out, _ = self.Decoder(c, obs_traj, obs_traj_rel, pred_traj_rel, seq_start_end)
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
            n_channel *= 1

            z_shapes.append((n_channel, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 2, input_size))

        return z_shapes

    def Decoder(self, input, obs_traj, obs_traj_rel, pred_traj_rel, seq_start_end):
        decoder_h = torch.unsqueeze(input, 0)
        batch = obs_traj.size(1)
        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).to(device)
        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]

        # Predict Trajectory
        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            pred_traj_rel,
            seq_start_end,
        )
        traj_pred, loss = decoder_out
        return traj_pred, loss