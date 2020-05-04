import torch
from model.Feature_decoder import Decoder
from model.Feature_encoder import Encoder
from model.Model_utils import Extractor, make_mlp, PoolHiddenNet
from model.Glow.Flow import Model
import torch.nn as nn

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
        self.z_shapes = self.calc_z_shapes(1, encoder_h_dim, n_block)
        # Extractor: an RNN encoder to extract data into hidden state
        self.extractor = Extractor(pooling, embedding_dim, encoder_h_dim,
                                   mlp_dim, bottleneck_dim, activation, batch_norm, cell)
        if pooling:
            self.z_shapes = self.calc_z_shapes(1, encoder_h_dim+bottleneck_dim, n_block)
            self.Flow = Model(1, encoder_h_dim+bottleneck_dim, n_flow, n_block, affine=True, conv_lu=True)
            mlp_decoder_context_dims = [
                (encoder_h_dim+bottleneck_dim)*2, mlp_dim, decoder_h_dim
            ]
            self.mlp = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
        else:
            self.Flow = Model(1, encoder_h_dim, n_flow, n_block, affine=True, conv_lu=True)

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
            grid_size=grid_size,
            neighborhood_size=neighborhood_size,
            cell=cell
        )

    def forward(self, obs_traj, obs_traj_rel, pred_traj, pred_traj_rel, seq_start_end):
        n_bins = 2. ** 5
        hidden_x = self.extractor(obs_traj, obs_traj_rel, seq_start_end).unsqueeze(0)
        hidden_y = self.extractor(pred_traj, pred_traj_rel, seq_start_end).unsqueeze(0)
        hidden_x = hidden_x.permute(1, 0, 2)
        input = hidden_y.permute(1, 0, 2)
        log_p_sum, logdet, z_outs = self.Flow(input + torch.rand_like(input) / n_bins, reverse=False) # obtain latent of pred trajectory
        recon_y = self.Flow(z_outs, reverse=True)
        hidden_x, hidden_y = torch.squeeze(hidden_x, 1), torch.squeeze(recon_y, 1)
        c = torch.cat((hidden_x, hidden_y), dim=1)
        out = self.Decoder(c, obs_traj, obs_traj_rel, seq_start_end)              # decode latent into pred traj
        return log_p_sum, logdet, out

    def inference(self, obs_traj, obs_traj_rel, seq_start_end):
        hidden_x = self.extractor(obs_traj, obs_traj_rel, seq_start_end).unsqueeze(0)
        hidden_x = hidden_x.permute(1, 0, 2)
        z_lists= self.sample(hidden_x.shape[0], self.z_shapes) # sampling from normal gaussian
        recon_y = self.Flow(z_lists, reverse=True)
        hidden_x, hidden_y = torch.squeeze(hidden_x, 1), torch.squeeze(recon_y, 1)
        c = torch.cat((hidden_x, hidden_y), dim=1)
        out = self.Decoder(c, obs_traj, obs_traj_rel, seq_start_end)
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

    def Decoder(self, input, obs_traj, obs_traj_rel, seq_start_end):
        if self.pooling:
            input = self.mlp(input)

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
            seq_start_end,
        )
        traj_pred, final_decoder_h = decoder_out
        return traj_pred



class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len=8, pred_len=8, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores
