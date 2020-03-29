import torch
import torch.nn.functional as F
from model.Model_utils import Extractor
from model.Feature_decoder import Decoder
from model.Loss import cvae_loss

class CVAE(torch.nn.Module):

    def __init__(self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64, decoder_h_dim=128,
                 mlp_dim=1024, num_layers=1, pooling_type='pool_net', pool_every_timestep=True,
                dropout=0, bottleneck_dim=1024, activation='relu', batch_norm=True, pooling=True,
                 neighborhood_size=2.0, grid_size=8
                 ):
        super(CVAE, self).__init__()
        self.num_layers = num_layers
        self.decoder_h_dim = decoder_h_dim

        # Extractor: an RNN encoder to extract data into hidden state
        self.extractor = Extractor(pooling,embedding_dim, encoder_h_dim,
                                   mlp_dim, bottleneck_dim, activation, batch_norm)

        # reparameterization trick
        if pooling:
            self.FC_1 = torch.nn.Linear((embedding_dim+bottleneck_dim)*2, mlp_dim)  # 1088*2 1024
            self.z_mean = torch.nn.Linear(mlp_dim, decoder_h_dim)   #1024, 64
            self.z_log_var = torch.nn.Linear(mlp_dim, decoder_h_dim)
            self.FC_2 = torch.nn.Linear(embedding_dim+bottleneck_dim+decoder_h_dim, mlp_dim)
            self.FC_4 = torch.nn.Linear(mlp_dim, decoder_h_dim)
        else:
            self.FC_1 = torch.nn.Linear(embedding_dim+mlp_dim, mlp_dim)
            self.z_mean = torch.nn.Linear(mlp_dim, decoder_h_dim)
            self.z_log_var = torch.nn.Linear(mlp_dim, decoder_h_dim)
            self.FC_2 = torch.nn.Linear(mlp_dim, decoder_h_dim)
            self.FC_4 = torch.nn.Linear(embedding_dim + bottleneck_dim + decoder_h_dim, decoder_h_dim)
        # inferece
        self.FC_3 = torch.nn.Linear(embedding_dim+bottleneck_dim, decoder_h_dim)

        # Decoder: produce prediction paths
        self.decoder = Decoder(
            pred_len,
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
            neighborhood_size=neighborhood_size
        )

    def reparameterize(self, z_mu, z_log_var):
        #eps = torch.randn(z_mu.size(0), z_mu.size(1)).to('cuda')
        eps = torch.randn(z_mu.size(0), z_mu.size(1))
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def cencoder(self, features):
        x = self.FC_1(features)
        x = F.leaky_relu(x, negative_slope=1e-4)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded

    def cdecoder(self, encoded, obs_traj, obs_traj_rel, seq_start_end):
        x = self.FC_2(encoded)  # 1024
        x = F.leaky_relu(x, negative_slope=1e-4)
        x = self.FC_4(x)  # 128
        decoded = torch.sigmoid(x)  #128

        decoder_h = torch.unsqueeze(decoded, 0)
        batch = obs_traj.size(1)
        # decoder_c = torch.zeros(
        #     self.num_layers, batch, self.decoder_h_dim
        # ).to('cuda')
        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        )
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

    def inference(self, obs_traj, obs_traj_rel, seq_start_end):
        hidden_x = self.extractor(obs_traj, obs_traj_rel, seq_start_end)
        hidden_x = self.FC_3(hidden_x)
        traj_pred = self.cdecoder(hidden_x, obs_traj, obs_traj_rel, seq_start_end)
        return traj_pred

    def forward(self, obs_traj, obs_traj_rel, pred_traj, pred_traj_rel, seq_start_end, user_noise=None):
        hidden_x = self.extractor(obs_traj, obs_traj_rel, seq_start_end)
        hidden_y = self.extractor(pred_traj, pred_traj_rel, seq_start_end)
        hidden = torch.cat((hidden_x, hidden_y), 1)     # concat(x, y)
        z_mean, z_log_var, encoded = self.cencoder(hidden)
        encoded = torch.cat((encoded, hidden_x), 1)  # concat(hidden, x)   1088+128
        traj_pred_rel = self.cdecoder(encoded, obs_traj, obs_traj_rel, seq_start_end)

        # compute loss: DKL + Recon_loss
        loss = cvae_loss(z_mean, z_log_var, pred_traj_rel, torch.sigmoid(traj_pred_rel))

        return loss