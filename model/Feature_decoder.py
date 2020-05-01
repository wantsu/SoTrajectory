import torch.nn as nn
from model.Model_utils import PoolHiddenNet, SocialPooling, make_mlp
import torch


class Decoder(nn.Module):
    """Decoder is the TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8, cell='LSTM'
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep
        self.cell = cell
        if self.cell == 'LSTM':
            self.decoder = nn.LSTM(
                embedding_dim, h_dim, num_layers, dropout=dropout
            )
        else:
            self.decoder = nn.GRU(
                embedding_dim, h_dim, num_layers, dropout=dropout
            )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        pred_traj = []  # prediction trajectory
        batch = last_pos.size(0)
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)
        if self.cell == 'GRU':
            state_tuple = state_tuple[0]
        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            # if self.pool_every_timestep:
            #     if self.cell == 'GRU':
            #         decoder_h = state_tuple
            #     else:
            #         decoder_h = state_tuple[0]
            #     pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
            #     decoder_h = torch.cat(
            #         [decoder_h.view(-1, self.h_dim), pool_h], dim=1)
            #     decoder_h = self.mlp(decoder_h)
            #     decoder_h = torch.unsqueeze(decoder_h, 0)
            #     if self.cell == 'GRU':
            #         state_tuple = (decoder_h, 0)
            #     else:
            #         state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj = torch.stack(pred_traj, dim=0)
        return pred_traj, state_tuple[0]