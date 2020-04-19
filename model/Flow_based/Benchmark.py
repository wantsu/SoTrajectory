import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(torch.nn.Module):

    def __init__(self, flag='LSTM', length=8, input_size=2, hidden_size=128,
                 num_layers=2, dropout=0):
        super(Model, self).__init__()
        self.model = nn.Module()
        if flag == 'LSTM':
            self.model = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        else:
            self.model = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout)

        self.mlp = nn.Linear(hidden_size, input_size)


    def forward(self, obs_traj_rel):
        output, _ = self.model(obs_traj_rel)
        lengh, batch, hidden_size = output.size()
        out = self.mlp(output.view(-1, lengh*batch, hidden_size))
        return out.view(lengh, batch, -1)
