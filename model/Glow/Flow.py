import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi
import numpy as np
from scipy import linalg as la


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()
        # loc: b;  scale: s;
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    # zero mean and unit variance
    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, length = input.shape
        # initialization
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        # Log_det = h * w * sum(log|s|)
        log_abs = torch.log(torch.abs(self.scale))

        logdet = length * torch.sum(log_abs)

        if self.logdet:
            return self.scale * input + self.loc, logdet  # s * (x + b)

        else:
            return self.scale * input + self.loc

    def reverse(self, output):
        return (output - self.loc) / self.scale   # y/s - b

# regular 1x1 Conv
class InvConv1d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)  # A = QR with Q being an orthogonal matrix and R being an upper triangular matrix
        weight = q.unsqueeze(2).unsqueeze(3)  # add dim 2,3 into (C,C,1,1)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, length = input.shape

        out = F.conv1d(input, self.weight)  # y = Wx
        logdet = (  # h * w * log|det(W)|
            length * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):  # W^{-1}
        return F.conv1d(output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class InvConv1dLU(nn.Module):  # LU 1x1 Conv
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)  # QR decomposition
        w_p, w_l, w_u = la.lu(q.astype(np.float32))  # A = PL(U + diag(s)) decomposition
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)   # diag(w_u)
        u_mask = np.triu(np.ones_like(w_u), 1)  # delete tangular numbers
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, length = input.shape

        weight = self.calc_weight()

        out = F.conv1d(input, weight)  # y = Wx
        logdet = length * torch.sum(self.w_s)  # h * w * sum(log|s|)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv1d(output, weight.squeeze().inverse().unsqueeze(2))


# Zero Initialization: zero the last conv of NN()
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv1d(in_channel, out_channel, 3, padding=0)  #
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1))

    def forward(self, input):
        # pad last dim (1, 1) and 2nd (1, 1), filling the value of padding with 1
        out = F.pad(input, [1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out

class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv1d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv1d(filter_size, in_channel if self.affine else in_channel // 2),  # zero initialization
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)   # x_a, x_b = split(x)
        if self.affine:        # in_a
            log_s, t = self.net(in_b).chunk(2, 1)  # (logs, t) = NN(x_b)
            s = torch.sigmoid(log_s + 2)  # s = torch.exp(log_s)
            out_a = (in_a + t) * s   # out_a = s * in_a + t
            out_b = in_b
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_b)
            print(in_a.shape, net_out.shape,'\n')
            out_a = in_a + net_out
            out_b = in_b
            logdet = None

        return torch.cat([out_a, out_b], 1), logdet   # concat(y_a, y_b)

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)  # y_a, y_b = split(y)

        if self.affine:
            log_s, t = self.net(out_b).chunk(2, 1)  #(logs, t) = NN(y_b)
            s = F.sigmoid(log_s + 2)   # s = torch.exp(log_s)
            in_a = out_a / s - t    # in_a = (out_a - t) / s
            in_b = out_b
        else:
            net_out = self.net(out_b)
            in_a = out_a - net_out
            in_b = out_b
        return torch.cat([in_a, in_b], 1)


# step of flow
class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv1dLU(in_channel)

        else:
            self.invconv = InvConv1d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)
        #
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


# Reparameterization trick
def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps    # u + sigma * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()
        #  1. squeeze
        squeeze_dim = in_channel * 2
        #  2. step of flow
        self.flows = nn.ModuleList()  # Holds submodules in a list.
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split
        # 3. split
        if split:
            self.prior = ZeroConv1d(in_channel * 1, in_channel * 2)

        else:
            self.prior = ZeroConv1d(in_channel * 2, in_channel * 4)

    def forward(self, input):
        b_size, n_channel, length = input.shape
        squeezed = input.view(b_size, n_channel, length // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 2)
        out = squeezed.contiguous().view(b_size, n_channel * 2, length // 2)

        logdet = 0
        # n flow per block
        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)   # z_i
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z
        # Reverse
        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, length = input.shape

        unsqueezed = input.view(b_size, n_channel // 2, 2, length)
        unsqueezed = unsqueezed.permute(0, 1, 3, 2)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 2, length * 2
        )

        return unsqueezed


class Model(nn.Module):
    def __init__(self, in_channel, length, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.mlps = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            length //=2
            self.mlps.append(torch.nn.Linear(length, length))
            n_channel *= 1

        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))
        self.mlps.append(torch.nn.Linear(length // 2, length // 2))

    def forward(self, input, reverse=False):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []
        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det
            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):

            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input

    def decoder(self, z_outs):
        outs = []
        for i, mlp in enumerate(self.mlps):
            out = mlp(z_outs[i])
            outs.append(out)

        pred_traj = self.reverse(outs, reconstruct=True)
        return pred_traj
