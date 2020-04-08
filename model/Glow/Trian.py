from tqdm import tqdm
from math import log
import torch.nn.functional as F
from configs.Params import parameters, Flow_params
from model.Loss import cal_fde, cal_ade, bce_loss
from utils.utils import relative_to_abs
import matplotlib.pyplot as plt
import logging
import sys
import torch
from torch import nn, optim
from utils.data import get_dset_path
from data.dataloader import data_loader
from model.Glow.Flow import Model

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def check_accuracy(args, loader, model, limit=False):

    metrics = {}
    disp_error, disp_error_l, disp_error_nl = [], [], []
    f_disp_error, f_disp_error_l, f_disp_error_nl = [], [], []
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]
            input = obs_traj_rel.permute(1, 2, 0)
            _, _, z_outs = model.forward(input)
            pred_traj_rel = model.decoder(z_outs).permute(2, 0, 1)
            pred_traj = relative_to_abs(pred_traj_rel, obs_traj[-1])
            ade, ade_l, ade_nl = cal_ade(pred_traj_gt, pred_traj, linear_ped, non_linear_ped)
            fde, fde_l, fde_nl = cal_fde(pred_traj_gt, pred_traj, linear_ped, non_linear_ped)

            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
                total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    model.train()
    return metrics


# def sample_data(batch_size, image_size):
#
#     dataset = datasets.MNIST('~/Tutorial/data', transform=transforms.ToTensor())
#     loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
#     loader = iter(loader)
#
#     while True:
#         try:
#             yield next(loader)
#
#         except StopIteration:
#             loader = DataLoader(
#                 dataset, shuffle=True, batch_size=batch_size
#             )
#             loader = iter(loader)
#             yield next(loader)


# def calc_z_shapes(n_channel, length, n_flow, n_block):
#     z_shapes = []
#
#     for i in range(n_block - 1):
#         length //= 2
#         n_channel *= 2
#
#         z_shapes.append((n_channel, length, length))
#
#     length //= 2
#     z_shapes.append((n_channel * 4, length, length))
#
#     return z_shapes


def calc_loss(log_p, logdet, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = 2 * 8 * 1 # channel * height * weight

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def main(args):
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')
    logger.info("Initializing train dataset")
    train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    val_loader = data_loader(args, val_path)
    # load model parameters
    flow_config = Flow_params()

    # Model
    model = Model(
        flow_config.in_channel, flow_config.feature_size, flow_config.n_flow, flow_config.n_block, affine=flow_config.affine, conv_lu=not flow_config.no_lu
    )
    model = model.to(args.device)
    model.train()
    logger.info('Here is the Flow:')
    logger.info(model)
    optimizer = optim.Adam(model.parameters(), lr=flow_config.lr)
    pbar = tqdm(range(args.num_epochs))
    n_bins = 2. ** flow_config.n_bits
    loss = []
    iteration = 0
    for i in pbar:
        for batch in train_loader:
            batch = [tensor for tensor in batch]
            (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            iteration += 1
            # Computing cost
            input = obs_traj_rel.permute(1, 2, 0)
            pred_traj_gt_rel = pred_traj_rel.permute(1, 2, 0)
            if i == 0:
                with torch.no_grad():
                    log_p, logdet, z_outs = model(input + torch.rand_like(input) / n_bins)

                    continue

            else:
                log_p, logdet, z_outs = model(input + torch.rand_like(input) / n_bins)

            logdet = logdet.mean()

            cost, log_p, log_det = calc_loss(log_p, logdet, n_bins)
            print('nll:', cost.item)
            # reconstruction loss
            pred_traj_rel = model.decoder(z_outs)
            mseLoss = F.mse_loss(pred_traj_rel, pred_traj_gt_rel, reduction='sum')
            #bceLoss = F.bce_loss(pred_traj_rel, pred_traj_gt_rel, reduction='sum')
            cost += mseLoss
            model.zero_grad()
            cost.backward()
            #warmup_lr = flow_config.lr * min(1, i * args.batch_size / (args.num_iterations * 10))
            warmup_lr = flow_config.lr
            optimizer.param_groups[0]['lr'] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f'Loss: {cost.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}'
            )
            loss.append(cost.item())
            if iteration % args.checkpoint_every:
                # Check stats on the validation set
                logger.info('\n Checking stats on val ...')
                metrics_val = check_accuracy(args, val_loader, model, limit=True)

                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(args, train_loader, model, limit=True)

                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))

        i += 1
        if i >= args.num_iterations:
            break

    plt.plot([i for i in range(len(loss))], loss)
    plt.show()

    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parameters()
    args.device = device
    model = main(args)
    torch.save(model.state_dict(), "flow_model")