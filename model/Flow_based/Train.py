from tqdm import tqdm
from configs.Params import parameters
import matplotlib.pyplot as plt
from utils.utils import relative_to_abs
from torch.nn import functional as F
from model.Loss import cal_fde, cal_ade
from math import log
import logging
import sys
import torch
from torch import nn, optim
from utils.data import get_dset_path
from data.dataloader import data_loader
from model.Flow_based.flow_based import Flow_based

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


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
            input = obs_traj_rel
            pred = model.inference(input)
            pred = relative_to_abs(pred.permute(2, 0, 1), obs_traj[-1])
            ade, ade_l, ade_nl = cal_ade(pred_traj_gt, pred, linear_ped, non_linear_ped)
            fde, fde_l, fde_nl = cal_fde(pred_traj_gt, pred, linear_ped, non_linear_ped)

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


def main(args):
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')

    logger.info("Initializing train dataset")
    train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    val_loader = data_loader(args, val_path)
    # load parameters
    model = Flow_based(in_channel=2, length=8, n_flow=8, n_block=1)
    model = model.to(args.device)
    model.train()
    logger.info('Here is the Model:')
    logger.info(model)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    n_bins = 2. ** 5
    loss = []
    iteration = 0
    pbar = tqdm(range(args.num_epochs))
    for i in pbar:
        for batch in train_loader:
            iteration += 1
            batch = [tensor for tensor in batch]
            (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, pred = model(obs_traj_rel, pred_traj_rel)

                    continue

            else:
                log_p, logdet, pred = model(obs_traj_rel, pred_traj_rel)

            logdet = logdet.mean()
            cost, log_p, log_det = calc_loss(log_p, logdet, n_bins)
            mseLoss = F.mse_loss(pred.permute(2, 0, 1), pred_traj_rel)
            print('flow_loss: ', cost.item())
            cost += mseLoss
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            loss.append(cost.item())
            pbar.set_description(
                f'Loss: {cost.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f};'
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
    #torch.save(model.state_dict(), "flowbased_model")