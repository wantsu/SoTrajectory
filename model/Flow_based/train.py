
from tqdm import tqdm
from configs.Params import parameters
import matplotlib.pyplot as plt
from utils.utils import relative_to_abs
from model.Loss import cal_fde, cal_ade
from torch.nn import functional as F
import logging
import sys
import torch
from torch import nn, optim
from model.Flow_based.Benchmark import Model
from utils.data import get_dset_path
from data.dataloader import data_loader


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
            pred = model(obs_traj_rel)
            pred = relative_to_abs(pred, obs_traj[-1])
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
    # train_path = get_dset_path(args.dataset_name, 'train')
    # val_path = get_dset_path(args.dataset_name, 'val')
    train_path = '/home/want/Project/SoTrajectory/toy/toydata/train'
    val_path = '/home/want/Project/SoTrajectory/toy/toydata/val'
    logger.info("Initializing train dataset")
    train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    val_loader = data_loader(args, val_path)
    # load parameters
    flag = 'LSTM'
    model = Model(flag=flag, length=8)
    model = model.to(args.device)
    model.train()
    logger.info('Here is the Model:')
    logger.info(model)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss = []
    iteration = 0
    pbar = tqdm(range(args.num_epochs))
    for i in pbar:
        for batch in train_loader:
            iteration += 1
            batch = [tensor for tensor in batch]
            (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            pred = model(obs_traj_rel)
            cost = F.mse_loss(pred, pred_traj_rel)
            print('loss: ', cost.item())
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            loss.append(cost.item())

            if iteration % args.checkpoint_every == 0:
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

    if flag == 'LSTM':
        torch.save(model.state_dict(), "toy_LSTM")
    else:
        torch.save(model.state_dict(), "toy_RNN")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parameters()
    args.device = device
    main(args)