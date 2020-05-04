from tqdm import tqdm
from configs.Params import parameters
import matplotlib.pyplot as plt
from utils.utils import relative_to_abs
import numpy as np
from torch.nn import functional as F
from model.Loss import cal_fde, cal_ade, gan_d_loss, gan_g_loss
from math import log, pi
import logging
import sys
import torch
from torch import nn, optim
from utils.data import get_dset_path
from data.dataloader import data_loader
from model.Flow_based.Gflow import Flow_based, TrajectoryDiscriminator

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def calc_loss(log_p, logdet, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = 1 * 64 # channel * encoder_dim

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
            pred = model.inference(obs_traj, obs_traj_rel, seq_start_end)
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
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')
    # train_path = '/home/want/Project/SoTrajectory/toy/toydata/train'
    # val_path = '/home/want/Project/SoTrajectory/toy/toydata/val'
    logger.info("Initializing train dataset")
    train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    val_loader = data_loader(args, val_path)
    # load parameters
    model = Flow_based(length=8, n_flow=4, n_block=1, pooling=args.pooling, cell=args.cell)
    model = model.to(args.device)
    model.train()
    logger.info('Here is the flow based model:')
    logger.info(model)
    #  Discriminator Model
    discriminator = TrajectoryDiscriminator(obs_len=8, pred_len=8)

    discriminator = discriminator.to(device)
    discriminator.train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    # loss function
    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(model.parameters(), lr=0.0001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001)
    d_loss = []
    g_loss = []
    pbar = tqdm(range(args.num_epochs))
    iter = 0
    for i in pbar:
        d_steps_left = 2
        g_steps_left = 1
        for batch in train_loader:
            batch = [tensor for tensor in batch]
            (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            if i == 0:
                with torch.no_grad():
                    log_p, logdet, pred = model(obs_traj, obs_traj_rel, pred_traj, pred_traj_rel, seq_start_end)

                    continue

            else:
                if d_steps_left > 0:
                    losses_d = discriminator_step(args, batch, model,
                                                  discriminator, d_loss_fn,
                                                  optimizer_d)
                    d_steps_left -= 1
                    d_loss.append(losses_d)
                # Generate step
                elif g_steps_left > 0:
                    losses_g = generator_step(args, batch, model,
                                              discriminator, g_loss_fn,
                                              optimizer_g)
                    g_steps_left -= 1
                    g_loss.append(losses_g)

                # Skip the rest if we are not at the end of an iteration
                if d_steps_left > 0 or g_steps_left > 0:
                    continue

                    # save loss
                if iter % args.print_every == 0:
                    logger.info('iter = {} / {}'.format(iter + 1, args.num_iterations))
                    for k, v in sorted(losses_d.items()):
                        logger.info('  [D] {}: {:.3f}'.format(k, v))
                    for k, v in sorted(losses_g.items()):
                        logger.info('  [G] {}: {:.3f}'.format(k, v))

                if iter % args.checkpoint_every == 0:
                    # Check stats on the validation set
                    logger.info('\n Checking stats on val ...')
                    metrics_val = check_accuracy(args, val_loader, model, limit=True)

                    logger.info('Checking stats on train ...')
                    metrics_train = check_accuracy(args, train_loader, model, limit=True)

                    for k, v in sorted(metrics_train.items()):
                        logger.info('  [train] {}: {:.3f}'.format(k, v))

                    for k, v in sorted(metrics_val.items()):
                        logger.info('  [val] {}: {:.3f}'.format(k, v))

                iter += 1
                d_steps_left = 2
                g_steps_left = 1
                if iter >= args.num_iterations:
                    break

    return model

def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d):
    batch = [tensor for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    # produce prediction path
    generator_out = generator.inference(obs_traj, obs_traj_rel, seq_start_end)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(generator_out, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip)
    optimizer_d.step()

    return losses


def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g):
    n_bins = 2. ** 5
    batch = [tensor for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    log_p, logdet, generator_out = generator(obs_traj, obs_traj_rel, pred_traj_gt, pred_traj_gt_rel, seq_start_end)
    logdet = logdet.mean()
    cost, log_p, log_det = calc_loss(log_p, logdet, n_bins)
    print(
        'flow_Loss: ', cost.item(), 'logP:', log_p.item(), 'logdet:', log_det.item()
    )
    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss + cost
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item() - cost.item()

    optimizer_g.zero_grad()
    loss.backward()
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)

    optimizer_g.step()

    return losses


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parameters()
    args.device = device
    model = main(args)
    #torch.save(model.state_dict(), "toy_flow_based_model")