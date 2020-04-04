from configs.Params import parameters, Cvae_params
from utils.data import get_dset_path
from data.dataloader import data_loader
from model.Loss import cal_fde, cal_ade
import logging
import sys
from tqdm import tqdm
import torch
import torch.optim as optim
from utils.utils import check_accuracy
from model.VAE.VAE import CVAE

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def main(args):

    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')

    logger.info("Initializing train dataset")
    train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    val_loader = data_loader(args, val_path)

    # read model parameters
    cvae_config = Cvae_params()

    # Model
    model = CVAE(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=cvae_config.embedding_dim,
        mlp_dim=cvae_config.mlp_dim,
        num_layers=cvae_config.num_layers,
        dropout=cvae_config.dropout,
        batch_norm=cvae_config.batch_norm,
        pool_every_timestep=args.pool_every_timestep,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        pooling = args.pooling
    )
    model = model.to(args.device)
    model.train()
    logger.info('Here is the CVAE:')
    logger.info(model)
    optimizer = optim.Adam(model.parameters(), lr=cvae_config.learning_rate)
    loss = []
    bar = tqdm(range(args.num_iterations))
    for t in bar:
        for batch in train_loader:
            batch = [tensor for tensor in batch]
            (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            cost = model(obs_traj, obs_traj_rel, pred_traj,
                         pred_traj_rel, seq_start_end)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            loss.append(cost.item())
            logger.info('CVAE cost: {:.3f}'.format(cost))

            # Check stats on the validation set
            logger.info('Checking stats on val ...')
            metrics_val = check_accuracy(args, val_loader, model, limit=True)

            logger.info('Checking stats on train ...')
            metrics_train = check_accuracy(args, train_loader, model, limit=True)

            for k, v in sorted(metrics_train.items()):
                logger.info('  [train] {}: {:.3f}'.format(k, v))

            for k, v in sorted(metrics_val.items()):
                logger.info('  [val] {}: {:.3f}'.format(k, v))

            t += 1
            if t >= args.num_iterations:
                break

    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parameters()
    args.device = device
    model = main(args)
    torch.save(model.state_dict(), "vae_model")