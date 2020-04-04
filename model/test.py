# this .py is used to test the performance of prediction of model.
# sample trajectory and give it to model then visualize the comparision
# between ground truth and prediction
import torch
from utils.data import get_dset_path
from configs.Params import parameters
from data.dataloader import data_loader
from utils.utils import relative_to_abs
from utils.visualization import visualization

args = parameters()
PATH = '/home/want/Project/SoTrajectory/model/Glow'
model = torch.load(PATH)
model.eval()
test_path = get_dset_path(args.dataset_name, 'test')
test_loader = data_loader(args, test_path)
for batch in test_loader:
    batch = [tensor for tensor in batch]
    (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
     non_linear_ped, loss_mask, seq_start_end) = batch

    linear_ped = 1 - non_linear_ped
    loss_mask = loss_mask[:, args.obs_len:]
    input = obs_traj_rel.permute(1, 2, 0)
    _, _, z_outs = model.forward(input)
    pred_traj_rel = model.decoder(z_outs).permute(2, 0, 1)
    traj_pred = relative_to_abs(pred_traj_rel, obs_traj[-1])
    visualization(pred_traj, traj_pred, args.obs_len)
