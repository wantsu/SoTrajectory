# this .py is used to test the performance of prediction of model.
# sample trajectory and give it to model then visualize the comparision
# between ground truth and prediction
import torch
import random
from utils.data import get_dset_path
from configs.Params import parameters
from data.dataloader import data_loader
from utils.utils import relative_to_abs
from utils.visualization import visualization
from model.Flow_based.flow_based import Flow_based
import matplotlib.pyplot as plt



def count(traj):
    left, middle, right = 0, 0, 0
    batch_size = traj.shape[1]
    for i in range(batch_size):
        temp = int(traj[:, i][:, 1][0] - traj[:, i][:, 1][-1])  # y1 - y16
        if temp == 0:
            middle += 1
        elif temp < 0:
            left += 1
        else:
            right += 1

    return left, middle, right


args = parameters()
PATH = '/home/want/Project/SoTrajectory/model/Flow_based/toy_flow_based_model'
model = Flow_based(in_channel=2, length=8, n_flow=8, n_block=1)
model.load_state_dict(torch.load(PATH))
model.eval()
# test_path = get_dset_path(args.dataset_name, 'test')
test_path = '/home/want/Project/SoTrajectory/toy/toydata/test'
test_loader = data_loader(args, test_path)
for batch in test_loader:
    batch = [tensor for tensor in batch]
    (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
     non_linear_ped, loss_mask, seq_start_end) = batch

    linear_ped = 1 - non_linear_ped
    loss_mask = loss_mask[:, args.obs_len:]
    traj_pred = model.inference(obs_traj, obs_traj_rel)
    traj_pred = relative_to_abs(traj_pred, obs_traj[-1])
    pred = torch.cat((obs_traj, traj_pred), dim=0)
    gt = torch.cat((obs_traj, pred_traj), dim=0)
    idx = random.randint(1, obs_traj.size()[1])
    plt.plot(gt[:, idx][:, 0], gt[:, idx][:, 1], c='g')
    plt.plot(pred[:, idx][:, 0].detach().numpy(), pred[:, idx][:, 1].detach().numpy(), c='r')
    plt.show()
    left, middle, right = count(pred)
    print('left: ', left, 'middle:', middle, 'right:', right)
 #   visualization(gt, pred, args.obs_len)