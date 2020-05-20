import torch
import pandas as pd
from utils.data  import get_dset_path
from configs.Params import parameters
from data.dataloader import data_loader
from utils.utils import relative_to_abs
from model.Flow_based.flow_based import Flow_based
from model.Flow_based.Benchmark import Model
import matplotlib.pyplot as plt

# obtain frame_id and ped_id
def frame_ped_id(path, X, Y):
    df = pd.read_table(path, sep='\t', names=['frame_id', 'ped_id', 'x', 'y'])
    newdf = df[(df.x == X) & (df.y == Y)]
    frame_id, ped_id = newdf.frame_id.values[0], newdf.ped_id.values[0]
    return str(frame_id), str(ped_id)

args = parameters()
num_traj = 3   # produce multiple trajs for single obs_traj by sampling
PATH = './eth'  # model path
file_path = './eth.txt'  # file that save predicted traj
data_path = '/home/want/Project/SoTrajectory/datasets/datasets/eth/test/biwi_eth.txt'
model = Flow_based(length=8, n_flow=4, n_block=1, pooling=args.pooling, cell=args.cell)
model.load_state_dict(torch.load(PATH))
model.eval()
test_path = get_dset_path(args.dataset_name, 'test')
test_loader = data_loader(args, test_path)
f = open(file_path, 'w')
for batch in test_loader:
    batch = [tensor for tensor in batch]
    (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
     non_linear_ped, loss_mask, seq_start_end) = batch

    linear_ped = 1 - non_linear_ped
    loss_mask = loss_mask[:, args.obs_len:]

    buffer = []
    for i in range(num_traj):
        traj_pred = model.inference(obs_traj, obs_traj_rel, seq_start_end)
        traj_pred = relative_to_abs(traj_pred, obs_traj[-1])
        pred = torch.cat((obs_traj, traj_pred), dim=0)
        gt = torch.cat((obs_traj, pred_traj), dim=0)
        idx = 1   # obtain traj of the idx th man
        gt_x, gt_y = gt[:, idx][:, 0].tolist(), gt[:, idx][:, 1].tolist()
        pred_x, pred_y = pred[:, idx][:, 0].detach().numpy().tolist(),  pred[:, idx][:, 1].detach().numpy().tolist()
        plt.plot(gt_x, gt_y, c='g')
        plt.plot(pred_x, pred_y, c='r')

        base_frame_id, base_ped_id = frame_ped_id(data_path, round(gt_x[-1], 2), round(gt_y[-1], 2))
        for idx, _ in enumerate(gt_x):
            frame_id, ped_id = frame_ped_id(data_path, round(gt_x[idx], 2), round(gt_y[idx], 2))
            if ped_id != base_ped_id:
                ped_id = base_ped_id
                frame_id = int(base_frame_id) + idx*10
            # load into buffer
            buffer += [frame_id + '\t' + ped_id + '\t' + str(round(gt_x[idx], 2)) + '\t' + str(round(gt_y[idx], 2)) \
                      + '\t' + str(round(pred_x[idx], 2)) + '\t' + str(round(pred_y[idx], 2))]
    plt.show()
    save = input("save this plot? yes or no:\n");
    if save in ['yes', 'y']:

        for line in buffer:
            f.writelines(line+'\n')
    else:
        print("Dicard this plot!\n")

f.close()
