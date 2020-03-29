from torch.utils.data import DataLoader
from data.datasets import TrajectoryDataset
from utils.data import seq_collate

def data_loader(args, data_path):

    datasets = TrajectoryDataset(data_path,
                                 obs_len=args.obs_len,
                                 pred_len=args.pred_len,
                                 skip=args.skip,
                                 delim=args.delim)

    loader = DataLoader(
        dataset=datasets,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=seq_collate)

    return loader