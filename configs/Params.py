import argparse
import os
from utils.utils import int_tuple

def parameters():

    parser = argparse.ArgumentParser()

    # Misc
    parser.add_argument('--device', default='gpu', type=str)
    parser.add_argument('--timing', default=0, type=int)
    parser.add_argument('--gpu_num', default="0", type=str)

    # Dataset options
    parser.add_argument('--dataset_name', default='zara1', type=str)
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--delim', default='\t')
    parser.add_argument('--loader_num_workers', default=4, type=int)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=8, type=int)
    parser.add_argument('--skip', default=1, type=int)

    # Optimization
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_iterations', default=20000, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)


    # Model
    parser.add_argument('--Gan_config', type=str, default='configs/Gan.config')

    # Pooling Options
    parser.add_argument('--pooling', default=True, type=bool)
    parser.add_argument('--pooling_type', default='pool_net')
    parser.add_argument('--pool_every_timestep', default=1, type=bool)

    # Pool Net Option
    parser.add_argument('--bottleneck_dim', default=1024, type=int)

    # Social Pooling Options
    parser.add_argument('--neighborhood_size', default=2.0, type=float)
    parser.add_argument('--grid_size', default=8, type=int)


    # Loss Options
    parser.add_argument('--l2_loss_weight', default=1, type=float)
    parser.add_argument('--best_k', default=0, type=int)

    # Output
    parser.add_argument('--output_dir', default=os.getcwd())
    parser.add_argument('--print_every', default=5, type=int)
    parser.add_argument('--checkpoint_every', default=25, type=int)
    parser.add_argument('--checkpoint_name', default='checkpoint')
    parser.add_argument('--checkpoint_start_from', default=None)
    parser.add_argument('--restore_from_checkpoint', default=1, type=int)
    parser.add_argument('--num_samples_check', default=5000, type=int)

    return parser.parse_args()

def Gan_params():

    Gan_config = argparse.ArgumentParser()
    # Model Options
    Gan_config.add_argument('--embedding_dim', default=64, type=int)
    Gan_config.add_argument('--num_layers', default=1, type=int)
    Gan_config.add_argument('--dropout', default=1, type=float)
    Gan_config.add_argument('--batch_norm', default=0, type=bool)
    Gan_config.add_argument('--mlp_dim', default=1024, type=int)

    # Generator Options
    Gan_config.add_argument('--encoder_h_dim_g', default=64, type=int)
    Gan_config.add_argument('--decoder_h_dim_g', default=128, type=int)
    Gan_config.add_argument('--noise_dim', default=(0,), type=int_tuple)
    Gan_config.add_argument('--noise_type', default='gaussian')
    Gan_config.add_argument('--noise_mix_type', default='ped')
    Gan_config.add_argument('--clipping_threshold_g', default=0, type=float)
    Gan_config.add_argument('--g_learning_rate', default=5e-4, type=float)
    Gan_config.add_argument('--g_steps', default=1, type=int)

    # Discriminator Options
    Gan_config.add_argumnum_iterationsent('--d_type', default='local', type=str)
    Gan_config.add_argument('--encoder_h_dim_d', default=64, type=int)
    Gan_config.add_argument('--d_learning_rate', default=5e-4, type=float)
    Gan_config.add_argument('--d_steps', default=2, type=int)
    Gan_config.add_argument('--clipping_threshold_d', default=0, type=float)

    Gan_config = Gan_config.parse_args()

    return Gan_config


def Cvae_params():

    cvae_config = argparse.ArgumentParser()

    # Model Options
    cvae_config.add_argument('--embedding_dim', default=64, type=int)
    cvae_config.add_argument('--num_layers', default=1, type=int)
    cvae_config.add_argument('--dropout', default=1, type=float)
    cvae_config.add_argument('--batch_norm', default=0, type=bool)
    cvae_config.add_argument('--mlp_dim', default=1024, type=int)
    cvae_config.add_argument('--learning_rate', default=0.001, type=float)

    cvae_config = cvae_config.parse_args()

    return cvae_config


def Flow_params():

    flow_config = argparse.ArgumentParser()

    #Model Options
    flow_config.add_argument(
        '--n_flow', default=32, type=int, help='number of flows in each block'
    )
    flow_config.add_argument('--n_block', default=1, type=int, help='number of blocks')
    flow_config.add_argument('--feature_size', default=8, type=int, help='image size')
    flow_config.add_argument(
        '--no_lu',
        action='store_true',
        help='use plain convolution instead of LU decomposed version',
    )
    flow_config.add_argument(
        '--affine', action='store_true', default=True, help='use affine coupling instead of additive'
    )
    flow_config.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    flow_config.add_argument('--n_bits', default=5, type=int, help='number of bits')
    flow_config.add_argument('--in_channel', default=2, type=int, help='image input channel')
    flow_config.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
    flow_config.add_argument('--n_sample', default=20, type=int, help='number of samples')

    flow_config = flow_config.parse_args()

    return flow_config