import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.pretraining_dataset import PretrainingDataset
from models.cbramod import CBraMod
from pretrain_trainer import Trainer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--parallel', type=bool, default=False)

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--clip_value', type=float, default=1)
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR')

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--in_dim', type=int, default=200, help='patch size')
    parser.add_argument('--out_dim', type=int, default=200)
    parser.add_argument('--d_model', type=int, default=200)
    parser.add_argument('--dim_feedforward', type=int, default=800)
    parser.add_argument('--seq_len', type=int, default=30, help='number of patches')
    parser.add_argument('--n_channels', type=int, default=19)

    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--need_mask', type=bool, default=True)
    parser.add_argument('--mask_ratio', type=float, default=0.5)

    parser.add_argument('--dataset_dir', type=str, required=True, help='root dir of H5 files')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--window_stride', type=int, default=None)
    parser.add_argument('--max_subjects', type=int, default=None,
                        help='Maximum number of subject H5 files to use; default uses all')
    parser.add_argument('--cache_open_files', action='store_true')
    parser.add_argument('--max_open_files', type=int, default=4)

    parser.add_argument('--model_dir', type=str, default='model_dir')
    params = parser.parse_args()

    print(params)
    setup_seed(params.seed)

    pretrained_dataset = PretrainingDataset(
        dataset_dir=params.dataset_dir,
        n_channels=params.n_channels,
        patch_num=params.seq_len,
        patch_size=params.in_dim,
        window_stride=params.window_stride,
        recursive=params.recursive,
        max_subjects=params.max_subjects,
        cache_open_files=params.cache_open_files,
        max_open_files=params.max_open_files,
    )

    print(len(pretrained_dataset))

    data_loader = DataLoader(
        pretrained_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(params.num_workers > 0),
    )

    model = CBraMod(
        params.in_dim,
        params.out_dim,
        params.d_model,
        params.dim_feedforward,
        params.seq_len,
        params.n_layer,
        params.nhead,
    )

    trainer = Trainer(params, data_loader, model)
    trainer.train()

    if hasattr(pretrained_dataset, "close"):
        pretrained_dataset.close()


if __name__ == '__main__':
    main()