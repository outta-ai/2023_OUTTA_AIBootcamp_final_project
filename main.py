import warnings
warnings.filterwarnings(action="ignore")

import argparse
import click
import os
from fix_seed import seed_fix
from train import train
from pathlib import Path
from typing import Union

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='sample_train.zip', type=Path, help="path of directory containing training dataset")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--report_interval', type=int, default=100, help='Report interval')
    parser.add_argument('--noise_dim', type=int, default=100, help= 'Input noise dimension to Generator')
    parser.add_argument('--projection_dim', type=int, default=128, help= 'Noise projection dimension')
    parser.add_argument('--clip_embedding_dim', type=int, default=512, help= 'CLIP embedding vector dimension')
    parser.add_argument('--checkpoint_path', type=Path, default='model_exp1', help='Checkpoint path')
    parser.add_argument('--result_path', type=Path, default='images_exp1', help='Generated image path')
    parser.add_argument('--use_uncond_loss', action="store_true")
    parser.add_argument('--use_contrastive_loss', action="store_true")
    parser.add_argument('--num_stage', type=int, default=1)
    parser.add_argument('--resume_checkpoint_path', default=None)
    parser.add_argument('--resume_epoch', type=int, default=-1)
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    seed_fix(0)
    train(args)

if __name__ == "__main__":
    main()
