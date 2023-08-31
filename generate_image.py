import warnings
warnings.filterwarnings(action="ignore")

import argparse
import click
import os
import torch
import torchvision
import clip
from fix_seed import seed_fix
from pathlib import Path
from network import Generator, Discriminator
from train_utils import *


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--load_epoch', type=int, required=True)
    parser.add_argument('--checkpoint_path', type=Path, required=True)
    parser.add_argument('--show_hyp', action='store_true')
    parser.add_argument('--clip_model', type=click.Choice(['B/32', 'L/14', 'B/16']), default='B/32')
    args = parser.parse_args()
    
    # seed_fix(40)
    hyp = torch.load(os.path.join(args.checkpoint_path, f"hyperparameter.pt"), map_location='cpu')
    if args.show_hyp:
        print_hyp(hyp)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model, _ = clip.load(f"ViT-{args.clip_model}", device=device)
    G = Generator(hyp['clip_embedding_dim'], hyp['projection_dim'], hyp['noise_dim'], hyp['g_in_chans'], hyp['g_out_chans'], hyp['num_stage'], device).to(device)
    D_lst = [
        Discriminator(hyp['projection_dim'], hyp['g_out_chans'], hyp['d_in_chans'], hyp['d_out_chans'], hyp['clip_embedding_dim'], curr_stage, device).to(device)
        for curr_stage in range(hyp['num_stage'])
    ]
    load_checkpoint(G, D_lst, args.checkpoint_path, args.load_epoch)


    prompt = clip.tokenize([args.prompt]).to(device)
    txt_feature = clip_model.encode_text(prompt)
    z = torch.randn(txt_feature.shape[0], hyp['noise_dim']).to(device)
    txt_feature = normalize(txt_feature.to(device)).type(torch.float32)

    fake_images, _, _ = G(txt_feature, z)
    fake_image_64 = denormalize_image(fake_images[-3].detach().cpu()) 
    fake_image_128 = denormalize_image(fake_images[-2].detach().cpu()) 
    fake_image_256 = denormalize_image(fake_images[-1].detach().cpu()) 
    # epoch_ret = torchvision.utils.make_grid(fake_image, padding=2, normalize=True)
    torchvision.utils.save_image(fake_image_64, "result_64.png")
    torchvision.utils.save_image(fake_image_128, "result_128.png")
    torchvision.utils.save_image(fake_image_256, "result_256.png")


if __name__ == "__main__":
    main()