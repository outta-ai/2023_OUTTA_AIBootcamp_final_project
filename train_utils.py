import torch
import os
import sys
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from typing import Optional, Callable
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)


def gather_all(dicts):
    return sum(dicts.values())


def KL_divergence(mu, log_sigma):
    kldiv = -log_sigma - 0.5 + (torch.exp(2 * log_sigma) + mu ** 2) * 0.5
    kldiv = torch.mean(torch.sum(kldiv, dim=1))
    return kldiv


def normalize(tensor: torch.tensor) -> torch.tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True)


def denormalize_image(x: torch.tensor) -> torch.tensor:
    return x * 0.5 + 0.5

def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
    ])

def custom_reshape(img, mode='bicubic', ratio=0.99):   # more to be implemented here
    full_size = img.shape[-2]
    prob = torch.rand(())

    if full_size < 224:
        pad_1 = torch.randint(0, 224-full_size, ())
        pad_2 = torch.randint(0, 224-full_size, ())
        m = torch.nn.ConstantPad2d((pad_1, 224-full_size-pad_1, pad_2, 224-full_size-pad_2), 1.)
        reshaped_img = m(img)
    else:
        cut_size = torch.randint(int(ratio*full_size), full_size, ())
        left = torch.randint(0, full_size-cut_size, ())
        top = torch.randint(0, full_size-cut_size, ())
        cropped_img = img[:, :, top:top+cut_size, left:left+cut_size]
        reshaped_img = F.interpolate(cropped_img , (224, 224), mode=mode, align_corners=False)
    return  reshaped_img


def clip_preprocess():
    return T.Compose([
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def save_hyp(args, g_in_chans, g_out_chans, d_in_chans, d_out_chans):
    hyp = {}
    hyp['clip_embedding_dim'] = args.clip_embedding_dim
    hyp['projection_dim'] = args.projection_dim
    hyp['noise_dim'] = args.noise_dim
    hyp['num_stage'] = args.num_stage
    hyp['use_uncond_loss'] = args.use_uncond_loss
    hyp['use_contrastive_loss'] = args.use_contrastive_loss
    hyp['batch_size'] = args.batch_size
    hyp['num_epochs'] = args.num_epochs
    hyp['learning_rate'] = args.learning_rate
    hyp['g_in_chans'] = g_in_chans
    hyp['g_out_chans'] = g_out_chans
    hyp['d_in_chans'] = d_in_chans
    hyp['d_out_chans'] = d_out_chans
    torch.save(hyp, os.path.join(args.checkpoint_path, f"hyperparameter.pt"))

    
def print_hyp(hyp: dict):
    print()
    for k, v in hyp.items():
        print(f"{k}: {v}")
    print()

def save_model(args, Generator, Discriminators, epoch: int, num_stage: int):
    torch.save(Generator.state_dict(), os.path.join(args.checkpoint_path, f"epoch_{epoch}_Gen.pt"))
    for i in range(num_stage):
        torch.save(Discriminators[i].state_dict(), os.path.join(args.checkpoint_path, f"epoch_{epoch}_Dis_{i}.pt"))


def load_checkpoint(G, D_lst, checkpoint_path, epoch):
    try:
        G.load_state_dict(torch.load(os.path.join(checkpoint_path, f"epoch_{epoch}_Gen.pt"), map_location='cpu'))
        G.eval()
        for i, D in enumerate(D_lst):
            D.load_state_dict(torch.load(os.path.join(checkpoint_path, f"epoch_{epoch}_Dis_{i}.pt"), map_location='cpu'))
            D.eval()
    except RuntimeError:
        print('Cannot load checkpoint; check the hyperparameter setting (e.g. num of stages)')
        exit(0)
    except FileNotFoundError:
        print('Cannot load checkpoint; check the directory')
        exit(0)
        