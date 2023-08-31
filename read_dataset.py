from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import os
import zipfile
import json
import PIL.Image
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import random
from train_utils import normalize

def cos_sim(x1, x2):
    return torch.dot(x1, x2) / (torch.norm(x1) * torch.norm(x2))

class ZipDataset(Dataset):
    def __init__(self, data_path, num_stage):
        self.data_path = data_path
        self.img_list = {}
        self.clip_img = {}
        self.clip_txt = {}
        self.idx_to_file = {}
        self.num_stage = num_stage
        self.img_sizes = [64 * (2 ** i) for i in range(self.num_stage)]
        self.load_zip_file(self.data_path)


    def file_ext(self, name: Union[str, Path]) -> str:
        return str(name).split('.')[-1]
    
    def is_image_ext(self, fname: Union[str, Path]) -> bool:
        ext = self.file_ext(fname).lower()
        return f'.{ext}' in PIL.Image.EXTENSION # type: ignore
    
    def image_transform(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def load_zip_file(self, source):
        with zipfile.ZipFile(source, mode='r') as z:
            # Load preprocessed CLIP image and text embeddings
            
            input_images = [str(f) for f in sorted(z.namelist()) if self.is_image_ext(f)]
            for idx, fname in enumerate(input_images):
                self.idx_to_file[idx] = fname
                with z.open(fname, 'r') as file:
                    img = PIL.Image.open(file)
                    self.img_list[fname] = self.image_transform()(T.Resize((64, 64))(img))
            # print('finished img files')

            if 'dataset.json' in z.namelist():
                # make clip image embedding dict.
                with z.open('dataset.json', 'r') as file:
                    img_embedding = json.load(file)['clip_img_features']
                    for fname, embedding in img_embedding:
                        self.clip_img[fname] = torch.tensor(embedding)
                        self.clip_img[fname] = normalize(self.clip_img[fname])
                # print('finished img feature')
                
                # make clip text embedding dict.
                with z.open('dataset.json', 'r') as file:
                    txt_embedding = json.load(file)['clip_txt_features']
                    for fname, embedding in txt_embedding:
                        self.clip_txt[fname] = torch.tensor(embedding)
                        self.clip_txt[fname] = normalize(self.clip_txt[fname])
                # print('finished txt feature')
            
            
            '''
            print(input_images)
            ['00000/img00000000.png', '00000/img00000001.png', '00000/img00000002.png', 
            '00000/img00000003.png', '00000/img00000004.png', '00001/img00000005.png', 
            '00001/img00000006.png', '00001/img00000007.png', '00001/img00000008.png', '00001/img00000009.png'
            ...
            ]
            '''

    def __len__(self):
        return len(self.idx_to_file)

    def __getitem__(self, idx):
        img_file = self.idx_to_file[idx]
        # print(img_file, int(img_file.split('img')[1].split('.')[0]))
        imgs = [self.img_list[img_file], *(T.Resize((self.img_sizes[i], self.img_sizes[i]))(self.img_list[img_file]) for i in range(1, self.num_stage))]
        img_embedding = self.clip_img[img_file]
        txt_embedding = self.clip_txt[img_file]
        txt_idx = random.randint(0, len(txt_embedding) - 1)
        # txt_idx = 0
        # print(cos_sim(img_embedding, txt_embedding[txt_idx]))
        # print(img.shape) # size : (3, 64, 64)
        return imgs, img_embedding, txt_embedding[txt_idx]