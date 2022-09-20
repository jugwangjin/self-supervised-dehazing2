
import os
import torch
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

import sys
sys.path.append('..')

import argparse


import os
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from math import log10
import numpy as np
import random
import torch.nn as nn

import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision
import sys
sys.path.append('..')

import model.model as model
import trainer.train_step as train_step
import trainer.saver as saver
import dataset

import os
import shutil
from tqdm import tqdm
import numpy
import random

import importlib

import argparse
import json
torch.manual_seed(20202464)

def to_psnr(output, gt):
    mse = F.mse_loss(output, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].detach().permute(0, 2, 3, 1).cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].detach().permute(0, 2, 3, 1).cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list

@torch.no_grad()
def main (args):
    psnr_list = []
    ssim_list = []

    fns = os.listdir(args["gtdir"])
    import PIL
    from PIL import Image
    import torchvision
    tf = torchvision.transforms.ToTensor()

    for fn in fns:
        # print(fn)
        if not (fn.endswith('.jpg') or fn.endswith('.png')):
            continue
        gtimg = Image.open(os.path.join(args["gtdir"], fn)).convert("RGB")
        try:
            inpimg = Image.open(os.path.join(args["inpdir"], fn)).convert("RGB")
        except:
            try:
                if fn.endswith(".jpg"):
                    inpimg = Image.open(os.path.join(args["inpdir"], fn.replace(".jpg", ".png"))).convert("RGB")
                else:
                    inpimg = Image.open(os.path.join(args["inpdir"], fn.replace(".png", ".jpg"))).convert("RGB")
            except:
                continue
        gtimg = tf(gtimg)
        inpimg = tf(inpimg)
        # print(gtimg.shape, inpimg.shape)
        if gtimg.shape != inpimg.shape:
            # print(inpimg.shape[1:])
            gtimg = torch.nn.Upsample(size=(inpimg.size(1), inpimg.size(2)))(gtimg.unsqueeze(0))[0]
            # print(gtimg.shape, inpimg.shape)
        psnr_list.extend(to_psnr(gtimg.unsqueeze(0), inpimg.unsqueeze(0)))
        ssim_list.extend(to_ssim_skimage(gtimg.unsqueeze(0), inpimg.unsqueeze(0)))


        # print(f'{fn}, psnr {to_psnr(gtimg.unsqueeze(0), inpimg.unsqueeze(0))} \
                    # ssmi {to_ssim_skimage(gtimg.unsqueeze(0), inpimg.unsqueeze(0))}')

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list) 

    print(f"psnr: {avr_psnr}, ssim: {avr_ssim}")
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtdir', type=str, required=True)
    parser.add_argument('--inpdir', type=str, required=True)
    args = parser.parse_args()
    args = vars(args)

    print(args)

    main(args)
