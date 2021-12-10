from __future__ import print_function

import argparse,os,random,torch

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
"""
class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d()
        )
"""
def custom_loader(path):
    ret = torch.load(path)
    return ret

#if __name__ == "__main__":
def main():
    manualSeed = 999
    print("Random Seed: ",manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    global workers,batch_size,image_size,nc,nz,ngf,ndf,lr,ngpu,beta1
    dataroot = "data/celeba" #"data/celeba"
    workers= 2
    batch_size=128
    image_size=64
    nc = 3
    nz = 100

    ngf = 64
    ndf = 64
    num_epochs = 5
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1
    #trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5))])
    trans = transforms.Compose([transforms.Resize(image_size),transforms.CenterCrop(image_size),
                                transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),])
    """
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                               ]))"""
    dataset = dset.DatasetFolder(root=dataroot,loader=custom_loader,transform=trans,extensions=".jpg")

    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                             shuffle=True,num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu >0) else "cpu")

if __name__ == "__main__":
    