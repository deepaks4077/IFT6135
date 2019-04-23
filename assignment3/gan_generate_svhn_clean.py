import os, sys
sys.path.append(os.getcwd())
import argparse

import numpy as np

from tensorboardX import SummaryWriter

from models.wgan_mod import *

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from torch.utils.data import dataset

import torch.nn.init as init
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../../data/svhn", help="path to the data directory")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--start_iter", type=int, default=0, help="starting iteration")
parser.add_argument("--end_iter", type=int, default=50000, help="number of iterations to train for")
parser.add_argument("--output_path", type=str, default="svhn_iwgan_dcgan/", help="output path where results will be stored")
parser.add_argument("--image_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--n_gener", type=int, default=1, help="number of training steps for generator per iter")
parser.add_argument("--lambda_gp", type=int, default=10, help="Gradient penalty lambda hyperparameter")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.output_path, exist_ok=True)

if len(opt.data_dir) == 0:
    raise Exception('Please specify path to data directory')


def generate_image(netG, noise=None):
    if noise is None:
        noise = gen_rand_noise()

    with torch.no_grad():
        noise_vector = noise 
    samples = netG(noise_vector)
    samples = samples.view(opt.batch_size, 3, opt.image_size, opt.image_size)
    samples = samples * 0.5 + 0.5
    return samples

def gen_rand_noise():
    noise = torch.randn(opt.batch_size, opt.latent_dim)
    noise = noise.to(device)
    noise = noise.view(opt.batch_size, opt.latent_dim, 1, 1)
    return noise

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
Tensor = torch.cuda.FloatTensor if cuda_available else torch.FloatTensor
fixed_noise = gen_rand_noise() 

Gener = torch.load(opt.output_path + "generator.pt")
Diss = torch.load(opt.output_path + "discriminator.pt")

Gener = Gener.to(device)
Diss = Diss.to(device)


def gen_samples(i):
    gen_images = generate_image(Gener, gen_rand_noise())
    for it in range(gen_images.size(0)):
        print("saving "+str(it)+"\n")
        torchvision.utils.save_image(gen_images[it], opt.output_path + "/gen_samples/gen_sample_{}_{}.png".format(i, it))

def latent_image_space_interpolation():
    alpha = np.arange(0,1.1,0.1)
    fixed_noise_1 = gen_rand_noise()
    fixed_noise_2 = gen_rand_noise()
    l_tensor_list = []

    for it in range(len(alpha)):
        z_new = alpha[it] * fixed_noise_1 + (1-alpha[it]) * fixed_noise_2
        g_img = generate_image(Gener, z_new)
        l_tensor_list.append(torch.squeeze(g_img, dim=0))

    stacked_l = torch.stack(l_tensor_list)

    torchvision.utils.save_image(stacked_l, opt.output_path + "/lsi_images_3/lsi_samples.png", nrow=11, padding=2)

    gen_img_fn1 = generate_image(Gener, fixed_noise_1)
    gen_img_fn2 = generate_image(Gener, fixed_noise_2)

    i_tensor_list = []
    for it in range(len(alpha)):
        mod_img = alpha[it] * gen_img_fn1 + (1-alpha[it]) * gen_img_fn2
        i_tensor_list.append(torch.squeeze(mod_img, dim=0))

    stacked_i = torch.stack(i_tensor_list)
    torchvision.utils.save_image(stacked_i, opt.output_path + "/isi_images_3/isi_samples.png", nrow=11, padding=2)


def study_disentanglement():
    pert = 4
    fixed_noise = gen_rand_noise()
    gen_img = generate_image(Gener, fixed_noise)
    torchvision.utils.save_image(gen_img, opt.output_path + "/disentanglement_images_3/dis_sample_orig.png")
    for it in range(opt.latent_dim):
        mod_noise = copy.deepcopy(fixed_noise)
        mod_noise[0][it] += pert
        #if it == 0 or it == 1:
            #print("mod noise is ", mod_noise)
        gen_img = generate_image(Gener, mod_noise)
        torchvision.utils.save_image(gen_img, opt.output_path + "/disentanglement_images_3/dis_sample_{}.png".format(it))

# n_runs = int(1000/opt.batch_size) + 1
# for i in range(n_runs):
#     gen_samples(i)


latent_image_space_interpolation()
#study_disentanglement()
