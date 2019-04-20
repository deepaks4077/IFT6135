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

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../../data/svhn", help="path to the data directory")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
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

def get_data_loader(dataset_location, batch_size):  
    image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
    ])
    trainvalid = datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=image_transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
            datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=image_transform
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = Tensor(np.random.random((real_data.size(0), 1, 1, 1)))
    alpha = alpha.to(device)
    fake_data = fake_data.view(opt.batch_size, 3, opt.image_size, opt.image_size)
    fake_data = fake_data[:real_data.size(0), :, :, :]
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda_gp
    return gradient_penalty

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


Gener = Generator(opt.channels)
Diss = Discriminator(opt.channels)

LR = 1e-4
optimizer_g = torch.optim.Adam(Gener.parameters(), lr=LR, betas=(0.5,0.999))
optimizer_d = torch.optim.Adam(Diss.parameters(), lr=LR, betas=(0.5,0.999))
one = torch.FloatTensor([1])
mone = one * -1
Gener = Gener.to(device)
Diss = Diss.to(device)
one = one.to(device)
mone = mone.to(device)

writer = SummaryWriter()
def train():
    train_loader, valid_loader, test_loader = get_data_loader("../../data/svhn", opt.batch_size)
    dataiter = iter(train_loader)
    for iteration in range(opt.start_iter, opt.end_iter):
        print("Iteration: " + str(iteration))
        #---------------------TRAIN G------------------------
        for p in Diss.parameters():
            p.requires_grad_(False)  # freeze Diss

        gen_cost = None
        for i in range(opt.n_gener):
            print("Generator iters: " + str(i))
            Gener.zero_grad()
            noise = gen_rand_noise()
            noise.requires_grad_(True)
            fake_data = Gener(noise)
            gen_cost = Diss(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost.backward(mone)
            gen_cost = -gen_cost
        
        optimizer_g.step()
        #---------------------TRAIN D------------------------
        for p in Diss.parameters():  # reset requires_grad
            p.requires_grad_(True)  
        for i in range(opt.n_critic):
            print("Critic iter: " + str(i))

            Diss.zero_grad()
            noise = gen_rand_noise()
            with torch.no_grad():
                noise_vector = noise  
            fake_data = Gener(noise_vector).detach()

            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(train_loader)
                batch = dataiter.next()
            batch = batch[0] 
            real_data = batch.to(device) 

            # train with real data
            disc_real = Diss(real_data)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = Diss(fake_data)
            disc_fake = disc_fake.mean()

            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(Diss, real_data, fake_data)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward(retain_graph=True)
            w_dist = disc_fake  - disc_real
            optimizer_d.step()
            #------------------VISUALIZATION----------
            if i == opt.n_critic-1:
                writer.add_scalar('data/disc_cost', disc_cost, iteration)
                writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)
                writer.add_scalar('data/wasserstein_distance', w_dist, iteration)

        #---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)

        if iteration % 200 == 199:
            gen_images = generate_image(Gener, fixed_noise)
            torchvision.utils.save_image(gen_images, opt.output_path + 'samples_{}.png'.format(iteration), nrow=8, padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, iteration)
    #----------------------Save model----------------------
            torch.save(Gener, opt.output_path + "generator.pt")
            torch.save(Diss, opt.output_path + "discriminator.pt")

train()


##cleaned up
