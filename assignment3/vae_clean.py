import argparse
import os
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
from torch.utils.data import dataset

import matplotlib.pyplot as plt
import numpy as np

from torch.optim import lr_scheduler

import time

os.makedirs("svhnimages_vae", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, default=32, help="input image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=100, help="dimension of z")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
parser.add_argument("--optim_lr", type=float, default=1e-5, help="learning rate for Adam")


args = parser.parse_args()
print("\n", args)

img_shape = (args.channels, args.img_size, args.img_size)
print("\nImage shape: {}".format(img_shape))

use_cuda = True if torch.cuda.is_available() else False

print("\n" ,"*"*40, "\nFound cuda: {}\n".format(use_cuda), "*"*40, "\n")

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])


def get_data_loader(dataset_location, batch_size):
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

trainloader, validloader, testloader = get_data_loader("./data/svhn", batch_size=args.batch_size)
print("\nTrainloader shape: {}".format(len(trainloader.dataset)))
print("\nValid loader shape: {} ".format(len(validloader.dataset)))
print("\nVest loader shape: {} ".format(len(testloader.dataset)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\nDevice: {}'.format(device))


def init_weights(m):
    if type(m) == nn.Conv2d:
        # nn.init.kaiming_uniform_(m.weight)
        nn.init.normal_(m.weight, 0.0, 0.02)
        m.bias.data.fill_(0.01)
    #     m.bias.data.zero_()

    if type(m) == nn.Linear:
        # nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        nn.init.normal_(m.weight, 0.0, 0.02)

    if type(m) == nn.ConvTranspose2d:
        # nn.init.kaiming_uniform_(m.weight)
        nn.init.normal_(m.weight, 0.0, 0.02)
        m.bias.data.fill_(0.01)


class VAE_DCGAN_ab(nn.Module):
    def __init__(self):
        super(VAE_DCGAN_ab, self).__init__()

        self.conv_1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=2, stride=2, padding=0), # 3*32*32, 128*16*16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv_1.apply(init_weights)

        #Conv 2
        self.conv_2=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0), # 128*16*16, 256*8*8
            nn.BatchNorm2d(256),
            # nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv_2.apply(init_weights)

        #Conv 3
        self.conv_3=nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=0), # 256*8*8, 512*4*4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv_3.apply(init_weights)


        # mu, sigma
        self.fc_enc_mu = nn.Sequential(
            nn.Linear(512*4*4, 100)
        )
        self.fc_enc_mu.apply(init_weights)

        self.fc_enc_var = nn.Sequential(
            nn.Linear(512*4*4, 100)
        )

        self.fc_enc_var.apply(init_weights)

        # decoder

        self.dcgan_gen_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1))

        self.dcgan_gen_block.apply(init_weights)

        self.output = nn.Tanh()


    def reparameterize_trick(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + (std * eps)


    def encode(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 4*4*512)

        mu = self.fc_enc_mu(x)
        log_var = F.softplus(self.fc_enc_var(x)) + 1e-5
        z = self.reparameterize_trick(mu, log_var)

        return z, mu, log_var


    def decode(self, z):
        z = self.dcgan_gen_block(z.contiguous())
        z = self.output(z)
        return z.view(-1, 3, 32, 32)


    def forward(self, x, prior=None):
        if prior is None: # usual
            z, mu, log_var = self.encode(x)
            z = z.view(-1, 100, 1, 1)
        else: # for generation
            z = torch.randn(args.batch_size, args.latent_dim).to(device)
            z = z.view(-1, 100, 1, 1)
            mu, log_var = 0.0, 0.0

        x_tilde = self.decode(z)

        return x_tilde, mu, log_var


    def generate(self):
        z = torch.randn(args.batch_size, args.latent_dim).to(device)
        x_tilde = self.decode(z)
        return x_tilde



def criterion(inputs, input_tilde, mu, log_var):

    n = input_tilde.size(0)
    recons_loss_mse = F.mse_loss(input_tilde.view(-1, 3*32*32), inputs.view(-1, 3*32*32),
                                 reduction="sum")

    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = (recons_loss_mse + kld_loss) / n
    return recons_loss_mse / n, kld_loss / n, total_loss


def visualize(dataloader, vae_model, betas,
              file_path="svhnimages_trial/", n=32, batch=2):

    dataloader = iter(dataloader)
    for b in range(batch):
        inputs, _ = next(dataloader)
    inputs = inputs[:n, :, :, :].to(device)
    inputs_tilde, _, _ = vae_model(inputs)
    inputs = inputs.cpu()
    inputs_tilde = inputs_tilde.cpu()

    inputs_grid = make_grid(inputs)
    inputs_tilde_grid = make_grid(inputs_tilde)

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    ndarr_inputs = inputs_grid.detach().numpy().transpose((1, 2, 0))
    ndarr_inputs = ndarr_inputs * std + mean

    ndarr_inputs_tilde = inputs_tilde_grid.detach().numpy().transpose((1, 2, 0))
    ndarr_inputs_tilde = ndarr_inputs_tilde * std + mean

    save_path = file_path + "test_dcgan_bs_" + \
                str(args.batch_size) + "_lr_" + str(args.optim_lr) + \
                "_betas_" + str(betas[0]) + "_" + str(betas[1])

    save_path_tilde = file_path + "test_tilde_dcgan_bs_" + \
                      str(args.batch_size) + "_lr_" + str(args.optim_lr) + \
                      "_betas_" + str(betas[0]) + "_" + str(betas[1])

    plt.imsave(save_path, ndarr_inputs)

    plt.imsave(save_path_tilde, ndarr_inputs_tilde)


def train():
    vae_model = VAE_DCGAN_ab().to(device)
    print("\n****** MODEL ******\n")
    print(vae_model)
    print("\n")

    betas = (0.5, 0.999)

    optimizer = optim.Adam(vae_model.parameters(), lr=args.optim_lr, betas=(0.5, 0.999))

    scheduler_plateau = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=10,
                                                       verbose=True, threshold=0.0001, threshold_mode='rel',
                                                       cooldown=0, min_lr=1e-9, eps=1e-08)

    best_valid_loss = np.inf

    loss_train_mb_array = []
    loss_valid_mb_array = []

    rc_loss_train_mb_array = []
    rc_loss_valid_mb_array = []

    kld_loss_train_mb_array = []
    kld_loss_valid_mb_array = []

    print("\n", "*"*20, " BATCH SIZE = {}".format(args.batch_size), "*"*20, "\n")

    start_time = time.time()

    model_save_path = ""

    for epoch in range(args.n_epochs):
        print('\n', '-'*10, ' Epoch: {} '.format(epoch), '-'*10)

        train_dataset_size = 0
        running_train_loss = 0.0

        # training
        vae_model.train()

        for inputs, _ in trainloader:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            # forward prop
            inputs_tilde, mu_train, log_var_train = vae_model(inputs)
            rc_loss_train, kld_loss_train, loss_train = criterion(inputs, inputs_tilde,
                                                                  mu_train, log_var_train)

            loss_train_mb_array.append(loss_train)
            rc_loss_train_mb_array.append(rc_loss_train)
            kld_loss_train_mb_array.append(kld_loss_train)

            # backprop
            loss_train.backward()
            optimizer.step()

            # metrics
            running_train_loss += loss_train.item()

            train_dataset_size += len(inputs)


        epoch_train_loss = running_train_loss / train_dataset_size

        print('Epoch training Loss: {}'.format(epoch_train_loss))

        # validation
        vae_model.eval() # set eval mode

        running_valid_loss = 0.0
        running_valid_kld_loss = 0.0
        running_valid_recons_loss = 0.0
        valid_dataset_size = 0

        with torch.no_grad():
            for inputs, _ in validloader:
                inputs = inputs.to(device)

                inputs_tilde, mu_valid, log_var_valid = vae_model(inputs)
                rc_loss_valid, kld_loss_valid, loss_valid = criterion(inputs, inputs_tilde,
                                                                      mu_valid, log_var_valid)

                loss_valid_mb_array.append(loss_valid)
                rc_loss_valid_mb_array.append(rc_loss_valid)
                kld_loss_valid_mb_array.append(kld_loss_valid)

                running_valid_loss += loss_valid.item()
                running_valid_kld_loss += kld_loss_valid.item()
                running_valid_recons_loss += rc_loss_valid.item()

                valid_dataset_size += len(inputs)

            epoch_valid_loss = running_valid_loss / valid_dataset_size
            epoch_valid_kld_loss = running_valid_kld_loss / valid_dataset_size
            epoch_valid_recons_loss = running_valid_recons_loss / valid_dataset_size

            print('Epoch valid Loss: {}, kld loss: {}, recons_loss: {}'.format(
                epoch_valid_loss, epoch_valid_kld_loss, epoch_valid_recons_loss))


        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss

            model_save_path = "vae_saved_model_dcgan_bs_" \
                              + str(args.batch_size) + "_lr_" + str(args.optim_lr) + \
                              "_betas_" + str(betas[0]) + "_" + str(betas[1])

            torch.save(vae_model, model_save_path + ".pt")

            visualize(testloader, vae_model, betas)

            print("\nNew best val loss = {}".format(best_valid_loss))
            print("Saved visualization and new model at {}\n".format(model_save_path))


        scheduler_plateau.step(epoch_valid_loss)


    print("\n** TRAINING DONE **")

    time_elapsed = time.time() - start_time

    print('Training completed in {}minutes {}secs'.format(time_elapsed // 60, time_elapsed % 60))

    print("\n** DONE **")

    return model_save_path + ".pt"


if __name__ == "__main__":
    model_save_path = train()
    print("DONE!!!!")

