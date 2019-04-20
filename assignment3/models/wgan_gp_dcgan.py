##my implementation
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
DIM=32
OUTPUT_DIM=32*32*3

class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dcgan_gen_block = nn.Sequential(
            # Z vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))

        #self.tanh = nn.Tanh()
        self.output = nn.Tanh()

    def forward(self, x):
        #x = x.view(32, 100, 1, 1)
        output = self.dcgan_gen_block(x.contiguous())
        output = self.output(output)
        output = output.view(-1, OUTPUT_DIM)
        return output


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dcgan_dis_block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        output = x.contiguous()
        output = output.view(-1, 3, 32, 32)
        output = self.dcgan_dis_block(output)
        output = self.output(output)
        output = output.view(-1)
        return output