#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from samplers import *

parser = argparse.ArgumentParser(description='TransE model')

parser.add_argument("--gpu", type=int, default=0,
                    help="Which GPU to use?")
parser.add_argument("--n_iter", type=int, default=100000,
                    help="No. if iterations to run the training loop for")
parser.add_argument("--batch_size", type=int, default=512,
                    help="Batch size of samples?")

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')


params = parser.parse_args()

params.device = None
if not params.disable_cuda and torch.cuda.is_available():
    params.device = torch.device('cuda:%d' % params.gpu)
else:
    params.device = torch.device('cpu')

print('\n'.join('%s: %s' % (k, str(v)) for k, v
                in sorted(dict(vars(params)).items())))

# plot p0 and p1
fig = plt.figure()

# empirical
xx = torch.randn(10000)


def f(x): return torch.tanh(x * 2 + 1) + x * 0.75


def d(x): return (1 - torch.tanh(x * 2 + 1)**2) * 2 + 0.75


plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5, 5)
# exact
xx = np.linspace(-5, 5, 1000)


def N(x): return np.exp(-x**2 / 2.) / ((2 * np.pi)**0.5)


plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1) * N(xx))
plt.plot(xx, N(xx))
plt.savefig('exact.png', dpi=fig.dpi)

# import the sampler ``samplers.distribution4''
# train a discriminator on distribution4 and standard gaussian
# estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######


class Model(nn.Module):
    def __init__(self, inp_dim, out_dim, n_hidden, n_layers):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()

        layer_sizes = [n_hidden] * n_layers + [out_dim]

        for layer in layer_sizes:
            self.layers.append(nn.Linear(inp_dim, layer))
            inp_dim = layer

        self.reset_params()

    def reset_params(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.zero_()

    def forward(self, inp):
        a = inp
        for layer in self.layers[:-1]:
            z = layer(a)
            a = F.relu(z)
        a = torch.sigmoid(self.layers[-1](a))
        return a


def js_objective(model, x_batch, y_batch):
    loss = torch.mean(torch.log(model(x_batch))) + torch.mean(torch.log(1 - model(y_batch)))
    return -loss


def train(model):

    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for i in range(params.n_iter):
        x_batch = torch.Tensor(next(iter(distribution4(params.batch_size)))).to(device=params.device)
        y_batch = torch.Tensor(next(iter(distribution3(params.batch_size)))).to(device=params.device)

        loss = js_objective(model, x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


model = Model(1, 1, 32, 2).to(device=params.device)

tic = time.time()
model.reset_params()
train(model)
toc = time.time()
print('Completed run in %fs' % (toc - tic))


# plotting things
# (1) plot the output of your trained discriminator
# (2) plot the estimated density contrasted with the true density


r = model(torch.Tensor(xx[:, np.newaxis]).to(device=params.device)).detach().cpu().numpy()  # evaluate xx using your discriminator; replace xx with the output
fig = plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(xx, r)
plt.title(r'$D(x)$')

estimate = r / (1 - r) * N(xx)[:, np.newaxis]  # np.ones_like(xx) * 0.2  # estimate the density of distribution4 (on xx) using the discriminator;
# replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1, 2, 2)
plt.plot(xx, estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1) * N(xx))
plt.legend(['Estimated', 'True'])
plt.title('Estimated vs True')

plt.savefig('estimate.png', dpi=fig.dpi)
