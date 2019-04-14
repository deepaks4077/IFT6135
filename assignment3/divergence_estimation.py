import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from samplers import *

N_ITER = 100000
BATCH_SIZE = 512


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
    loss = torch.log(torch.Tensor([2])) + 0.5 * torch.mean(torch.log(model(x_batch))) + 0.5 * torch.mean(torch.log(1 - model(y_batch)))
    return -loss


def train(model, phi):

    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for i in range(N_ITER):
        x_batch = torch.Tensor(next(iter(distribution1(0, BATCH_SIZE))))
        y_batch = torch.Tensor(next(iter(distribution1(phi, BATCH_SIZE))))

        loss = js_objective(model, x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def js(model, phi):

    x_batch = torch.Tensor(next(iter(distribution1(0, 10000))))
    y_batch = torch.Tensor(next(iter(distribution1(phi, 10000))))

    with torch.no_grad():
        js_estimate = torch.log(torch.Tensor([2])) + 0.5 * torch.mean(torch.log(model(x_batch))) + 0.5 * torch.mean(torch.log(1 - model(y_batch)))

    return js_estimate


def main():
    model = Model(2, 1, 32, 2)

    js_estimate = []
    for phi in range(-10, 11, 1):
        model.reset_params()
        train(model, phi / 10)
        js_estimate.append(js(model, phi / 10))

    fig = plt.figure(figsize=(15, 6))
    x = np.linspace(-1, 1, 21)
    plt.plot(x, js_estimate, 'o-')
    plt.savefig('js_estimate.png', dpi=fig.dpi)


if __name__ == '__main__':
    main()
