import argparse
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
parser.add_argument("--n_iter", type=int, default=1000,
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
    loss = torch.log(torch.Tensor([2])).to(device=params.device) + 0.5 * torch.mean(torch.log(model(x_batch))) + 0.5 * torch.mean(torch.log(1 - model(y_batch)))
    return -loss


def train(model, phi):

    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for i in range(params.n_iter):
        x_batch = torch.Tensor(next(iter(distribution1(0, params.batch_size)))).to(device=params.device)
        y_batch = torch.Tensor(next(iter(distribution1(phi, params.batch_size)))).to(device=params.device)

        loss = js_objective(model, x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def js(model, phi):

    x_batch = torch.Tensor(next(iter(distribution1(0, 1000)))).to(device=params.device)
    y_batch = torch.Tensor(next(iter(distribution1(phi, 1000)))).to(device=params.device)

    with torch.no_grad():
        js_estimate = torch.log(torch.Tensor([2])) + 0.5 * torch.mean(torch.log(model(x_batch))) + 0.5 * torch.mean(torch.log(1 - model(y_batch)))

    return js_estimate


def main():
    model = Model(2, 1, 32, 2).to(device=params.device)

    js_estimate = []
    for phi in range(-10, 11, 1):
        print('Running for phi = %f' % (phi / 10))
        model.reset_params()
        train(model, phi / 10)
        js_estimate.append(js(model, phi / 10))

    fig = plt.figure(figsize=(15, 6))
    x = np.linspace(-1, 1, 21)
    plt.plot(x, js_estimate, 'o-')
    plt.savefig('js_estimate.png', dpi=fig.dpi)


if __name__ == '__main__':
    main()
