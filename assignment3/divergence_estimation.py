import time
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
parser.add_argument("--n_iter", type=int, default=100000,
                    help="No. if iterations to run the training loop for")
parser.add_argument("--batch_size", type=int, default=512,
                    help="Batch size of samples?")
parser.add_argument("--lam", type=float, default=10,
                    help="Gradient penalty weight.")

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


def wd_objective(model, x_batch, y_batch, a):
    z = a * x_batch + (1 - a) * y_batch
    z.requires_grad = True

    out = model(z)

    inp_grad = torch.autograd.grad(out, z, grad_outputs=torch.ones(out.shape),
                                   retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)

    grad_penalty = torch.mean((torch.norm(inp_grad[0], dim=1) - 1)**2)

    loss = torch.mean(model(x_batch)) - torch.mean(model(y_batch)) - params.lam * grad_penalty

    return -loss


def train(model, phi):

    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for i in range(params.n_iter):
        x_batch = torch.Tensor(next(iter(distribution1(0, params.batch_size)))).to(device=params.device)
        y_batch = torch.Tensor(next(iter(distribution1(phi, params.batch_size)))).to(device=params.device)
        a = torch.Tensor(np.random.uniform(0, 1, (params.batch_size, 1))).to(device=params.device)

        loss = wd_objective(model, x_batch, y_batch, a)
        # loss = js_objective(model, x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def js(model, phi):

    x_batch = torch.Tensor(next(iter(distribution1(0, 1000)))).to(device=params.device)
    y_batch = torch.Tensor(next(iter(distribution1(phi, 1000)))).to(device=params.device)

    with torch.no_grad():
        js_estimate = torch.log(torch.Tensor([2])) + 0.5 * torch.mean(torch.log(model(x_batch))) + 0.5 * torch.mean(torch.log(1 - model(y_batch)))

    return js_estimate


def wd(model, phi):

    x_batch = torch.Tensor(next(iter(distribution1(0, 1000)))).to(device=params.device)
    y_batch = torch.Tensor(next(iter(distribution1(phi, 1000)))).to(device=params.device)

    with torch.no_grad():
        wd_estimate = torch.mean(model(x_batch)) - torch.mean(model(y_batch))

    return wd_estimate


def main():
    model = Model(2, 1, 32, 2).to(device=params.device)

    wd_estimate = []
    for phi in range(-10, 11, 1):
        tic = time.time()
        model.reset_params()
        train(model, phi / 10)
        toc = time.time()
        print('Completed run for phi = %f in %fs' % (phi / 10, toc - tic))
        wd_estimate.append(wd(model, phi / 10))

    fig = plt.figure(figsize=(15, 6))
    x = np.linspace(-1, 1, 21)
    plt.plot(x, wd_estimate, 'o-')
    plt.savefig('wd_estimate.png', dpi=fig.dpi)


if __name__ == '__main__':
    main()
