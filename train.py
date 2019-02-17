import argparse
import logging
import time
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from core import *
from managers import *
from utils import *

import torch

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='CNN model for MNIST')

parser.add_argument("--experiment_name", type=str, default="default",
                    help="A folder with this name would be created to dump saved models and log files")

parser.add_argument("--nEpochs", type=int, default=10,
                    help="Learning rate of the optimizer")
parser.add_argument("--batch_size", type=int, default=256,
                    help="Batch size")
parser.add_argument("--eval_every", type=int, default=1,
                    help="Interval of epochs to evaluate the model?")
parser.add_argument("--save_every", type=int, default=50,
                    help="Interval of epochs to save a checkpoint of the model?")

parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience")
parser.add_argument("--optimizer", type=str, default="SGD",
                    help="Which optimizer to use?")
parser.add_argument("--lr", type=float, default=0.01,
                    help="Learning rate of the optimizer")
parser.add_argument("--momentum", type=float, default=0.9,
                    help="Momentum of the SGD optimizer")

parser.add_argument("--debug", type=bool_flag, default=False,
                    help="Run the code in debug mode?")

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')

params = parser.parse_args()

logging.basicConfig(level=logging.INFO)

initialize_experiment(params)

params.device = None

params.use_gpu = not params.disable_cuda and torch.cuda.is_available()

if params.use_gpu:
    params.device = torch.device('cuda')
else:
    params.device = torch.device('cpu')

logging.info(params.device)

model = initialize_model(params).to(device=params.device)

train_data_loader, valid_data_loader = get_train_valid_loader(DATA_PATH,
                                                              params.batch_size,
                                                              valid_size=0.1,
                                                              shuffle=True,
                                                              num_workers=4,
                                                              pin_memory=params.use_gpu)

test_data_loader = get_test_loader(DATA_PATH,
                                   params.batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=params.use_gpu)

trainer = Trainer(params, model, train_data_loader)
tr_validator = Evaluator(params, model, train_data_loader)
validator = Evaluator(params, model, valid_data_loader)
tester = Evaluator(params, model, test_data_loader)

tr_acc = []
val_acc = []
tr_losses = []
val_losses = []

for e in range(params.nEpochs):
    tic = time.time()
    for b, batch in enumerate(train_data_loader):
        loss = trainer.one_step(batch)
    toc = time.time()

    logging.info('Epoch %d with loss: %f in %f'
                 % (e, loss, toc - tic))

    # pdb.set_trace()
    if (e + 1) % params.eval_every == 0:
        tr_log_data = tr_validator.get_log_data()
        val_log_data = validator.get_log_data()
        logging.info('Train performance: %f, Validattion performance: %f' % (tr_log_data['acc'], val_log_data['acc']))
        tr_acc.append(tr_log_data['acc'])
        tr_losses.append(tr_log_data['loss'])
        val_acc.append(val_log_data['acc'])
        val_losses.append(val_log_data['loss'])
        # to_continue = trainer.save_model(val_log_data)

        # if not to_continue:
        #     break

    if (e + 1) % params.save_every == 0:
        torch.save(model, os.path.join(params.exp_dir, 'cnn_checkpoint.pth'))

test_log = tester.get_log_data()
logging.info('Test performance:' + str(test_log))

fig = plt.figure(figsize=(15, 6))
plt.plot(tr_acc, 'k-')
plt.plot(tr_losses, 'k--')
plt.plot(val_acc, 'r-')
plt.plot(val_losses, 'r--')
plt.title('Train and validation accuracies')
plt.legend(['Train set accuracy', 'Train set loss', 'Validation set accuracy', 'Validation set loss'])
plt.savefig(params.experiment_name + '.png', dpi=fig.dpi)
