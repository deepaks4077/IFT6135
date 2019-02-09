import argparse
import logging
import time
import pdb

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
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--eval_every", type=int, default=5,
                    help="Interval of epochs to evaluate the model?")
parser.add_argument("--save_every", type=int, default=10,
                    help="Interval of epochs to save a checkpoint of the model?")

parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience")
parser.add_argument("--optimizer", type=str, default="SGD",
                    help="Which optimizer to use?")
parser.add_argument("--lr", type=float, default=0.01,
                    help="Learning rate of the optimizer")

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

train_data_loader, valid_data_loader, _, class_to_idx, idx_to_class = data_loader(train_to_valid_ratio=0.8,
                                                                                  root_dir=DATA_PATH,
                                                                                  batch_size=params.batch_size)

trainer = Trainer(params, model, train_data_loader)
validator = Evaluator(params, model, valid_data_loader, idx_to_class)

tb_logger = Logger(params.exp_dir)

for e in range(params.nEpochs):
    tic = time.time()
    for inputs, labels, ids in train_data_loader:
        loss = trainer.one_step(inputs, labels)
        # print(loss)
    toc = time.time()

    tb_logger.scalar_summary('loss', loss, e)

    logging.info('Epoch %d with loss: %f in %f'
                 % (e, loss, toc - tic))

    # pdb.set_trace()
    if (e + 1) % params.eval_every == 0:
        log_data = validator.get_log_data()
        logging.info('Performance:' + str(log_data))

        for tag, value in log_data.items():
            tb_logger.scalar_summary(tag, value, e + 1)

        to_continue = trainer.save_model(log_data)

        if not to_continue:
            break

    if (e + 1) % params.save_every == 0:
        torch.save(model, os.path.join(params.exp_dir, 'cnn_checkpoint.pth'))

# test_log = tester.get_log_data()
# logging.info('Test performance:' + str(test_log))
