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
parser.add_argument("--output_name", type=str, default="output",
                    help="A folder with this name would be created to dump saved models and log files")

parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size")

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

_, _, test_data_loader, _, idx_to_class = data_loader(train_to_valid_ratio=0.8,
                                                      root_dir=DATA_PATH,
                                                      batch_size=params.batch_size)

tester = Evaluator(params, model, test_data_loader, idx_to_class)

test_log = tester.get_log_data()
logging.info('Test performance:' + str(test_log))

tester.save_predictions(params.output_name)
