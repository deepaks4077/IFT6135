import os
import argparse
import logging
import json
import torch
import tensorflow as tf
import numpy as np
import scipy.misc
from io import BytesIO

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from core import CNNModel1, CNNModel2, ResNet

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(MAIN_DIR, 'data')
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train2id.txt')
VALID_DATA_PATH = os.path.join(DATA_PATH, 'valid2id.txt')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test2id.txt')


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. use 0 or 1")


def initialize_experiment(params):

    exps_dir = os.path.join(MAIN_DIR, 'experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    params.exp_dir = os.path.join(exps_dir, params.experiment_name)

    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)

    file_handler = logging.FileHandler(os.path.join(params.exp_dir, "log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump(vars(params), fout)


def initialize_model(params):

    if os.path.exists(os.path.join(params.exp_dir, 'best_model.pth')):
        logging.info('Loading existing model from %s' % os.path.join(params.exp_dir, 'best_model.pth'))
        model = torch.load(os.path.join(params.exp_dir, 'best_model.pth'))
    else:
        logging.info('No existing model found. Initializing new model..')
        if params.model == 'resnet':
            model = ResNet(params)
        elif params.model == 'cnn1':
            model = CNNModel1(params)
        elif params.model == 'cnn2':
            model = CNNModel2(params)

    return model

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


class ImageFolderWithPaths(datasets.ImageFolder):
    # Extends torchvision.datasets.ImageFolder

    # override __getitem__
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_and_path = (original_tuple + (path,))
        return tuple_and_path

# data_loader


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print('Device: {}'.format(device))


def data_loader(train_to_valid_ratio=0.8,
                root_dir=DATA_PATH,
                batch_size=32):

    train_valid_data = ImageFolderWithPaths(
        os.path.join(root_dir, 'trainset'), data_transforms['train'])

    test_data = ImageFolderWithPaths(os.path.join(
        root_dir, 'testset'), data_transforms['test'])

    print('class_to_idx: {}'.format(train_valid_data.class_to_idx))
    class_to_idx = train_valid_data.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print('Size of train_valid_data[0]: {}\n'.format(len(train_valid_data[0][0].shape)))

    train_valid_data_size = len(train_valid_data)
    train_valid_indices = list(range(train_valid_data_size))
    split = int(np.ceil(train_to_valid_ratio * train_valid_data_size))
#     print('split = {}'.format(split))

#     print('Image size = {}, Label = {}\n'.format(train_valid_data[0][0].shape, train_valid_data[0][1]))

    # shuffle the indices
    np.random.shuffle(train_valid_indices)
    train_indices, valid_indices = train_valid_indices[:split], train_valid_indices[split:]

    print('len(train_indices): {}'.format(len(train_indices)))
    print('len(valid_indices): {}'.format(len(valid_indices)))

    print('Size of test data: {}'.format(len(test_data)))

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(train_valid_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_valid_data, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return (train_loader, valid_loader, test_loader,
            class_to_idx, idx_to_class)
