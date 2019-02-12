import torch
import torch.optim as optim
import logging
import os
from torch import nn
import pdb


class Trainer():
    def __init__(self, params, model, data_loader):
        self.params = params
        self.model = model
        self.data_loader = data_loader
        self.criterion = nn.CrossEntropyLoss()

        self.model_params = list(model.parameters())

        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), self.model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model_params, lr=params.lr, momentum=params.momentum)

        self.bad_count = 0
        self.best_acc = 0
        self.last_acc = 0

    def one_step(self, inputs, labels):
        inputs = inputs.to(device=self.params.device)
        labels = labels.to(device=self.params.device)

        self.model.train()
        logits = self.model(inputs)

        loss = self.criterion(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save_model(self, log_data):
        if log_data['acc'] > self.last_acc:
            self.bad_count = 0

            if log_data['acc'] > self.best_acc:
                torch.save(self.model, os.path.join(self.params.exp_dir, 'best_model.pth'))  # Does it overwrite or fuck with the existing file?
                logging.info('Better model found w.r.t accuracy. Saved it!')
                self.best_acc = log_data['acc']
        else:
            self.bad_count = self.bad_count + 1
            if self.bad_count > self.params.patience:
                logging.info('Darn it! I dont have any more patience to give this model.')
                return False
        self.last_acc = log_data['acc']
        return True
