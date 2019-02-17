import torch
from torch import nn
import pdb


class Evaluator():
    def __init__(self, params, model, data_loader):
        self.params = params
        self.model = model
        self.data_loader = data_loader
        self.criterion = nn.CrossEntropyLoss()

    def get_log_data(self):
        acc = torch.empty(len(self.data_loader))
        loss = 0
        for i, batch in enumerate(self.data_loader):
            # pdb.set_trace()

            batch[0] = batch[0].to(device=self.params.device)
            batch[1] = batch[1].to(device=self.params.device)

            scores = self.model(batch)
            pred = torch.argmax(scores, dim=-1)
            acc[i] = torch.mean((pred == batch[1]).double())

            loss += self.criterion(scores, batch[1]).item()

        log_data = dict([
            ('acc', torch.mean(acc)),
            ('loss', (loss / len(self.data_loader)))])
        return log_data
