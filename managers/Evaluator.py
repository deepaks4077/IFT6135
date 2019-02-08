import torch
import csv
import pdb


class Evaluator():
    def __init__(self, params, model, data_loader, idx_to_class):
        self.params = params
        self.model = model
        self.data_loader = data_loader
        self.idx_to_class = idx_to_class

    def get_log_data(self):
        acc = torch.empty(len(self.data_loader))
        for i, (inputs, labels, ids) in enumerate(self.data_loader):
            # pdb.set_trace()

            inputs = inputs.to(device=self.params.device)
            labels = labels.to(device=self.params.device)

            scores = self.model(inputs)
            pred = torch.argmax(scores, dim=-1)
            acc[i] = torch.mean((pred == labels).double())

        log_data = dict([
            ('acc', torch.mean(acc))])

        return log_data

    def save_predictions(self, model, outfile_name):

        model.eval()

        ids_and_predictions = dict()

        for inputs, labels, ids in self.data_loader:
            ids_list = [x.split('/')[-1].split('.')[0] for x in ids]
            inputs = inputs.to(device=self.params.device)
            labels = labels.to(device=self.params.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            preds_list = list(preds.cpu().numpy())

            for (idx, pred) in zip(ids_list, preds_list):
                idx = int(idx)
                ids_and_predictions[idx] = self.idx_to_class[pred]

        print(len(ids_and_predictions))

        with open(outfile_name + '.csv', 'w') as out_file:
            csv_writer = csv.writer(out_file)
            csv_writer.writerow(['id', 'label'])
            for idx in range(1, len(ids_and_predictions) + 1):
                csv_writer.writerow([idx, ids_and_predictions[idx]])
