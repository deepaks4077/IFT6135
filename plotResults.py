import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
EXP_PATH = os.path.join(MAIN_DIR, 'experiments/')

file_names = ['Best model_vgg_training_and_validation_loss',
              'VGG_dropout=0.5_l2_1e-3_run',
              'VGG_dropout=0.5_l2_2e-3_run',
              'VGG_dropout=0.5_l2_3e-4_run',
              'VGG_dropout=0.5_l2_5e-3_run',
              'VGG_dropout=0.5_l2_5e-4_run',
              'VGG_dropout=0.6_l2_1.5e-3_learning_rate_1e-3_run']


def getLogData(file_name):
    training_loss = []
    training_accuracy = []

    validation_loss = []
    validation_accuracy = []

    training_epochs_recorded = 0
    validation_epochs_recorded = 0
    with open(file_name) as log_file:
        lines = log_file.readlines()
        for line in lines:
            if line.startswith('Training Loss'):
                line_content = line.split(' ')
                curr_training_loss = float(line_content[2].strip(','))
                curr_training_accuracy = float(line_content[4].strip(','))
                training_loss.append(curr_training_loss)
                training_accuracy.append(curr_training_accuracy)
                training_epochs_recorded += 1

            if line.startswith('Validation Loss'):
                line_content = line.split(' ')
                curr_validation_loss = float(line_content[2].strip(','))
                curr_validation_accuracy = float(line_content[4].strip(','))
                validation_loss.append(curr_validation_loss)
                validation_accuracy.append(curr_validation_accuracy)
                validation_epochs_recorded += 1

    return training_loss, training_accuracy, validation_loss, validation_accuracy


if __name__ == '__main__':
    for file in file_names:
        t_l, t_a, v_l, v_a = getLogData(EXP_PATH + file + '.txt')

        fig = plt.figure(figsize=(8, 6))
        plt.plot(t_l, 'k-')
        plt.plot(t_a, 'k--')
        plt.plot(v_l, 'r-')
        plt.plot(v_a, 'r--')
        plt.title('Learning curves')
        plt.xlabel('Epochs')
        plt.legend(['Train set loss', 'Train set accuracy', 'Validation set loss', 'Validation set accuracy'])
        plt.savefig(EXP_PATH + file + '.png', dpi=fig.dpi)
