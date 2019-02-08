from torch import nn
import pdb


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.convolutional_layers = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=3, out_channels=18, kernel_size=(3, 3), padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # 2
            nn.Conv2d(in_channels=18, out_channels=36, kernel_size=(3, 3), padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # 3
            nn.Conv2d(in_channels=36, out_channels=72, kernel_size=(3, 3), padding=1),
            nn.ReLU(),

            # 4
            nn.Conv2d(in_channels=72, out_channels=72, kernel_size=(3, 3), padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # 5
            nn.Conv2d(in_channels=72, out_channels=144, kernel_size=(3, 3), padding=1),
            nn.ReLU(),

            # 6
            nn.Conv2d(in_channels=144, out_channels=144, kernel_size=(3, 3), padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        # FC layers
        self.fc_layers = nn.Sequential(
            nn.Linear(144 * 4 * 4, 144 * 4 * 4), nn.ReLU(),
            nn.Linear(144 * 4 * 4, 500), nn.ReLU(),
            nn.Linear(500, 2)
        )

    # forward pass
    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(-1, 144 * 4 * 4)
        x = self.fc_layers(x)
        return x
