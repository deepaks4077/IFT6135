from torch import nn
import pdb


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), padding=1),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 3), padding=1),
                                   nn.ReLU())
        self.downSample = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), padding=1),
                                        nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.relu = nn.ReLU()

    def forward(self, input):
        x = input
        l1 = self.conv1(input)
        l2 = self.conv2(l1)

        l2 += self.downSample(x)

        out = self.relu(l2)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.block1 = ResBlock(3, 18)
        self.block2 = ResBlock(18, 36)
        self.block3 = ResBlock(36, 72)
        self.block4 = ResBlock(72, 144)

        self.fc_layers = nn.Sequential(
            nn.Linear(144 * 4 * 4, 144 * 4 * 4), nn.ReLU(),
            nn.Linear(144 * 4 * 4, 500), nn.ReLU(),
            nn.Linear(500, 2)
        )

    def forward(self, input):
        l1 = self.block1(input)
        l2 = self.block2(l1)
        l3 = self.block3(l2)
        l4 = self.block4(l3)

        l4 = l4.view(-1, 144 * 4 * 4)

        logits = self.fc_layers(l4)
        return logits


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # 1
        # (3, 64, 64)
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=18, kernel_size=(3, 3), padding=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        # 2
        # (18, 32, 32)
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=18, out_channels=36, kernel_size=(3, 3), padding=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        # 3
        # (36, 16, 16)
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=36, out_channels=72, kernel_size=(3, 3), padding=1),
                                    nn.ReLU())

        # 4
        # (72, 16, 16)
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=72, out_channels=72, kernel_size=(3, 3), padding=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        # 5
        # (72, 8, 8)
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=72, out_channels=144, kernel_size=(3, 3), padding=1),
                                    nn.ReLU())

        # 6
        # (144, 8, 8)
        self.layer6 = nn.Sequential(nn.Conv2d(in_channels=144, out_channels=144, kernel_size=(3, 3), padding=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        # FC layers
        # (144, 4, 4)
        self.fc_layers = nn.Sequential(
            nn.Linear(144 * 4 * 4, 144 * 4 * 4), nn.ReLU(),
            nn.Linear(144 * 4 * 4, 500), nn.ReLU(),
            nn.Linear(500, 2)
        )

        self.downSample = nn.Sequential(nn.Conv2d(in_channels=36, out_channels=144, kernel_size=(3, 3), padding=1),
                                        nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        self.relu = nn.ReLU()

    # forward pass
    def forward(self, x):
        # pdb.set_trace()
        l1 = self.layer1(x)
        l2 = self.layer2(l1)

        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)

        l5 = self.relu(l5 + self.downSample(l2))
        l6 = self.layer6(l5)

        l6 = l6.view(-1, 144 * 4 * 4)
        logits = self.fc_layers(l6)
        return logits
