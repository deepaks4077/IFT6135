from torch import nn
import pdb


class CNNModel(nn.Module):
    def __init__(self, params):
        super(CNNModel, self).__init__()
        self.params = params

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.act2 = nn.ReLU()

        self.ff = nn.Linear(196, 10)  # 4*7*7

    def forward(self, batch):
        '''
        batch: (B x 1 x 28 x 28)
        '''
        # pdb.set_trace()
        l1 = self.conv1(batch[0])  # (B x 2 x 28 x 28)
        l1 = self.pool1(l1)  # (B x 2 x 14 x 14)
        l1 = self.act1(l1)  # (B x 2 x 14 x 14)

        l2 = self.conv2(l1)  # (B x 4 x 14 x 14)
        l2 = self.pool2(l2)  # (B x 4 x 7 x 7)
        l2 = self.act2(l2)  # (B x 4 x 7 x 7)

        l2 = l2.view(batch[0].size()[0], -1)

        logits = self.ff(l2)
        # pdb.set_trace()
        return logits
