import torch
import torch.nn as nn
from utils.tools import conv_batch, fc_batch

class TNet(nn.Module):
    """
    Input or Feature transform module for PointNet
    """
    
    def __init__(self, conv_channels, device="cpu"):
        """
        :param conv_channels: 多层卷积维度
        :param device:
        """
        super(TNet, self).__init__()
        self.d = conv_channels[0][0]
        self.device = device

        self.conv1 = conv_batch(*conv_channels[0])
        self.conv2 = conv_batch(*conv_channels[1])
        self.conv3 = conv_batch(*conv_channels[2])

        self.fc1 = fc_batch(1024, 512)
        self.fc2 = fc_batch(512, 256)
        self.weight = nn.Linear(256, self.d * self.d, bias=False)

    def forward(self, x):
        """
        :param x: shape: [-1, d, n]
        :return:  shape: [-1, d, d]
        """
        batch_size = x.shape[0]

        x = self.conv1(x)   # [-1, 64, n]
        x = self.conv2(x)   # [-1, 128, n]
        x = self.conv3(x)   # [-1, 1024, n]

        x = torch.max(x, dim=2, keepdim=True)[0]  # [-1, 1024, 1]
        x = x.view(-1, 1024)  # [-1, 1024]

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.weight(x)   # [-1, d*d]

        iden = torch.eye(self.d, requires_grad=True).flatten().view(1, self.d * self.d)\
            .repeat(batch_size, 1).to(self.device)  # iden.shape: [-1, d*d]

        x += iden  # [-1, d*d]
        x = x.view(-1, self.d, self.d)  # [-1, d, d]

        return x


if __name__ == '__main__':
    x = torch.randn(4, 16, 3).permute(0, 2, 1)

    Tnet = TNet(conv_channels=[(3, 64), (64, 128), (128, 1024)])
    y = Tnet(x)
    print(y.shape)