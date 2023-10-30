import torch
import torch.nn as nn
from tnet import TNet
from utils.tools import conv_batch, fc_batch


class PointNet(nn.Module):
    """
    PointNet
    """
    def __init__(self, k, mode="cls", device="cpu"):
        """
        :param k: 类别个数
        """
        super(PointNet, self).__init__()
        self.k = k
        self.mode = mode
        self.device = device

        self.input_transform = TNet(conv_channels=[(3, 64), (64, 128), (128, 1024)], device=device)
        self.feature_transform = TNet(conv_channels=[(64, 64), (64, 128), (128, 1024)], device=device)

        self.conv1 = conv_batch(3, 64)
        self.conv2 = conv_batch(64, 128)
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=1024)
        )

        assert mode in ["cls", "seg"], "mode just only be 'cls' or 'seg'"

        if mode == "cls":
            # use for classification
            self.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(256, k)
            )
        elif mode == "seg":
            # use for segmentation
            self.seg_conv = nn.Sequential(
                conv_batch(1088, 512),
                conv_batch(512, 256),
                conv_batch(256, 128),
                nn.Conv1d(128, k, 1, 1)
            )

    def forward(self, x):
        """
        :param x: shape: [-1, d, n] (d=3)
        :return:  shape:
        """
        n = x.shape[2]

        # first T-net compute
        trans = self.input_transform(x)  # [-1, d, d]
        x = x.permute(0, 2, 1)  # [-1, d, n] -> [-1, n, d]
        x = torch.bmm(x, trans) # [-1, n, d] @ [-1, d, d] -> [-1, n, d]
        x = x.permute(0, 2, 1)  # [-1, d, n]

        x = self.conv1(x) # [-1, 64, n]

        # second T-net compute
        trans = self.feature_transform(x)
        x = x.permute(0, 2, 1)
        x = torch.bmm(x, trans)
        points_feature = x.permute(0, 2, 1)  # [-1, 64, n]

        x = self.conv2(points_feature)
        x = self.conv3(x)   # [-1, 1024, n]

        global_feature = torch.max(x, dim=2, keepdim=True)[0].view(-1, 1024) # [-1, 1024]

        if self.mode == "cls":
            logits = self.classifier(global_feature)
            return logits  # [-1, k]
        elif self.mode == "seg":
            global_feature = global_feature.unsqueeze(-1).repeat(1, 1, n)  # [-1, 1024, n]
            points_feature = torch.cat((points_feature, global_feature), dim=1)
            logits = self.seg_conv(points_feature)

            logits = logits.permute(0, 2, 1).contiguous()
            return logits  # [-1, n, k]


if __name__ == '__main__':
    x = torch.randn(4, 16, 3).permute(0, 2, 1)

    net = PointNet(k=10, mode="cls", device="cpu")
    y = net(x)
    print(y.shape)



        
