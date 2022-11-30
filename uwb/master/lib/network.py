import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _,_= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1,1)
        return x * y.expand_as(x)

class UWBNet(nn.Module):
    def __init__(self):
        super(UWBNet, self).__init__()
        self.selayer = SELayer(1)
        #self.conv0 = torch.nn.Conv2d(1, 16, (1,1))
        self.conv1 = torch.nn.Conv2d(1, 6, (3,3),padding=1)
        self.pool1 = torch.nn.AvgPool2d(2)
        self.bn1 = torch.nn.BatchNorm2d(6)
        self.conv2 = torch.nn.Conv2d(6, 16, (3,3),padding=1)
        self.pool2 = torch.nn.AvgPool2d(2)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.fc1 = torch.nn.Linear(16,120)
        self.fc2 = torch.nn.Linear(120,3)
        self.sigmoid =  torch.nn.Sigmoid()
    def forward(self, x):
        batch = x.shape[0]
        x = x.unsqueeze(1)
        #x = self.conv0(x)
        #x = self.selayer(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        if batch != 1:
            x= self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        if batch !=1:
            x = self.bn2(x)
        x=x.view(batch,-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class UWBdataNet(nn.Module):
    def __init__(self):
        super(UWBdataNet, self).__init__()
        #self.selayer = SELayer(1)
        #self.conv0 = torch.nn.Conv2d(1, 16, (1,1))
        self.conv1 = torch.nn.Conv1d(1, 6, 3,padding=1)
        self.pool1 = torch.nn.AvgPool1d(2)
        self.bn1 = torch.nn.BatchNorm1d(6)
        self.conv2 = torch.nn.Conv1d(6, 16, 3,padding=1)
        self.pool2 = torch.nn.AvgPool1d(2)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.fc1 = torch.nn.Linear(64,16)
        self.fc2 = torch.nn.Linear(16, 1)
        #self.fc3 = torch.nn.Linear(64, 1)

        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        batch = x.shape[0]
        x = x.unsqueeze(1)
        #x = self.conv0(x)
        #x = self.selayer(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        if batch != 1:
            x= self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        if batch !=1:
            x = self.bn2(x)
        x=x.view(batch,-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


