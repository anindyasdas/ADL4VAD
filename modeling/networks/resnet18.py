import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F




class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])
        
        
def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
                     
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block
    
    
def idBlock(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

class feature_resnet18(nn.Module):
    def __init__(self):
        super(feature_resnet18, self).__init__()
        self.net = models.resnet18(pretrained=True)
        #upsample layer3
        self.upsample4_1 = upBlock(512, 512)
        self.upsample4_2 = upBlock(512, 512)
        self.upsample3_1 = upBlock(256, 256)
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x1 = self.net.layer1(x)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        x3_up =self.upsample3_1(x3)
        x4_up=self.upsample4_2(self.upsample4_1(x4))
        return torch.cat([x2, x3_up, x4_up], dim=1), x4
        
class feature_resnet50_2(nn.Module):
    def __init__(self):
        super(feature_resnet50_2, self).__init__()
        self.net = models.wide_resnet50_2(pretrained=True)
        #upsample layer3
        self.upsample4_1 = upBlock(2048, 2048)
        self.upsample4_2 = upBlock(2048, 2048)
        self.upsample3_1 = upBlock(1024, 1024)
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x1 = self.net.layer1(x)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        x3_up =self.upsample3_1(x3)
        x4_up=self.upsample4_2(self.upsample4_1(x4))
        return torch.cat([x2, x3_up, x4_up], dim=1), x4
        
        

        
        
class feature_wide_resnet50(nn.Module):
    def __init__(self):
        super(feature_wide_resnet50, self).__init__()
        self.net = models.wide_resnet50_2(pretrained=True)
        #upsample layer3
        self.upsample4_1 = upBlock(2048, 512)
        self.upsample4_2 = upBlock(512, 512)
        #self.upsample4_3 = upBlock(512, 512)
        self.upsample3_1 = upBlock(1024, 256)
        
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x1 = self.net.layer1(x)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        x5= self.net.avgpool(x4)
        x5 = x5.view(int(x5.size(0)), -1)
        x5=self.net.fc(x5)
        
        return x5, x4
        


class feature_resnet50(nn.Module):
    def __init__(self):
        super(feature_resnet50, self).__init__()
        self.net = models.wide_resnet50_2(pretrained=True)
        self.upsample4_1 = upBlock(2048, 512)
        self.upsample4_2 = upBlock(512, 512)
        self.upsample3_1 = upBlock(1024, 256)
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x1 = self.net.layer1(x)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x3_up = self.upsample3_1(x3)
        x4 = self.net.layer4(x3)
        x4_up=self.upsample4_2(self.upsample4_1(x4))
        x=torch.cat([x2, x3_up, x4_up], dim=1)
        return x

