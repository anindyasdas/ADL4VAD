import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.networks import build_feature_extractor, NET_OUT_DIM


class HolisticHead(nn.Module):
    def __init__(self, in_dim, dropout=0):
        super(HolisticHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        emb=F.relu(self.fc1(x))
        x = self.drop(emb)
        x = self.fc2(x)
        return torch.abs(x)

class PlainHead(nn.Module):
    def __init__(self, args, in_dim, topk_rate=0.1):
        super(PlainHead, self).__init__()
        self.args=args
        self.scoring = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, padding=0)
        self.topk_rate = topk_rate

    def forward(self, x):
        
        x = self.scoring(x)
        x = x.view(int(x.size(0)), -1)
        topk = max(int(x.size(1) * self.topk_rate), 1)
        x = torch.topk(torch.abs(x), topk, dim=1)[0]
        x = torch.mean(x, dim=1).view(-1, 1)
        return x
        
class ProbHead(nn.Module):
    def __init__(self, in_dim):
        super(ProbHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc=nn.Linear(in_dim,1)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(int(x.size(0)), -1)
        x=F.sigmoid(self.fc(x))
        return x
        

class SemiADNet(nn.Module):
    def __init__(self, args):
        super(SemiADNet, self).__init__()
        self.args = args
        self.feature_extractor = build_feature_extractor(self.args.backbone)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.in_c = NET_OUT_DIM[self.args.backbone]+NET_OUT_DIM[self.args.backbone]//2+NET_OUT_DIM[self.args.backbone]//4 #NET_OUT_DIM[backbone]
        base_width=64
        out_channel=2
        self.anomaly_head = PlainHead(self.args, self.in_c, self.args.topk)
        self.prob_head = ProbHead(NET_OUT_DIM[self.args.backbone])
        self.db1 = nn.Sequential(
            nn.Conv2d(self.in_c, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), #replacing with deterministic process -nearesr neighbor
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
                                 
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channel, kernel_size=3, padding=1))
                                 
        
        



        


        
    def forward(self, image):

        image_scaled =  image
        feature, x_feature = self.feature_extractor(image_scaled)
        out_mask= self.fin_out(self.up3(self.up2(self.up1(self.db1(feature)))))
        
        abnormal_scores = self.anomaly_head(feature)
        prob=self.prob_head(x_feature)
            
            
        return out_mask, abnormal_scores, prob
