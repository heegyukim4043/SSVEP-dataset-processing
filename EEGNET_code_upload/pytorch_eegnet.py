'''
The input to the network is a 4D tensor with dimensions 
(batch_size, 1, n_channels, n_samples), 
where batch_size is the number of samples in each mini-batch, 
n_channels is the number of EEG channels, 
and n_samples is the number of time samples per channel. 
The output is a 2D tensor with dimensions (batch_size, n_classes), 
where n_classes is the number of target classes.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet_torch(nn.Module):
    def __init__(self, nb_classes, Chans, Samples, dropoutRate=0.5, kernelLength=32,
                 F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet_torch, self).__init__()

        if dropoutType == 'SpatialDropout2D':
            self.dropoutType = nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.dropoutType = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernelLength), padding=(0, int(kernelLength / 2)),
                      bias = False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (Chans, 1), groups = F1),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            self.dropoutType(dropoutRate)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16),  
                      padding=(0,int(kernelLength/4)), groups=F2, bias = False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.BatchNorm2d(F2),
            self.dropoutType(dropoutRate)
        )

        self.flatten = nn.Flatten()
        #flatten 이후에 너무 많다 싶으면 layer 추가해서 좀 줄여.
        # norm(불필요), dropout(요망)
        
        self.dense = nn.Sequential(
            nn.Linear(F2 * int(Samples / 32), nb_classes),
            nn.BatchNorm1d(nb_classes),
            nn.Softmax()
        ) 

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)


class EEGNet_SSVEP_torch(nn.Module):
    def __init__(self, nb_classes=12, Chans=8, Samples=256,
                 dropoutRate=0.5, kernLength=256, F1=96,
                 D=1, F2=96, dropoutType='Dropout'):
        super(EEGNet_SSVEP_torch, self).__init__()

        # Dropout type
        if dropoutType == 'SpatialDropout2D':
            self.dropout = nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.dropout = nn.Dropout
        else:
            raise ValueError("dropoutType must be 'SpatialDropout2D' or 'Dropout'")

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            self.dropout(dropoutRate)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            self.dropout(dropoutRate)
        )

        # ▶️ 이 dummy forward로 flatten size 자동 측정
        with torch.no_grad():
            x = torch.zeros(1, 1, Chans, Samples)
            x = self.firstconv(x)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            self.flatten_dim = x.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, nb_classes)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classifier(x)
        return F.softmax(x, dim=1)