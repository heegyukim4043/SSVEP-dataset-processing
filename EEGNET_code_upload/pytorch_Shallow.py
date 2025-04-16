import torch
import torch.nn as nn

'''
                     ours        original paper        for 512
    pool_size        1, 35       1, 75                 1, 155
    strides          1, 7        1, 15                 1, 30
    conv filters     1, 13       1, 25                 1, 50
'''

class ShallowConvNet(nn.Module):
    def __init__(self, nb_classes, Chans, Samples, 
                 dropoutRate = 0.5):
        super(ShallowConvNet, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size = (1, 50)),
            nn.BatchNorm2d(40, eps = 1e-5, momentum = 0.9),
            nn.Dropout(dropoutRate),
            nn.Conv2d(40, 40, kernel_size=(Chans, 1), bias = False),
            nn.BatchNorm2d(40, eps = 1e-5, momentum = 0.9),
            nn.Dropout(dropoutRate)
        )
        self.pooling = nn.AvgPool2d(kernel_size = (1, 155), stride = (1, 30))
        self.dropout = nn.Dropout(dropoutRate)
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(40*11, nb_classes),
            nn.Softmax(dim =1)            
        )
    def forward(self, x):
        x = self.block1(x)
        x = torch.pow(x, 2)
        x = self.pooling(x)
        x = torch.log(x)
        x = self.dropout(x)
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