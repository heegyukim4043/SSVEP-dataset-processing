import torch
import torch.nn as nn
import torch.nn.functional as f

class DeepConvNet(nn.Module):
    def __init__(self, nb_classes, Chans = 64, Samples = 256, dropoutRate = 0.6):
        super(DeepConvNet, self).__init__()
        
        #defining layers
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,10)),
            nn.Conv2d(25, 25, kernel_size = (Chans, 1)),
            nn.BatchNorm2d(25, eps = 1e-05, momentum = 0.9),
            nn.Dropout(dropoutRate),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = (1,3), stride = (1,3)),
            nn.Dropout(dropoutRate)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size =(1,10)),
            nn.BatchNorm2d(50, eps = 1e-05, momentum = 0.9),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = (1,3), stride = (1,3)),
            nn.Dropout(dropoutRate)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 10)),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(dropoutRate)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 10)),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(dropoutRate)
        )
        
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(200, nb_classes),
            nn.BatchNorm1d(nb_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
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