import torch

class EEGNet_torch(torch.nn.Module):
    def __init__(self, chans = 22, times = 500, dropout = 0.5, kernelLength = 25, F1 = 8, D = 2, F2 = 16, num_classes = 4):
        super().__init__()

        self.cat_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, F1, (1, kernelLength), padding=(0, int(kernelLength / 2))),
            torch.nn.BatchNorm2d(F1),
            
            torch.nn.Conv2d(F1, F2, (chans, 1), groups=1),
            torch.nn.BatchNorm2d(F2),
            torch.nn.ELU(),
            torch.nn.AvgPool2d((1,4)),
            #torch.nn.BatchNorm2d(F2),
            torch.nn.Dropout(dropout),
            
            torch.nn.Conv2d(F2, F2, kernel_size=(1,int(kernelLength/2)), 
                            padding=(0,int(kernelLength/4)), groups=F2),
            torch.nn.Conv2d(F2, F2, kernel_size=1),
            torch.nn.BatchNorm2d(F2),
            torch.nn.ELU(),
            torch.nn.AvgPool2d((1,8)),
            # torch.nn.BatchNorm2d(F2),
            torch.nn.Dropout(dropout),

            torch.nn.Flatten(),
            torch.nn.Linear(F2*times/4/8, num_classes),

            torch.nn.Softmax()
        ) #순서가 고대 mshallownet 구조 (activation 통과 후 batchnorm: 3~4% 증가)


    def forward(self, x):
        x = self.cat_layers(x)
        return x
        
        
        
model = EEGNet_torch(
    chans=22,
    times=125,
    dropout=0.8,
    kernelLength=13,
    F1=8, D=3, F2=24,
    num_classes=4
    )