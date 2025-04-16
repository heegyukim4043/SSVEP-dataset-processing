from pytorch_eegnet import EEGNet_torch
from pytorch_Shallow import ShallowConvNet
from pytorch_deepconvnet import DeepConvNet
from draw_convnet import draw_convnet
import torch
import torch.nn as nn
import os
import torch.optim as opt


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = [0]
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device: ', device)

kernel_size, chans, samples = 1, 32, 512
'''
model = EEGNet_torch(nb_classes = 6, Chans = chans, Samples = samples, 
                dropoutRate=0.5, kernelLength=32, F1=8, D=2, F2=16,
                dropoutType='Dropout').to(device)

model = ShallowConvNet(nb_classes =6, Chans = chans, Samples = samples, 
                        dropoutRate = 0.5).to(device)
'''
model = DeepConvNet(nb_classes =6, Chans = chans, Samples = samples, 
                        dropoutRate = 0.5).to(device)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight = class_weights)
optimizer = opt.Adam(model.parameters())

draw_convnet(model, (268, 1, 32, 512))

