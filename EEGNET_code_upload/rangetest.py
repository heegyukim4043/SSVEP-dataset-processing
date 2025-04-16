import scipy.io
import numpy as np

sub = 7
dat_name = (".\data_deep\data_512_bi_TC\subject%02d_traindata.mat" %(sub))
train_data_file = scipy.io.loadmat(dat_name)
data_train = train_data_file['train_EEG']

for nums in range(1, 11):
    
    data_train = train_data_file['train_EEG']
    data_train = data_train[0]
    
    print(np.shape(data_train))
    print(np.shape(data_train[nums-1]))
    
    data_train = data_train[nums - 1]

    print(np.shape(data_train))


from pytorch_eegnet import EEGNet_torch
from pytorch_Shallow import ShallowConvNet
from pytorch_deepconvnet import DeepConvNet
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mpl
import scipy.io
import os
import torch
import torch.optim as opt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from captum.attr import DeepLift
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = [0]
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device: ', device)

kernel_size, chans, samples = 1, 32, 512
result_path = './result/shallow/1000epoch_cool'
#Shallow/test_500_cold_dropout_p5_triple' 
#torch_5fold_500epoch_cold_repeated' #'./result_512_bi_CH'
if not os.path.isdir(result_path):
    os.mkdir(result_path)

exclude = [22,30]
'''
model = EEGNet_torch(nb_classes = 6, Chans = chans, Samples = samples, 
                dropoutRate=0.7, kernelLength=32, F1=8, D=2, F2=16,
                dropoutType='Dropout').to(device)
'''
model = ShallowConvNet(nb_classes =3, Chans = chans, Samples = samples, 
                        dropoutRate = 0.7).to(device)
'''
model = DeepConvNet(nb_classes =6, Chans = chans, Samples = samples, 
                        dropoutRate = 0.7).to(device)
'''
class_weights = torch.tensor([1.0, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight = class_weights)
optimizer = opt.Adam(model.parameters())

# Train the model
for sub in [x for x in range(2, 44) if x not in exclude]:
    dat_name = (".\data_deep\labelsrand_5fold_allnorm\subject%02d_traindata.mat" %(sub))
    seq_name = ('.\data_deep\labelsrand_5fold_allnorm\subject%02d_trainseq.mat' %(sub))

    train_data_file = scipy.io.loadmat(dat_name)
    train_seq_file = scipy.io.loadmat(seq_name)

    test_name = (".\data_deep\labelsrand_5fold_allnorm\subject%02d_testdata.mat" %(sub))
    test_seq_name = ('.\data_deep\labelsrand_5fold_allnorm\subject%02d_testseq.mat' %(sub))

    test_data_file = scipy.io.loadmat(test_name)
    test_seq_file = scipy.io.loadmat(test_seq_name)

    total_acc = []
    
    i = [1, 2, 3, 4, 5]
    random.shuffle(i)


    for j in range(1,6):
        nums = i[j-1]
        data_train = train_data_file['train_EEG']
        data_train = data_train[0]
        data_train = data_train[nums - 1]
        
        seq_train = train_seq_file['train_seq']
        seq_train = seq_train[0]
        seq_train = seq_train[nums-1]
        
        data_test = test_data_file['test_EEG']
        data_test = data_test[0]       
        data_test = data_test[nums - 1]
        
        seq_test = test_seq_file['test_seq']
        seq_test = seq_test[0]
        seq_test = seq_test[nums-1]
        '''
        checkpointer = ModelCheckpoint(filepath='./tmp/checkpoint_s' + str(sub) + '_' + str(nums) +
                                '.h5', verbose = 0, save_best_only = True)
                                '''                    
        #data_train = torch.from_numpy(data_train)
        data_train =torch.from_numpy(data_train.transpose(2,0,1)).float().unsqueeze(-1)
        print(np.shape(data_train))        
        seq_train =  torch.from_numpy(seq_train).squeeze().long()

        print(np.shape(seq_train))

        data_test = torch.from_numpy(data_test.transpose(2,0,1)).float().unsqueeze(-1)
        print(np.shape(data_test))
        seq_test =  torch.from_numpy(seq_test).squeeze().long()
        
        data_train, data_val, train_label, seq_val = train_test_split(
            data_train, seq_train, test_size = 0.2, stratify = seq_train)
        print(np.shape(train_label), 'size')

        num_classes = torch.unique(train_label).size(0)
        print(num_classes)
        
        train_label = F.one_hot(train_label.long() - 1)
        seq_val = F.one_hot(seq_val.long() - 1)
        seq_test = F.one_hot(seq_test.long() - 1)

        batch_size = 32
        print(np.shape(train_label), 'seq_train')
        print(np.shape(data_train))
        train_dataset = TensorDataset(data_train.permute(0,3,1,2).to(device), 
                                      train_label.to(device))
        print(np.shape(data_train))
        train_loader = DataLoader(train_dataset, batch_size)
        val_dataset = TensorDataset(data_val.permute(0,3,1,2).to(device), 
                                    seq_val.to(device))
        val_loader = DataLoader(val_dataset, batch_size)
        test_dataset = TensorDataset(data_test.permute(0,3,1,2).to(device), 
                                     seq_test.to(device))
        test_loader = DataLoader(test_dataset)
       # print(train_loader, 'input dataset')