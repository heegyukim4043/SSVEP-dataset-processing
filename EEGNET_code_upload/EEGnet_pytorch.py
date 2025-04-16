from pytorch_eegnet import EEGNet_torch
from pytorch_Shallow import ShallowConvNet
from pytorch_deepconvnet import DeepConvNet
from EEGModels import EEGNet_SSVEP
from pytorch_eegnet import EEGNet_SSVEP_torch
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

batch_size = 32
nb_ep = 300
#kernel_size, chans, samples = 1, 44, 512
kernel_size, chans, samples = 1, 44, 512*2
result_path = './result_ssvep_2sec'
load_path = '.\data_preproc_fb_2sec'

#Shallow/test_500_cold_dropout_p5_triple' 
#torch_5fold_500epoch_cold_repeated' #'./result_512_bi_CH'
if not os.path.isdir(result_path):
    os.mkdir(result_path)

class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight = class_weights)


# Train the model
for sub in [x for x in range(1, 41)]:
    dat_name =  load_path + ("\subject%02d_traindata.mat" %(sub))
    seq_name = load_path + ("\subject%02d_trainseq.mat" %(sub))
    print(dat_name)
    print(seq_name)

    train_data_file = scipy.io.loadmat(dat_name)
    train_seq_file = scipy.io.loadmat(seq_name)

    test_name = load_path + ("\subject%02d_testdata.mat" %(sub))
    test_seq_name = load_path +('\subject%02d_testseq.mat' %(sub))

    test_data_file = scipy.io.loadmat(test_name)
    test_seq_file = scipy.io.loadmat(test_seq_name)

    total_acc = []
    
    i = [1, 2, 3, 4, 5]
    random.shuffle(i)


    for j in range(1,6):
        model = EEGNet_SSVEP_torch(nb_classes=40, Chans=chans, Samples=samples,
                                   dropoutRate=0.5, kernLength=64, F1=64,
                                   D=16, F2=16, dropoutType='Dropout').to(device)
        optimizer = opt.Adam(model.parameters())

        nums = i[j-1]
        data_train = train_data_file['train_EEG']
        data_train = data_train[0]
        data_train = data_train[nums - 1]
        
        seq_train = train_seq_file['train_seq']
        seq_train = seq_train[0]
        seq_train = seq_train[nums-1]
        #print('seq_train')

        #seq_train = seq_train-1
        #print(seq_train)
        
        data_test = test_data_file['test_EEG']
        data_test = data_test[0]       
        data_test = data_test[nums - 1]
        
        seq_test = test_seq_file['test_seq']
        seq_test = seq_test[0]
        seq_test = seq_test[nums-1]
        #print('seq_test')


        #seq_test = seq_test-1
        #print(seq_test)

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


        
        # train_label = F.one_hot(train_label.long() - 11)
        # seq_val = F.one_hot(seq_val.long() - 11)
        # seq_test = F.one_hot(seq_test.long() - 11)
        label_shift = train_label.min()
        train_label = train_label - label_shift
        train_label = F.one_hot(train_label.long())

        label_shift = seq_val.min()
        seq_val = seq_val - label_shift
        seq_val = F.one_hot(seq_val.long())

        label_shift = seq_test.min()
        seq_test = seq_test - label_shift
        seq_test = F.one_hot(seq_test.long())

        # save_seq_test = seq_test


        print(np.shape(train_label), 'seq_train')
        print(np.shape(data_train))
        train_dataset = TensorDataset(data_train.permute(0,3,1,2).to(device), 
                                      train_label.to(device))
        print(np.shape(data_train))
        train_loader = DataLoader(train_dataset, batch_size)
        val_dataset = TensorDataset(data_val.permute(0,3,1,2).to(device), 
                                    seq_val.to(device))
        val_loader = DataLoader(val_dataset, batch_size)
        test_dataset = TensorDataset(data_test.permute(0,3,1,2).to(device), seq_test.to(device))
        test_loader = DataLoader(test_dataset)
        #print(train_loader, 'input dataset')
     
        train_acc = []
        val_acc = []           
        for epoch in range(nb_ep):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                #print('inputsize:')
                #print(inputs.shape)

                if targets.ndim > 1:
                    targets = targets.argmax(dim=-1)

                targets = targets.long().to(device)

                optimizer.zero_grad()
                # targets = targets.float()
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                #correct += (predicted == targets.argmax(dim=-1)).sum().item()
                correct += (predicted == targets).sum().item()
            train_acc.append(correct/total)
            print('Epoch %d training loss: %.3f' % (epoch+1, loss.item()))
            
            
            model.eval()
            val_correct = 0
            val_total = 0
            for inputs, targets in val_loader:
                val_inputs = inputs
                val_outputs = model(val_inputs)

                if targets.ndim > 1:
                    targets = targets.argmax(dim=-1)  # [batch_size]

                targets = targets.long().to(device)

                val_targets = targets
                val_loss = criterion(val_outputs, val_targets)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += targets.size(0)
                #val_correct += (val_predicted == targets.argmax(dim=-1)).sum().item()
                val_correct += (val_predicted == targets).sum().item()
            val_acc.append(val_correct/val_total)
            print('Epoch %d validation loss: %.3f' % (epoch+1, val_loss.item()))
    
    
    
        # Save the PyTorch model
        torch.save(model.state_dict(), result_path +'/weight_final_sub' + 
                   str(sub) + '.pth')

        # Load the PyTorch model
        # model.load_state_dict(torch.load('./tmp/weight_final_sub' + str(sub) + '.pth'))

        # Make prediction on test set
        with torch.no_grad():
            model.eval()
            inputs = data_test.permute(0, 3, 1, 2).to(device)
            inputs.requires_grad_()

            probs = F.softmax(model(inputs), dim=-1)
            preds = torch.argmax(probs, dim=-1)
            targets = seq_test.argmax(dim=-1).to(device)


            acc = torch.mean((preds == targets).float())

            total_acc.append(acc)
            print("Classification accuracy: ", total_acc)
            preds = preds.cpu().numpy()
            #seq_test = seq_test.argmax(dim=-1).cpu().numpy()
            #deep_lift = DeepLift(model)
            deep_lift = DeepLift(model.to(device))
            labels = train_label.argmax(dim=-1).to(device)
            #labels_idx = labels.to(device)
            labels_idx = seq_test.argmax(dim=-1).to(device)

            print('inputs size')
            print(inputs.shape)

            print('label size')
            print(labels_idx.shape)

           # print(labels_idx)
            #print(save_seq_test)

           # print(labels)
            #attr_0 = deep_lift.attribute(data_train.permute(0,3,1,2).to(device), target = labels)
            #attr_0 = deep_lift.attribute(inputs.cpu(), target=labels.cpu())
            attr_0 = deep_lift.attribute(inputs, target=labels_idx)
           # print(np.shape(attr_0))
            attr_0 = attr_0.cpu().numpy()          
            scipy.io.savemat(result_path + '/Subject' + str(sub)+'_'+str(nums) + 
                             '.mat',
                        {'test_classified_Label':preds,
                         'test_true_label': labels_idx.cpu().numpy(),
                         'attributions': attr_0,
                         'labels':train_label.argmax(dim=-1).cpu().numpy(),
                         'train_acc': train_acc,
                         'val_acc': val_acc
                         })
        
    total_acc = np.array([tensor.cpu().numpy() for tensor in total_acc])
    mat_result_file = ('/result_label_Subject%02d_5fold.mat' %(sub))
    scipy.io.savemat(result_path + mat_result_file,
                {'acc': total_acc, 'seq': i}) #, 'seq': i
    
           

