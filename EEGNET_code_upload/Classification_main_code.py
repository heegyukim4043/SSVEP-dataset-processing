import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mpl
import scipy.io
from EEGModels import EEGNet
from EEGModels import ShallowConvNet
from EEGModels import DeepConvNet
from deepexplain.tensorflow import DeepExplain

import os
import random

tf.compat.v1.disable_eager_execution()

from keras import Model
from keras import utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

gpus = [0]

#kernels, chans, samples = 1, 32, 512
kernels, chans, samples = 1, 31, 512


#result_path = './result/testtest' #'./result_512_bi_CH'
result_path = './result_ssvep/testtest' #'./result_512_bi_CH'

if not os.path.isdir(result_path):
    os.mkdir(result_path)

         
for sub in range(1, 2):

    print('Start_dl')
    #dat_name = (".\proc_dat\subject%02d_traindata.mat" %(sub))
    #seq_name = ('.\proc_dat\subject%02d_trainseq.mat' %(sub))
    dat_name = (".\data_preproc_ssvep\subject%02d_traindata.mat" %(sub))
    seq_name = ('.\data_preproc_ssvep\subject%02d_trainseq.mat' %(sub))

    train_data_file = scipy.io.loadmat(dat_name)
    train_seq_file = scipy.io.loadmat(seq_name)

    #test_name = (".\proc_dat\subject%02d_testdata.mat" %(sub))
    #test_seq_name = ('.\proc_dat\subject%02d_testseq.mat' %(sub))
    test_name = (".\data_preproc_ssvep\subject%02d_testdata.mat" %(sub))
    test_seq_name = ('.\data_preproc_ssvep\subject%02d_testseq.mat' %(sub))


    test_data_file = scipy.io.loadmat(test_name)
    test_seq_file = scipy.io.loadmat(test_seq_name)

    total_acc = []

    #i = [1,2,3,4,5]
    i = [1, 2, 3, 4, 5, 6]
    random.shuffle(i)
    #for j in range(1, 5):
    for j in range(1, 7):
        '''
        if j==6:
            j=1
        
        print(i)
        nums = i[j-1]   
        print(nums)
        sleep(1)
        
        if j >= 6:
            j = j-5'''
        nums =j
        #model = EEGNet(nb_classes = 40, Chans = chans, Samples = samples, dropoutRate = 0.5,
        #        kernLength= 32, F1 = 8, D =2, F2 = 16, dropoutType = 'Dropout')
        model = EEGNet(nb_classes = 40, Chans = chans, Samples = samples, dropoutRate = 0.5,
                kernLength= 32, F1 = 8, D =2, F2 = 16, dropoutType = 'Dropout')
        '''      
        model = ShallowConvNet(nb_classes = 6, Chans = chans, Samples = samples, dropoutRate = 0.5)

        model = DeepConvNet(nb_classes=2, Chans=chans, Samples=samples, dropoutRate = 0.3)
                '''

        model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
        num_Par = model.count_params()    
        
        #class_weights = {0:1, 1:1, 2:1}
        class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
                         10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1,
                         20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1,
                         30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1}
        
        data_train = train_data_file['train_EEG']
        data_train = data_train[0]
        data_train = data_train[nums - 1]
        
        seq_train = train_seq_file['train_seq']
        seq_train = seq_train[0]
        seq_train = seq_train[nums - 1]
        seq_train = seq_train-10
        
        data_test = test_data_file['test_EEG']
        data_test = data_test[0]       
        data_test = data_test[nums - 1]
        
        seq_test = test_seq_file['test_seq']
        seq_test = seq_test[0]
        seq_test = seq_test[nums - 1]
        seq_test = seq_test-10

        checkpointer = ModelCheckpoint(filepath='./tmp_ssvep/checkpoint_s' + str(sub) + '_' + str(nums) +
                                '.h5', verbose = 0, save_best_only = True)

        data_train = np.transpose(data_train, (2,0,1))
        data_train = np.expand_dims(data_train, axis = -1)
        print(np.shape(data_train))

        seq_train = np.transpose(seq_train)
        data_test = np.transpose(data_test, (2,0,1))
        data_test = np.expand_dims(data_test, axis = -1)
        print(np.shape(data_test))

        seq_test = np.transpose(seq_test)

        data_train, data_val, seq_train, seq_val = train_test_split(
            data_train, seq_train, test_size = 0.2, stratify = seq_train)

        seq_train = utils.to_categorical(seq_train -1)
        seq_val = utils.to_categorical(seq_val - 1)
        seq_test = utils.to_categorical(seq_test -1)


        print('X_train shape:', data_train.shape)
        print('X_test shape:', data_test.shape)
        print('X_train seq shape:', seq_train.shape)

        print(data_train.shape[0], 'train samples')
        print(data_val.shape[0], 'val samples')
        print(seq_train.shape[0], 'seq samples')
        print(seq_val.shape[0], 'seq val samples')
        print(data_test.shape[0], 'test samples')

        #fittedModel = model.fit(data_train, seq_train, batch_size = 3, epochs = 500,
                              #  verbose=2, validation_data=(data_val, seq_val),
                               # callbacks=[checkpointer], class_weight=class_weights)
        fittedModel = model.fit(data_train, seq_train, batch_size = 32, epochs = 1000,
                                verbose=2, validation_data=(data_val, seq_val),
                                callbacks=[checkpointer], class_weight=class_weights)

        model.save('./tmp_ssvep/weight_final_sub' + str(sub) + '.h5')
        model.save_weights('./tmp_ssvep/weight_final_sub_only_weight' + str(sub) + '.h5')

        probs = model.predict(data_test)
        preds = probs.argmax(axis=-1)
        acc = np.mean(preds == seq_test.argmax(axis=-1))
        print("Classification accuracy: %f " % (acc))
        total_acc.append(acc)
        print('total_acc = ', total_acc)

        train_loss_curve = fittedModel.history['loss']
        val_loss_curve = fittedModel.history['val_loss']
        train_curve = fittedModel.history['acc']
        val_curve = fittedModel.history['val_acc']

        data_train = np.concatenate((data_train, data_val), axis=0)
        seq_train = np.concatenate((seq_train, seq_val), axis=0)

        train_probs = model.predict(data_train)
        train_preds = train_probs.argmax(axis=-1)
        train_acc = np.mean(train_preds == seq_train.argmax(axis=-1))
        print("Train Classification accuracy: %f " % (train_acc))
        print('sub =', sub, 'currently done: ', j)
        

        '''        
        with DeepExplain(session=K.get_session()) as de:
            input_tensor = model.layers[0].input
            fModel = Model(inputs=input_tensor, outputs=model.layers[-1].output)
            target_tensor = fModel(input_tensor)
            print(np.shape(input_tensor), np.shape(target_tensor))
            print(np.shape(seq_train), np.shape(data_train))

            # can use epsilon-LRP as well if you like.
            attributions = de.explain('deeplift', target_tensor*seq_train, input_tensor, data_train)
            attributions_test = de.explain('deeplift', target_tensor*seq_test, input_tensor, data_test)
            # attributions = de.explain('elrp', target_tensor * Y_test, input_tensor, X_test)
            np.size(attributions)
            scipy.io.savemat('./tmp/Subject' + str(sub)+'_'+str(nums) + '.mat',
                        {'attributions_train': attributions,
                        'train_true_label': seq_train,
                        'train_true_label_seq': seq_train.argmax(axis=-1),
                        'train_classified_label': train_preds,
                        'attributions_test': attributions_test,
                        'test_true_label': seq_test,
                        'test_true_label_seq':seq_test.argmax(axis=-1),
                        'test_classified_label': preds})
        
        '''
       

    mat_result_file = ('/result_label_Subject%02d_1_cold.mat' %(sub))    

    print('Total Acc: ', np.mean(total_acc))
    scipy.io.savemat(result_path + mat_result_file,
                {'acc': total_acc,
                    'train_loss_curve':train_loss_curve,
                    'train_curve':train_curve,
                    'val_loss_curve':val_loss_curve,
                    'val_curve':val_curve, 'seq':i
                    })



