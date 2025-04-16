from sklearn.model_selection import train_test_split #
import numpy as np
import scipy.io as sio
import os
from deepexplain.tensorflow import DeepExplain

import tensorflow as tf
#tf.disable_eager_execution()
tf.executing_eagerly()

from EEGModels import EEGNet, EEGNet_SSVEP


from keras import Model
from keras import utils as np_utils
# from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K




## CPU = False, GPU = True
gpus = [1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #'#'.join(map(str, gpus))
os.environ['KMP_DUPLICATE_LIB_OK']='False'

# data param
nSub = 1
kernels, chans, samples = 1, 31, 512
result_path = './20230112_EEGNet'
if not os.path.isdir(result_path):
    os.mkdir(result_path)

total_data = sio.loadmat(
    './data_preproc_ssvep/subject01_traindata.mat')

test_tmp = sio.loadmat(
    './data_preproc_ssvep/subject01_testdata.mat')

train_label_load = sio.loadmat(
    './data_preproc_ssvep/subject01_trainseq.mat')

test_label_load = sio.loadmat(
    './data_preproc_ssvep/subject01_testseq.mat')



model = EEGNet(nb_classes=40, Chans=chans, Samples=samples,
                   dropoutRate=0.4, kernLength=32, F1=96, D=2, F2=96,
                   dropoutType='Dropout')
# model setup
#model = EEGNet_SSVEP(nb_classes = 40, Chans = chans, Samples = samples,
#             dropoutRate = 0.5, kernLength = 128, F1 = 96,
 #            D = 2, F2 = 96, dropoutType = 'Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# count number of parameters in the model
numParams = model.count_params()

# set a valid path for your system to record model checkpoints

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting # the weights all to be 1
class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4:1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1,
                 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1}

tot_acc = []


for sub in range(1,nSub+1):
    checkpointer = ModelCheckpoint(filepath='./tmp_ssvep/checkpoint_s' + str(sub) + '.h5', save_best_only=True, verbose=1)

    train_data = total_data
    train_label = train_label_load
    test_data = test_tmp
    test_label = test_label_load

    train_label = train_label['train_seq']
    train_label = train_label[0]
    train_label = train_label[0]
    train_label = train_label - 10 # labels: 1 to N
    train_label = np.transpose(train_label)

    train_data = train_data['train_EEG']
    train_data = train_data[0]
    train_data = train_data[0] # data: np.array ch x time x trial

    train_data = np.transpose(train_data, (2, 0, 1))  # trial x channal x time
    train_data = np.expand_dims(train_data, axis=-1) # trial x channal x time x 1

    test_label = test_label['test_seq']
    test_label = test_label[0]
    test_label = test_label[0]
    test_label = test_label - 10
    test_label = np.transpose(test_label)

    test_data = test_data['test_EEG']
    test_data = test_data[0]
    test_data = test_data[0]

    test_data = np.transpose(test_data, (2, 0, 1))
    test_data = np.expand_dims(test_data, axis=-1)

    train_data, val_data, train_label, val_label = train_test_split(train_data,train_label, test_size=0.2, stratify=train_label)

    train_label = np_utils.to_categorical(train_label - 1)
    val_label = np_utils.to_categorical(val_label - 1)
    test_label = np_utils.to_categorical(test_label - 1)


    print('X_train shape:', train_data.shape)
    print(train_data.shape, 'train samples')
    print(val_data.shape, 'val samples')
    print(test_data.shape, 'test samples')


    #trial x channel x time x kernel
    ################################################################################
    # fit the model. Due to very small sample sizes this can get
    # pretty noisy run-to-run, but most runs should be comparable to xDAWN +
    # Riemannian geometry classification (below)
    ################################################################################
    fittedModel = model.fit(train_data, train_label, batch_size=64, epochs=250,
                            verbose=2, validation_data=(val_data, val_label),
                            callbacks=[checkpointer], class_weight=class_weights)

    model.save('./tmp_ssvep/weight_final_sub' + str(sub) + '.h5')
    model.save_weights('./tmp_ssvep/weight_final_sub_only_weight' + str(sub) + '.h5')
    # load optimal weights
    # model.load_weights('/tmp/checkpoint.h5')

    ###############################################################################
    # can alternatively used the weights provided in the repo. If so it should get
    # you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
    # system.
    ###############################################################################

    # WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5
    # model.load_weights(WEIGHTS_PATH)

    ###############################################################################
    # make prediction on test set.
    ###############################################################################

    probs = model.predict(test_data)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == test_label.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))
    tot_acc.append(acc)


    #print('history',fittedModel.history.keys())

    train_loss_curve = fittedModel.history['loss']
    val_loss_curve = fittedModel.history['val_loss']
    #train_curve = fittedModel.history['accuracy']
    train_curve = fittedModel.history['acc']
    val_curve = fittedModel.history['val_acc']

    train_data = np.concatenate((train_data, val_data), axis=0)
    train_label = np.concatenate((train_label, val_label), axis=0)

    train_probs = model.predict(train_data)
    train_preds = train_probs.argmax(axis=-1)
    train_acc = np.mean(train_preds == train_label.argmax(axis=-1))
    print("Train Classification accuracy: %f " % (train_acc))

    # Deep explain XAI: https://github.com/marcoancona/DeepExplain
    with DeepExplain(session=K.get_session()) as de:
        print('stard_deepexplain')
        input_tensor = model.layers[0].input
        print('input_tensor')
        fModel = Model(inputs=input_tensor, outputs=model.layers[-2].output)
        print('fModel')
        target_tensor = fModel(input_tensor)
        print('target_tensor')

        print('input_tensor', input_tensor)
        print('target_tensor', target_tensor)
        print('xs', train_data.shape)
        print('ys', train_label.shape)

        # can use epsilon-LRP as well if you like.
        # attributions = de.explain('deeplift', target_tensor * train_label, input_tensor, train_data)
        attributions = de.explain('deeplift', target_tensor, input_tensor,xs= train_data,ys=train_label )
        attributions_test = de.explain('deeplift', target_tensor, input_tensor, xs=test_data,ys=test_label)
        # attributions = de.explain('elrp', target_tensor * Y_test, input_tensor, X_test)
        np.size(attributions)
        sio.savemat('./tmp_ssvep/Subject' + str(sub) + '.mat',
                    {'attributions_train': attributions,
                     'train_true_label': train_label,
                     'train_classified_label': train_preds,
                     'attributions_test': attributions_test,
                     'test_true_label': test_label,
                     'test_classified_label': preds})

print(tot_acc)
print('Total Acc: ', np.mean(tot_acc))

sio.savemat(result_path + '/result_label_Subject.mat',
            {'acc': tot_acc,
             'train_loss_curve':train_loss_curve,
             'train_curve':train_curve,
             'val_loss_curve':val_loss_curve,
             'val_curve':val_curve
             })
