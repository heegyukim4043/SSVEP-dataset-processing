from sklearn.model_selection import train_test_split #
import numpy as np
import scipy.io as sio
import os
from deepexplain.tensorflow import DeepExplain

import tensorflow as tf
#tf.disable_eager_execution()
tf.executing_eagerly()

from EEGModels import EEGNet, EEGNet_SSVEP

import keras
from keras import Model
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import model_from_json

import json
import h5py
from keras.models import load_model



sub = 1
#model_path = './tmp_ssvep/weight_final_sub1.h5'
#custom_objects = {"Functional": tf.keras.Model}

model = load_model('./tmp_ssvep/weight_final_sub1.h5')



total_data = sio.loadmat(
        './data_preproc_ssvep/subject01_traindata.mat')

test_tmp = sio.loadmat(
        './data_preproc_ssvep/subject01_testdata.mat')

train_label_load = sio.loadmat(
        './data_preproc_ssvep/subject01_trainseq.mat')

test_label_load = sio.loadmat(
        './data_preproc_ssvep/subject01_testseq.mat')

train_data = total_data
train_label = train_label_load
test_data = test_tmp
test_label = test_label_load

train_label = train_label['train_seq']
train_label = train_label[0]
train_label = train_label[0]
train_label = train_label - 10  # labels: 1 to N
train_label = np.transpose(train_label)

train_data = train_data['train_EEG']
train_data = train_data[0]
train_data = train_data[0]  # data: np.array ch x time x trial

train_data = np.transpose(train_data, (2, 0, 1))  # trial x channal x time
train_data = np.expand_dims(train_data, axis=-1)  # trial x channal x time x 1

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

train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.2,
                                                                stratify=train_label)

train_label = np_utils.to_categorical(train_label - 1)
val_label = np_utils.to_categorical(val_label - 1)
test_label = np_utils.to_categorical(test_label - 1)

probs = model.predict(test_data)
preds = probs.argmax(axis=-1)

train_probs = model.predict(train_data)
train_preds = train_probs.argmax(axis=-1)






with DeepExplain(session=K.get_session()) as de:
    print('start')
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs=model.layers[-2].output)
    target_tensor = fModel(input_tensor)

    print('load model')
    # can use epsilon-LRP as well if you like.
    #attributions = de.explain('deeplift', target_tensor * train_label, input_tensor, train_data)
    attributions = de.explain('grad*input', target_tensor, input_tensor, test_data, ys=test_label)

    print('attributions')

    #attributions_test = de.explain('deeplift', target_tensor * test_label, input_tensor, test_data)
    #print('attributions_test')

    #np.size(attributions)
    print(np.size(attributions))
    """
    sio.savemat('./tmp_ssvep/Subject' + str(sub) + '.mat',
                {'attributions_train': attributions,
                 'train_true_label': train_label,
                 'train_classified_label': train_preds,
                 'attributions_test': attributions_test,
                 'test_true_label': test_label,
                 'test_classified_label': preds})
    """