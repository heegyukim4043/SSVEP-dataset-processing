import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mpl
import scipy.io
from EEGModels import EEGNet
from EEGModels import ShallowConvNet
from EEGModels import DeepConvNet
from sklearn.metrics import f1_score
from deepexplain.tensorflow import DeepExplain

import os

#tf.compat.v1.disable_eager_execution()

from tensorflow.keras import Model
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

gpus = 0