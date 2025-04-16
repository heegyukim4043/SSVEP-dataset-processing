from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile, sys, os
sys.path.insert(0, os.path.abspath('..'))

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Ensure TensorFlow 1.x compatibility


import keras
from keras.datasets import mnist
from keras.optimizers import Adadelta
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from utils_deepexplain import plot, plt

# Import DeepExplain
from deepexplain.tensorflow import DeepExplain

# Build and train a network.

batch_size = 128
num_classes = 10
epochs = 3

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = (x_train - 0.5) * 2
x_test = (x_test - 0.5) * 2
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
# ^ IMPORTANT: notice that the final softmax must be in its own layer
# if we want to target pre-softmax units

model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

tf.disable_eager_execution()
session = tf.Session()
tf.keras.backend.set_session(session)


print("Session created:", session)
# K.get_session()

session.run(tf.global_variables_initializer())

print('start')
with DeepExplain(session=session) as de:  # <-- init DeepExplain context
    # Need to reconstruct the graph in DeepExplain context, using the same weights.
    # With Keras this is very easy:a
    # 1. Get the input tensor to the original model
    input_tensor = model.layers[0].input

    # 2. We now target the output of the last dense layer (pre-softmax)
    # To do so, create a new model sharing the same layers untill the last dense (index -2)
    fModel = Model(inputs=input_tensor, outputs=model.layers[-2].output)
    target_tensor = fModel(input_tensor)

    xs = x_test[0:10]
    ys = y_test[0:10]

    print('input_tensor',input_tensor)
    print('target_tensor',target_tensor)
    print('xs',xs.shape)
    print('ys',ys.shape)

    attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
    # attributions_sal   = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
    # attributions_ig    = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys)
    # attributions_dl    = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
    # attributions_elrp  = de.explain('elrp', target_tensor, input_tensor, xs, ys=ys)
    # attributions_occ   = de.explain('occlusion', target_tensor, input_tensor, xs, ys=ys)

    # Compare Gradient * Input with approximate Shapley Values
    # Note1: Shapley Value sampling with 100 samples per feature (78400 runs) takes a couple of minutes on a GPU.
    # Note2: 100 samples are not enough for convergence, the result might be affected by sampling variance
    attributions_sv = de.explain('shapley_sampling', target_tensor, input_tensor, xs, ys=ys, samples=100)
print('end')

n_cols = 6
n_rows = int(len(attributions_gradin) / 2)
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3*n_cols, 3*n_rows))

print('plot')
for i, (a1, a2) in enumerate(zip(attributions_gradin, attributions_sv)):
    row, col = divmod(i, 2)
    plot(xs[i].reshape(28, 28), cmap='Greys', axis=axes[row, col*3]).set_title('Original')
    plot(a1.reshape(28,28), xi = xs[i], axis=axes[row,col*3+1]).set_title('Grad*Input')
    plot(a2.reshape(28,28), xi = xs[i], axis=axes[row,col*3+2]).set_title('Shapley Values')

plt.show()