"""                      Week2 - Programming Assignment 3
"""

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


from keras.models import Sequential
from keras import regularizers



X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255.
X_test = X_test_orig/255.

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

# print ("number of training examples = " + str(X_train.shape[0]))
# print ("number of test examples = " + str(X_test.shape[0]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test.shape))



def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
   
    weight_decay = 1e-4;
    baseMapNum = 32
    model = Sequential()
    model.add(Conv2D(filters=baseMapNum, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), 
    input_shape=(input_shape,input_shape,3)))
    model.add(Activation('relu'))

    model.add(BatchNormalization())
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))

    model.add(BatchNormalization())
    model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))

    model.add(BatchNormalization())
    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    

    return model


"""
You have now built a function to describe your model. To train and test this model, there are four steps in Keras:
    1. Create the model by calling the function above
    2. Compile the model by calling model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])
    3. Train the model on train data by calling model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)
    4. Test the model on test data by calling model.evaluate(x = ..., y = ...)
"""

happyModel = HappyModel(X_train.shape[1])
happyModel.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])
happyModel.fit(x = X_train, y = Y_train, epochs = 15, batch_size = 32, shuffle=True)
# happyModel.summary()
# plot_model(happyModel)

preds = happyModel.evaluate(X_train, Y_train)
print ("Test Accuracy = " + str(preds[1]))
preds = happyModel.evaluate(X_test,Y_test)
print ("Test Accuracy = " + str(preds[1]))



""" Result : 
optimizer = "Adam", loss = "binary_crossentropy"
epochs = 15, batch_size = 32, shuffle=True
Test Accuracy = 0.9749999992052714
Test Accuracy = 0.9266666642824809
"""