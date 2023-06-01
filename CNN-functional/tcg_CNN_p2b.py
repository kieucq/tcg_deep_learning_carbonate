#
# NOTE: This machine learning program is for predicting TC formation, using
#       input dataset in the NETCDF format. The program treats different
#       2D input fields as different channels of an image. This specific
#       program requires a set of 12 2D-variables (12-channel image) and
#       consists of three stages
#       - Stage 1: reading NETCDF input and generating (X,y) data with a
#                  given image sizes, which are then saved by pickle;
#       - Stage 2: import the saved pickle (X,y) pair and build a CNN model
#                  with a given training/validation ratio, and then save
#                  the train model under tcg_CNN.model.
#       - Stage 3: import the trained model from Stage 2, and make a list
#                  of prediction from normalized test data.
#
# INPUT: This Stage 2 script requires two specific input datasets that are
#        generated from Step 1, including
#        1. tcg_X.pickle: data contains all images of yes/no TCG events, each
#           of these images must have 12 channels
#        2. tcg_y.pickle: data contains all labels of each image (i.e., yes
#           or no) of TCG corresponding to each data in X.
#
#        Remarks: Note that each channel must be normalized separealy. Also
#        the script requires a large memory allocation. So users need to have
#        GPU version to run this.
#
# OUTPUT: A CNN model built from Keras saved under tcg_CNN.model
#
# HIST: - 27, Oct 22: Created by CK
#       - 01, Nov 22: Modified to include more channels
#       - 17, Nov 23: cusomize it for jupiter notebook
#       - 21, Feb 23: use functional model instead of sequential model  
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu)
#
#==========================================================================
import tensorflow as tf
import numpy as np
import pickle
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
#
# read in data output from Part 1
#
pickle_in = open("tcg_X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("tcg_y.pickle","rb")
y = pickle.load(pickle_in)
y = np.array(y)
number_channels=X.shape[3]
print('Input shape of the X features data: ',X.shape)
print('Input shape of the y label data: ',y.shape)
print('Number of input channel extracted from X is: ',number_channels)
#
# normalize the data before training the model
#
nsample = X.shape[0]
for i in range(nsample):
    for var in range(number_channels):    
        maxvalue = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
        #print('Normalization factor for sample and channel',i,var,', is: ',abs(maxvalue))
        X[i,:,:,var] = X[i,:,:,var]/abs(maxvalue)
        maxnew = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
        #print('-->After normalization of sample and channel',i,var,', is: ',abs(maxnew))
        #input('Enter to continue...')
print("Finish normalization...")
#
# build a range of CNN models with different number of dense layers, layer sizes
# convolution layers to optimize the performance
#
dense_layers = [0]
layer_sizes = [32]
conv_layers = [3, 5]
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-layer-{}-dense.model_24".format(conv_layer, layer_size, dense_layer)
            print('--> Running configuration: ',NAME)

            inputs = keras.Input(shape=X.shape[1:])            
            x = layers.Conv2D(filters=layer_size,kernel_size=conv_layer,activation="relu",name="my_conv2d_1")(inputs)
            x = layers.MaxPooling2D(pool_size=2,name="my_pooling_1")(x)
            x = layers.Conv2D(filters=layer_size*2,kernel_size=conv_layer,activation="relu",name="my_conv2d_2")(x)
            x = layers.MaxPooling2D(pool_size=2,name="my_pooling_2")(x)
            if conv_layer == 3:
                x = layers.Conv2D(filters=layer_size*4,kernel_size=conv_layer,activation="relu",name="my_conv2d_3")(x)
                x = layers.MaxPooling2D(pool_size=2,name="my_pooling_3")(x)

            if X.shape[1] > 128:
                x = layers.Conv2D(filters=256,kernel_size=conv_layer,activation="relu",name="my_conv2d_4")(x)
                x = layers.MaxPooling2D(pool_size=2,name="my_pooling_4")(x)
                x = layers.Conv2D(filters=256,kernel_size=conv_layer,activation="relu",name="my_conv2d_5")(x)
            x = layers.Flatten(name="my_flatten")(x)
            outputs = layers.Dense(1,activation="sigmoid",name="my_dense")(x)
            model = keras.Model(inputs=inputs,outputs=outputs,name="my_functional_model")
            model.summary()
            keras.utils.plot_model(model)
            
            callbacks=[keras.callbacks.ModelCheckpoint(NAME,save_best_only=True)]
            model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
            history = model.fit(X, y, batch_size=90, epochs=30, validation_split=0.1, callbacks=callbacks)
#
# Visualize the output of the training model (work for jupyter notebook only)
#
import matplotlib.pyplot as plt
check_visualization = "no"
if check_visualization== "yes":
    #print(history.__dict__)
    #print(history.history)
    val_accuracy = history.history['val_accuracy']
    accuracy = history.history['accuracy']
    epochs = history.epoch 
    plt.plot(epochs,val_accuracy,'r',label="val accuracy")
    plt.plot(epochs,accuracy,'b',label="train accuracy")
    plt.legend()

    plt.figure()
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    plt.plot(epochs,val_loss,'r',label="val loss")
    plt.plot(epochs,loss,'b',label="train loss")
    plt.legend()
    plt.show()


