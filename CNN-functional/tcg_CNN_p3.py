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
# INPUT: This Stage 3 script reads in the CNN trained model "tcg_CNN.model"
#        that is generated from Step 2.
#
#        Remarks: Note that the input data for this script must be on the
#        same as in Step 1 with standard 19 vertical
#        levels 1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600,
#        550, 500, 450, 400, 350, 300, 250, 200. Also, all field vars must
#        be resize to cover an area of 30x30 around the TC center for the
#        positive data cases.
#        Similar to Step 2, this Step 3 needs to also have a large mem
#        allocation so that it can be run properly.
#
# OUTPUT: A list of probability forecast with the same dimension as the
#        number of input 12-channel images.
#
# HIST: - 01, Nov 22: Created by CK
#       - 02, Nov 22: Modified to optimize it
#       - 05. Jun 23: Rechecked and added F1 score function for a list of model
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu)
#
#==========================================================================
import cv2
import tensorflow as tf
import os
from tqdm import tqdm
import netCDF4
import numpy as np
#
# Function to return input data as an numpy array
#
def prepare(filepath):
    IMG_SIZE = 30
    number_channels = 12
    f = netCDF4.Dataset(filepath)
    abv = f.variables['absvprs']
    nx = np.size(abv[0,0,:])
    ny = np.size(abv[0,:,0])
    nz = np.size(abv[:,0,0])
    #print('Dimension of input NETCDF is: ',nx,ny,nz)
    a2 = np.zeros((nx,ny,number_channels))
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,0] = abv[1,j,i]    # abs vort at 950 mb
    rel = f.variables['rhprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,1] = rel[7,j,i]    # RH at 750 mb
    sfc = f.variables['pressfc']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,2] = sfc[j,i]      # surface pressure
    tmp = f.variables['tmpprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,3] = tmp[15,j,i]   # temperature at 400 mb
    tsf = f.variables['tmpsfc']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,4] = tsf[j,i]      # surface temperature
    ugr = f.variables['ugrdprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,5] = ugr[3,j,i]    # u-wind at 900 mb
            a2[i,j,6] = ugr[17,j,i]   # u-wind at 300 mb
    vgr = f.variables['vgrdprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,7] = vgr[3,j,i]    # v-wind at 900 mb
            a2[i,j,8] = vgr[17,j,i]   # v-wind at 300 mb
    hgt = f.variables['hgtprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,9] = hgt[3,j,i]    # geopotential at 850 mb
    wgr = f.variables['vvelprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,10] = wgr[3,j,i]   # w-wind at 900 mb
            a2[i,j,11] = wgr[17,j,i]  # w-wind at 300 mb
    new_array = cv2.resize(a2, (IMG_SIZE, IMG_SIZE))
    #
    # normalize the data
    #
    #print('Number of channels to normalize is: ',number_channels)
    for var in range(number_channels):
        maxvalue = new_array[:,:,var].flat[np.abs(new_array[:,:,var]).argmax()]
        #print('Normalization factor for channel',var,', is: ',abs(maxvalue))
        new_array[:,:,var] = new_array[:,:,var]/abs(maxvalue)
    out_array = np.reshape(new_array, (-1, IMG_SIZE, IMG_SIZE, number_channels))
    #print('reshape new_array returns: ',out_array.shape)
    #input('Enter to continue...')
    return out_array
#
# build an F1-score function for later use
#
def F1_score(y_true,y_prediction,true_class,true_threshold):
    T = len(y_true)
    if len(y_prediction) != T:
        print("Prediction and true label arrays have different size. Stop")
        return
    P = 0
    TP = 0 
    FN = 0
    TN = 0
    FP = 0
    for i in range(T):
        if y_true[i] == true_class:
            P = P + 1       
            if y_prediction[i] >= true_threshold:
                TP += 1 
            else:
                FN += 1
        else:
            if y_prediction[i] >= true_threshold:
                FP += 1 
            else:
                TN += 1            
    N = T - P    
    F1 = 2.*TP/(2.*TP + FP + FN)
    Recall = TP/float(TP+FN)
    if TP == 0 and FP == 0: 
        Precision = 0.
    else:    
        Precision = TP/float(TP+FP)
    return F1, Recall, Precision
#
# loop thru all best-saved CNN trained models and make a prediction. Note that prediction is applied one by one instead 
# of a batch input. 
#
DATADIR = "/N/slate/ckieu/deep-learning/data/ncep_binary_30x30_00h/testing"
bestmodels = ["3-conv-32-layer-0-dense.model_00h","3-conv-32-layer-1-dense.model_00h","3-conv-32-layer-2-dense.model_00h",
              "5-conv-32-layer-0-dense.model_00h","5-conv-32-layer-1-dense.model_00h","5-conv-32-layer-2-dense.model_00h"]
CATEGORIES = ["neg", "pos"]
F1_performance = []
for bestmodel in bestmodels:
    model = tf.keras.models.load_model(bestmodel)
    prediction_total = 0
    prediction_yes = 0
    prediction_history = []
    truth_history = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        for img in tqdm(os.listdir(path)):    
            try:
                img_dir = DATADIR + '/' + category + '/' + img
                print('Processing image:', img_dir)
                #print('Input image dimension is: ',prepare(img_dir).shape)
                prediction = model.predict([prepare(img_dir)])
                print("TC formation prediction is",prediction,round(prediction[0][0]),CATEGORIES[round(prediction[0][0])])
                prediction_history.append(prediction[0][0])
                if round(prediction[0][0]) == 1:
                    prediction_yes += 1
                if category == "pos":
                    truth_history.append(1)
                else:
                    truth_history.append(0)
                prediction_total += 1    
                if prediction_total > 1000:
                    break
            except Exception as e:
                pass   
#
# Compute F1 score for each best model now
#
    print(prediction_history)
    F1_performance.append([bestmodel,F1_score(truth_history,prediction_history,1,0.5)]) 
#
# Print out the F1 performance of all models
#
print("F1, Recall, Precision for all models are:")
for i in range(len(bestmodels)):
    print("Model:", F1_performance[i])


