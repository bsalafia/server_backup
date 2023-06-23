











# np.save(os.path.join(SaveResults,'accuracyCNNSMILELOO'),score)
# np.save(os.path.join(SaveResults,'lossCNNSMILELOO'),loss1)

# print(EDFFiles[indices[0]])

#   print(indices)
#   test.append(test_index)

# # print(test[2])

# for z in test:
#   indices=ReadMatFiles(dirname,z)
#   print(EDFFiles)
# -*- coding: utf-8 -*-
"""1DCNNServer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XIaeyIZmYN-0bMvxywlL5M5nTjaRBOJe
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
from keras import optimizers, regularizers
import keras.backend as K
from keras import regularizers
from tensorflow.keras.layers import InputLayer
from keras.layers import Input
import time
import tensorflow as tf
import os
import scipy
import h5py
import glob, os
# import BaseLineModel
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Concatenate
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import f1_score,plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve,roc_curve,roc_auc_score,auc
from sklearn.metrics import precision_recall_fscore_support,precision_recall_curve
import matplotlib.pyplot as plt;
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D, GlobalAveragePooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from keras import regularizers
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import InputLayer
from keras.layers import Input
# from google.colab import drive
from sklearn.model_selection import LeaveOneOut
import gc
gc.collect()
# drive.mount('/content/drive')

def PatientsName():
    Name=['chb01','chb02','chb03','chb04','chb05','chb06','chb07','chb08','chb09','chb10',
    'chb11','chb12','chb13','chb14','chb15','chb16','chb17','chb18','chb19','chb20','chb21',
    'chb22','chb23','chb24']

    return Name

def PatientsEDFFile(dirname):

    os.chdir(dirname)
    a=[]
    X=[]
    Y=[]
    k=0
    for file in glob.glob("*.mat"):

        a.append(file)
        # print(a)

    return a


def TestDataLoadCNN(model,x_test,y_test,threshold):

   # classes=np.unique(y_test)
  y_pred=model.predict(x_test)
  pred_prob = model.predict_proba(x_test)
  predicted_class=np.zeros(y_pred.shape)
  y_test_thresh=np.zeros(y_test.shape)

  predicted_class[y_pred>threshold]=1
  y_test_thresh[y_test>threshold]=1

    ## The output of the predictor sometimes has an extra dimention of size 1
    ## np.squeeze removes this extra empty dimention
  predicted_class = np.squeeze(predicted_class)
  y_test_thresh = np.squeeze(y_test_thresh)
  pred_prob = np.squeeze(pred_prob)

  return x_test,y_test,y_pred,y_test_thresh,predicted_class,pred_prob


def TestDataLoadmerge(model,x_test,y_test,threshold):

   # classes=np.unique(y_test)
  y_pred=model.predict(x_test)
  pred_prob = model.predict_on_batch(x_test)
  predicted_class=np.zeros(y_pred.shape)
  y_test_thresh=np.zeros(y_test.shape)

  predicted_class[y_pred>threshold]=1
  y_test_thresh[y_test>threshold]=1

    ## The output of the predictor sometimes has an extra dimention of size 1
    ## np.squeeze removes this extra empty dimention
  predicted_class = np.squeeze(predicted_class)
  y_test_thresh = np.squeeze(y_test_thresh)
  pred_prob = np.squeeze(pred_prob)

  return x_test,y_test,y_pred,y_test_thresh,predicted_class,pred_prob

def  Conv_BN_Act_Pool(filtNo,filtsize1,filtsize2,input1,activation,PoolSize):
    conv1 = Conv1D(filtNo,filtsize1)(input1)
    conv2 = Conv1D(filtNo, filtsize2)(conv1)
    BN=BatchNormalization(axis=-1)(conv2)
    ActFunc=Activation(activation)(BN)
    pool1=MaxPooling1D(pool_size=PoolSize)(ActFunc)

    return pool1
    # model = Model(inputs = input1, outputs = pool1)

def define_model():
    vectorsize=18
    input_shape=(1024,18)
    denseSize=8
    activation='relu'

    filtsize1=22
    filtNo1=8
    filtsize2=10
    filtNo2=16

    PoolSize=2

    input1 = Input(input_shape)

    model1=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,input1,activation,PoolSize)
    model2=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model1,activation,PoolSize)
    model3=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model2,activation,PoolSize)
    model4=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model3,activation,PoolSize)
    model5=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model4,activation,PoolSize)


    conv6=Conv1D(filtNo1,1)(model5)
    drop1=Dropout(0.25)(conv6)
    flat=Flatten()(drop1)

# Fully connected layer

    dense=Dense(denseSize)(flat)
##################################################################
    dim_data =int(vectorsize*(vectorsize+1)/2)
    vector_input = Input((dim_data,))

    # Concatenate the convolutional features and the vector input
    concat_layer= Concatenate()([vector_input, flat])
    denseout = Dense(100, activation='relu')(concat_layer)
    denseout = Dense(50, activation='relu')(denseout)
    output = Dense(1, activation='sigmoid')(denseout)

    # define a model with a list of two inputs
    model = Model(inputs=[input1, vector_input], outputs=output)


    model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['accuracy'])

    return model

model=define_model()
model.summary()

def PatientsName():
    Name=['chb01','chb02','chb03','chb04','chb05','chb06','chb07','chb08','chb09','chb10',
    'chb11','chb12','chb13','chb14','chb15','chb16','chb17','chb18','chb19','chb20','chb21',
    'chb22','chb23','chb24']

    return Name

def PatientsEDFFile(dirname):

    os.chdir(dirname)
    a=[]
    X=[]
    Y=[]
    k=0
    for file in glob.glob("*.mat"):

        a.append(file)
        # print(a)

    return a

def ReadMatFiles(dirname,indx):

  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]
  ind=[]
  Xfile=[]
  Yfile=[]
  ind=[]

  MI2=[]
  X=[]
  Y=[]

  for j in list(indx):
    print(j)
    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
    ind.append(indices)

  ind=np.concatenate(ind,axis=0)

  for k in range(len(ind)):
    # print(ind[k])
    matfile=loadmat(os.path.join(EDF[int(ind[k])]))
    x=matfile['X_4sec']
    y=matfile['Y_label_4sec']
    mi=matfile['estimated_MI']
    MI=np.zeros((mi.shape[0],153))
    for j in range(mi.shape[0]):

      mi2=mi[j,:,:]
      mi_mod=list(mi2[np.triu_indices(18,k=1)])
      MI[j,:]=mi_mod

    y=np.transpose(y)
    X.append(x)
    Y.append(y)
    MI2.append(list(MI))


  X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI2=np.concatenate(MI2,axis=0)


  return X,Y,MI2

# version 1: filters = 5 kernelsize = 9
## Version2
#     filter1=8
#     filter2=16

#     kernelsize1=7
#     kernelsize2=9
## Version3
#     filter1=8
#     filter2=16

#     kernelsize1=10
#     kernelsize2=22

## Version4
#     filter1=8 ---> at the first CNN and only two dense
#     filter2=16

#     kernelsize1=10
def ReadMatFilesDiff(dirname,indx):
  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]
  ind=[]
  Xfile=[]
  Yfile=[]
  ind=[]

  MI2=[]
  X=[]
  Y=[]
  for j in list(indx):

    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
    ind.append(indices)

  ind=np.concatenate(ind,axis=0)

  for k in range(len(ind)):

    matfile=loadmat(os.path.join(dirname,EDF[int(ind[k])]))
    x=matfile['X_4sec']
    y=matfile['Y_label_4sec']
    mi=matfile['estimated_MI']
    MI=np.zeros((mi.shape[0],153))
    for j in range(mi.shape[0]):

      mi2=mi[j,:,:]
      mi_mod=list(mi2[np.triu_indices(18,k=1)])
      MI[j,:]=mi_mod
    # MI=np.concatenate(MI,axis=0)
    y=np.transpose(y)
    X.append(x)
    Y.append(y)
    MI2.append(list(MI))


  X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI2=np.concatenate(MI2,axis=0)

  MI_diff=np.zeros((MI2.shape[0]-1,MI2.shape[1]))

  for j in range(MI2.shape[0]-1):
    MI_diff[j,:]=MI2[j+1,:]-MI2[j,:]

  return X[1:,:,:],Y[1:,:],MI2[1:,:],MI_diff


def ReadMatFilesSMILEonly(dirname1,dirname2,indx):

  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname1)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]
  ind=[]
  MI=[]
  X=[]
  Y=[]
  for j in list(indx):
    print(j)
    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
    ind.append(indices)
  ind=np.concatenate(ind,axis=0)
  for k in range(len(ind)):
    # print(ind[k])
    matfile=loadmat(os.path.join(dirname1,EDF[int(ind[k])]))
    matfile2=loadmat(os.path.join(dirname2,EDF[int(ind[k])]))
    ylabel=matfile2['Y_label']
    x=matfile['X_4sec']
    y=matfile['Y_label_4sec']
    mi=matfile['estimated_MI']
    for j in range(mi.shape[0]):
      mi2=mi[j,:,:]
      mi_mod=list(mi2[np.triu_indices(18)])
      MI.append(mi_mod)
    # y=np.transpose(y)
    ylabel=np.transpose(ylabel)
    # X.append(x)
    Y.append(ylabel)
  # X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI=np.array(MI)
  MI=MI.reshape(MI.shape[0],MI.shape[1],1)
  return MI,Y

def ReadMatFilesSMILEDiffonly(dirname1,dirname2,indx):

  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname1)

  Name=PatientsName()
  Xfile=[]
  Yfile=[]
  ind=[]
  MI=[]
  X=[]
  Y=[]

  for j in list(indx):
    print(j)
    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
    ind.append(indices)

  ind=np.concatenate(ind,axis=0)

  for k in range(len(ind)):
    # print(ind[k])
    matfile=loadmat(os.path.join(dirname1,EDF[int(ind[k])]))
    matfile2=loadmat(os.path.join(dirname2,EDF[int(ind[k])]))
    ylabel=matfile2['Y_label']
    x=matfile['X_4sec']
    y=matfile['Y_label_4sec']
    mi=matfile['estimated_MI']

    for j in range(mi.shape[0]):

      mi2=mi[j,:,:]
      mi_mod=list(mi2[np.triu_indices(18,k=1)])
      MI.append(mi_mod)

    # y=np.transpose(y)
    ylabel=np.transpose(ylabel)
    # X.append(x)
    Y.append(ylabel)

  # X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI=np.array(MI)

  MI_diff=np.zeros((MI.shape[0]-1,MI.shape[1]))

  for j in range(MI.shape[0]-1):

    MI_diff[j,:]=MI[j+1,:]-MI[j,:]

  MI=MI.reshape(MI.shape[0],MI.shape[1],1)
  MI_diff=MI_diff.reshape(MI_diff.shape[0],MI_diff.shape[1],1)

  return MI[1:,:,:],Y[1:,:],MI_diff


def Calculation(y_test_thresh, predicted_class,pred_prob):

    precision, recall, fscore, _ = precision_recall_fscore_support(y_test_thresh, predicted_class, average='weighted')
    fpr, tpr, _ = roc_curve(y_test_thresh, pred_prob)

    precision, recall, _ = precision_recall_curve(y_test_thresh, pred_prob)
    PR=auc(recall, precision)
    ROC=roc_auc_score(y_test_thresh, pred_prob)

    return fpr,tpr,PR,ROC

dirname='/home/baharsalafian/6FoldCrossSMILE'
dirname2='/home/baharsalafian/SMILELabel'
SaveResults='/home/baharsalafian/EyalRes'
modeldir='/home/baharsalafian/EyalRes'




pred=[]
act=[]

fold_no=1
batchsize=128
epoch=10
start_time = time.time()
FoldNum=6
model='CNN6fold10timesweighted'
savename='cnfmat_cnn10timesweightedtrain'
threshold=0.2
kfold = KFold(n_splits=FoldNum, shuffle=False)
for th in [0.3,0.4,0.5,0.6,0.7]:
  fold_no=1
  for trainindx, testindx in kfold.split(range(24)):
    predicted_targets = np.array([])
    actual_targets = np.array([])
    X_test,Y_test,mi_test= ReadMatFiles(dirname,trainindx)
    ModelName1=model+ str(fold_no)+'.h5'
    model1=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))
    ytest = (Y_test > threshold).astype(int)
    ypred1=model1.predict(X_test)
    ypred = (ypred1 > th).astype(int)

    pred=np.append(predicted_targets, ypred)
    act=np.append(actual_targets, ytest)
    fold_no=fold_no+1
  cnf_matrix = confusion_matrix(act, pred)
  np.save(os.path.join(SaveResults, savename+str(th)),  cnf_matrix)




print("--- %s seconds ---" % (time.time() - start_time))

# np.save(os.path.join(SaveResults,'accuracyCNNSMILELOO
# np.save(os.path.join(SaveResults,'accuracyCNNSMILELOO'),score)
# np.save(os.path.join(SaveResults,'lossCNNSMILELOO'),loss1)

# print(EDFFiles[indices[0]])

#   print(indices)
#   test.append(test_index)

# # print(test[2])

# for z in test:
#   indices=ReadMatFiles(dirname,z)
#   print(EDFFiles)
