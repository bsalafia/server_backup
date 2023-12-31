# -*- coding: utf-8 -*-
"""1DCNNServer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XIaeyIZmYN-0bMvxywlL5M5nTjaRBOJe
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
from keras import optimizers, regularizers
import keras.backend as K
from keras import regularizers
from tensorflow.keras.layers import InputLayer
from keras.layers import Input

import tensorflow as tf

import scipy
import h5py
import glob
# import BaseLineModel
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Concatenate

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
import time
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

def ReadMatFiles(dirname,indx):

  EDF=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]


  indices = [i for i, elem in enumerate(EDF) if Name[indx] in elem]

  return indices,EDF

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
    matfile=loadmat(os.path.join(EDF[int(ind[k])]))
    x=matfile['X_4sec']
    y=matfile['Y_label_4sec']
    mi=matfile['estimated_MI']

    for j in range(mi.shape[0]):

      mi2=mi[j,:,:]
      mi_mod=list(mi2[np.triu_indices(18)])
      MI.append(mi_mod)

    y=np.transpose(y)
    X.append(x)
    Y.append(y)

  X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI=np.array(MI)

  return X,Y,MI

def Data(dirname):
  X=[]
  Y=[]
  MI=[]
  EDF=PatientsEDFFile(dirname)

  for i in range(len(EDF)):

    matfile=loadmat(os.path.join(dirname, EDF[i]))
    x=matfile['X_4sec']
    y=matfile['Y_label_4sec']
    mi=matfile['estimated_MI']

    for j in range(mi.shape[0]):
      mi2=mi[j,:,:]
      mi_mod=list(mi2[np.triu_indices(18)])
      MI.append(mi_mod)

    y=np.transpose(y)
    X.append(x)
    Y.append(y)


  X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI=np.array(MI)

  return X,Y,MI

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
#     kernelsize2=22

## Version5
#     filter1=8 ---> at the first CNN and only two dense
#     filter2=16

#     kernelsize1=22
#     kernelsize2=10

def define_model2():


    model = Sequential()

    filter1=8
    filter2=16

    kernelsize1=22
    kernelsize2=10

    model.add(Conv1D(filter1, kernelsize2, input_shape=(1024,18)))
    model.add(Conv1D(filter1, kernelsize1))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filter1, kernelsize2))
    model.add(Conv1D(filter1, kernelsize1))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filter1, kernelsize2))
    model.add(Conv1D(filter1, kernelsize1))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filter2, kernelsize2))
    model.add(Conv1D(filter2, kernelsize1))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filter2, kernelsize2))
    model.add(Conv1D(filter2, kernelsize1))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filter1, 1))

    model.add(Dropout(0.25))
    model.add(Flatten())

# Fully connected layer

    model.add(Dense(8))
    model.add(Dense(8))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['accuracy'])

    return model

dirname='/home/baharsalafian/SMILEData'
SaveResults='/home/baharsalafian/CNNResults'
X=[0,1,2,4,5,6,7,8,9,10,12,13,16,17,18,19,20,21,22]
loo = LeaveOneOut()
test=[]
score=[]
loss1=[]
fold_no=1
batchsize=128
epoch=10
start_time = time.time()
# model=define_model2()
for trian_index, test_index in loo.split(X):


#   print(trian_index)
#   test.append(train_index)
# print(len(trian_index))
  X1= [X[index] for index in trian_index]
  X2= [X[index] for index in test_index]
  # X_train,Y_train,mi_train = ReadMatFiles(dirname,X1)

  X_test,Y_test,mi_test = ReadMatFiles(dirname,X2)

  ModelName='CNNLOO'+ str(fold_no)+'.h5'
  model=tf.keras.models.load_model(os.path.join(SaveResults,ModelName))
  # model.fit(X_train,Y_train,validation_split=0.2,batch_size=batchsize , epochs=epoch,verbose = 2)

  # X_train = None
  # Y_train = None
  # gc.collect()

  loss, acc = model.evaluate(X_test,Y_test, verbose=2)
  score.append(acc)
  loss1.append(loss)
  # model.save('/home/baharsalafian/CNNResults/CNNLOO'+ str(fold_no)+'.h5')
  fold_no=fold_no+1
  # np.save(os.path.join(SaveResults,'X_test_'+str(fold_no)),X_test)
  # np.save(os.path.join(SaveResults,'Y_test_'+str(fold_no)),Y_test)

  X_test = None
  Y_test = None
  gc.collect()
np.save(os.path.join(SaveResults,'accuracyCNNLOO'),score)
np.save(os.path.join(SaveResults,'lossCNNLOO'),loss1)
print("--- %s seconds ---" % (time.time() - start_time))

# np.save(os.path.join(SaveResults,'accuracyCNNLOO'),score)
# np.save(os.path.join(SaveResults,'lossCNNLOO'),loss1)

# print(EDFFiles[indices[0]])

#   print(indices)
#   test.append(test_index)

# # print(test[2])

# for z in test:
#   indices=ReadMatFiles(dirname,z)
#   print(EDFFiles)
