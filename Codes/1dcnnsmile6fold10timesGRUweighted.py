# -*- coding: utf-8 -*-
"""CNNNSMILEModels.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yz_BpmDOOkww1Un3aK-NFBEfii4L7Bba
"""

# Commented out IPython magic to ensure Python compatibility.
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import scipy
import h5py
import glob, os
from scipy.io import loadmat

from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,TimeDistributed, GRU,Concatenate
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, GlobalAveragePooling1D,MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from keras import regularizers
from numpy import mean
from numpy import std
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.datasets import cifar10
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D, GlobalAveragePooling1D,Bidirectional
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
import time
import gc
# from google.colab import drive
# drive.mount('/content/drive')

def  Conv_BN_Act_Pool(filtNo,filtsize1,filtsize2,input1,activation,PoolSize):
    conv1 = Conv1D(filtNo,filtsize1)(input1)
    conv2 = Conv1D(filtNo, filtsize2)(conv1)
    BN=BatchNormalization(axis=-1)(conv2)
    ActFunc=Activation(activation)(BN)
    pool1=MaxPooling1D(pool_size=PoolSize)(ActFunc)

    return pool1

def define_model_CNNGRU():

  memory=3
  vectorsize=18
  input_shape=(1024,18)
  input_shape_GRU=(memory,1024,18)
  denseSize=8
  activation='relu'
  filtsize1=22
  filtNo1=8
  filtsize2=10
  filtNo2=16
  PoolSize=2
  input1 = Input(input_shape)
  inputGRU=Input(input_shape_GRU)
  model1=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,input1,activation,PoolSize)
  model2=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model1,activation,PoolSize)
  model3=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model2,activation,PoolSize)
  model4=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model3,activation,PoolSize)
  model5=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model4,activation,PoolSize)
  conv6=Conv1D(filtNo1,1)(model5)
  drop1=Dropout(0.25)(conv6)
  flat=Flatten()(drop1)
  cnn=Model(inputs=input1,outputs=flat)
  encoded_frames = TimeDistributed(cnn)(inputGRU)
  encoded_sequence = Bidirectional(GRU(50, return_sequences=True))(encoded_frames)
  output=TimeDistributed(Dense(1,activation='sigmoid'))(encoded_sequence)

  model = Model(inputs=inputGRU, outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['accuracy'])
  return model


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

def define_model_CNNSMILEGRU():


  denseSize=8
  activation='relu'
  filtsize1=22
  filtNo1=8
  filtsize2=10
  filtNo2=16
  PoolSize=2
  ##################
  memory=3
  vectorsize=18
  input_shape=(1024,18)
  input_shape_GRU=(memory,1024,18)
  dim_data =int(vectorsize*(vectorsize+1)/2)-18
  input1 = Input(input_shape)
  input1GRU=Input(input_shape_GRU)
  input2GRU= Input((memory,dim_data))

  model1=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,input1,activation,PoolSize)
  model2=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model1,activation,PoolSize)
  model3=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model2,activation,PoolSize)
  model4=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model3,activation,PoolSize)
  model5=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model4,activation,PoolSize)
  conv6=Conv1D(filtNo1,1)(model5)
  drop1=Dropout(0.25)(conv6)
  flat=Flatten()(drop1)
  # dense=Dense(denseSize)(flat)
##################################################################

  vector_input = Input((dim_data,))
  # Concatenate the convolutional features and the vector input
  concat_layer= Concatenate()([flat,vector_input])
  # denseout = Dense(100, activation='relu')(concat_layer)
  # denseout = Dense(50, activation='relu')(denseout)
  # output = Dense(1, activation='sigmoid')(denseout)
  # define a model with a list of two inputs
  cnn = Model(inputs=[input1, vector_input], outputs=concat_layer)

  encoded_frames = TimeDistributed(cnn)([input1GRU,input2GRU])
  encoded_sequence = Bidirectional(GRU(50, return_sequences=True))(encoded_frames)
  output=TimeDistributed(Dense(1,activation='sigmoid'))(encoded_sequence)

  model = Model(inputs=[input1GRU,input2GRU], outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['accuracy'],sample_weight_mode="temporal")

  return model

def define_model_CNNSMILEDiffGRU():

  denseSize=8
  activation='relu'
  filtsize1=22
  filtNo1=8
  filtsize2=10
  filtNo2=16
  PoolSize=2
  ##################
  memory=3
  vectorsize=18
  input_shape=(1024,18)
  input_shape_GRU=(memory,1024,18)
  dim_data =int(vectorsize*(vectorsize+1)/2)-18
  input1 = Input(input_shape)
  input1GRU=Input(input_shape_GRU)
  input2GRU= Input((memory,dim_data,))
  input3GRU= Input((memory,dim_data,))

  model1=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,input1,activation,PoolSize)
  model2=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model1,activation,PoolSize)
  model3=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model2,activation,PoolSize)
  model4=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model3,activation,PoolSize)
  model5=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model4,activation,PoolSize)
  conv6=Conv1D(filtNo1,1)(model5)
  drop1=Dropout(0.25)(conv6)
  flat=Flatten()(drop1)
  # dense=Dense(denseSize)(flat)
##################################################################

  vector_input1 = Input((dim_data,))
  vector_input2=Input((dim_data,))
  # Concatenate the convolutional features and the vector input
  concat_layer= Concatenate()([flat,vector_input1,vector_input2])
  # denseout = Dense(100, activation='relu')(concat_layer)
  # denseout = Dense(50, activation='relu')(denseout)
  # output = Dense(1, activation='sigmoid')(denseout)
  # define a model with a list of two inputs
  cnn = Model(inputs=[input1, vector_input1,vector_input2], outputs=concat_layer)

  encoded_frames = TimeDistributed(cnn)([input1GRU,input2GRU,input3GRU])
  encoded_sequence = Bidirectional(GRU(50, return_sequences=True))(encoded_frames)
  output=TimeDistributed(Dense(1,activation='sigmoid'))(encoded_sequence)

  model = Model(inputs=[input1GRU,input2GRU,input3GRU], outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['accuracy'])

  return model

def ReadMatFiles(dirname,indx):


  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
  # print(EDF)
  Name=PatientsName()
  # print(Name)
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
    matfile=loadmat(os.path.join(dirname,EDF[int(ind[k])]))
    x=matfile['X_4sec']
    y=matfile['Y_label']
    mi=matfile['estimated_MI']

    # MI=np.concatenate(MI,axis=0)
    # y=np.transpose(y)
    X.append(x)
    Y.append(y)
    MI.append(mi)


  X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI=np.concatenate(MI,axis=0)


  return X,Y,MI

def ReadMatFilesDiff(dirname,indx):

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
  print(indx)
  for j in list(indx):
    print(j)
    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]

    ind.append(indices)

  ind=np.concatenate(ind,axis=0)

  for k in range(len(ind)):
    # print(ind[k])
    matfile=loadmat(os.path.join(dirname,EDF[int(ind[k])]))
    x=matfile['X_4sec']
    y=matfile['Y_label']
    mi=matfile['estimated_MI']



    # MI=np.concatenate(MI,axis=0)
    # y=np.transpose(y)
    X.append(x)
    Y.append(y)
    MI.append(mi)


  X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI=np.concatenate(MI,axis=0)

  MI_diff=np.zeros((MI.shape[0]-1,MI.shape[1],MI.shape[2]))

  for j in range(MI.shape[0]-1):
    MI_diff[j,:,:]=MI[j+1,:,:]-MI[j,:,:]




  return X[1:,:,:,:],Y[1:,:],MI[1:,:,:],MI_diff


dirname='/home/baharsalafian/GRUData10times'

SaveResults='/home/baharsalafian/CNNSMILEGRURes'

test=[]
score=[]
loss1=[]
fold_no=1
batchsize=128
epoch=10
start_time = time.time()
model=define_model_CNNSMILEGRU()
FoldNum=6
threshold=0.5
kfold = KFold(n_splits=FoldNum, shuffle=False)
for trainindx, testindx in kfold.split(range(24)):


#
  X_train,Y_train,mi_train = ReadMatFiles(dirname,trainindx)

  X_test,Y_test,mi_test= ReadMatFiles(dirname,testindx)
  ytest = (Y_test > threshold).astype(int)
  ytrain = (Y_train > threshold).astype(int)
  print(ytrain.shape)
  sample_weight = np.ones(shape=(ytrain.shape),)

  ytrain1=np.zeros((ytrain.shape[0],ytrain.shape[1],1))
  ytrain1[:,:,0]=ytrain
  sample_weight[ytrain == 1] = 20.0
  model.fit([X_train,mi_train],ytrain1,sample_weight=sample_weight,validation_split=0.2,batch_size=batchsize , epochs=epoch,verbose = 2)

  X_train = None
  Y_train = None
  gc.collect()

  loss, acc = model.evaluate([X_test,mi_test],ytest, verbose=2)
  score.append(acc)
  loss1.append(loss)
  model.save('/home/baharsalafian/CNNSMILEGRURes/CNNSMILEGRUweighted'+ str(fold_no)+'.h5')
  fold_no=fold_no+1
  # np.save(os.path.join(SaveResults,'X_test_'+str(fold_no)),X_test)
  # np.save(os.path.join(SaveResults,'Y_test_'+str(fold_no)),Y_test)

  X_test = None
  Y_test = None
  gc.collect()

np.save(os.path.join(SaveResults,'accuracyCNNSMILEGRUweighted'),score)
np.save(os.path.join(SaveResults,'lossCNNSMILEGRUweighted'),loss1)

print("--- %s seconds ---" % (time.time() - start_time))
