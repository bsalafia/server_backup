# -*- coding: utf-8 -*-
"""Copy of CNNSMILEGRU_Bs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Fgyv5Og_PPfexz3ih-4uEJqa-PfQsKSA
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import tensorflow as tf
import scipy
import h5py
import glob, os
from scipy.io import loadmat
import math
from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# from keras.datasets import mnist
from tensorflow.keras.regularizers import l2, l1_l2
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,TimeDistributed, GRU,Concatenate
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, GlobalAveragePooling1D,MaxPooling1D,AveragePooling1D
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
from keras.callbacks import LearningRateScheduler
from sklearn import preprocessing
# from keras import regularizers
# from regularizers import l1_l2
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import InputLayer
from keras.layers import Input
import time
import gc
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(zca_whitening=True)
# from google.colab import drive
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

# def recall(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall_keras = true_positives / (possible_positives + K.epsilon())
#     return recall_keras


# def precision(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision_keras = true_positives / (predicted_positives + K.epsilon())
#     return precision_keras
def precision(y_true, y_pred, threshold_shift=0):

    # just in case
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))

    precision = tp / (tp + fp)
    return precision


def recall(y_true, y_pred, threshold_shift=0):

    # just in case
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)))

    recall = tp / (tp + fn)
    return recall


def fbeta(y_true, y_pred, beta = 2, threshold_shift=0):
    # just in case
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall)

def f1_score_met(y_true, y_pred,threshold_shift=0):

    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision1 = tp / (tp + fp)
    recall1 = tp / (tp + fn)

    p = precision1
    r = recall1
    return 2 * ((p * r) / (p + r))

def  Conv_BN_Act_Pool(filtNo,filtsize1,filtsize2,input1,activation,PoolSize,l2_size,drop_size):

  conv1 = Conv1D(filtNo,filtsize1,kernel_regularizer=l2(l2_size))(input1)
  conv2 = Conv1D(filtNo, filtsize2,kernel_regularizer=l2(l2_size))(conv1)
  BN=BatchNormalization(axis=-1)(conv2)
  ActFunc=Activation(activation)(BN)
  pool1=MaxPooling1D(pool_size=PoolSize)(ActFunc)
  # out=Dropout(drop_size)(pool1)

  return pool1

def define_SMILE(l2_size,drop_size):


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
  model1=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,input1,activation,PoolSize,l2_size,drop_size)
  model2=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model1,activation,PoolSize,l2_size,drop_size)
  model3=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model2,activation,PoolSize,l2_size,drop_size)
  model4=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model3,activation,PoolSize,l2_size,drop_size)
  model5=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model4,activation,PoolSize,l2_size,drop_size)
  conv6=Conv1D(filtNo1,1)(model5)
  drop1=Dropout(0.25)(conv6)

  flat=Flatten()(drop1)
  dense=Dense(denseSize)(flat)
################################################################
  dim_data =int(vectorsize*(vectorsize+1)/2)-18
  vector_input = Input((dim_data,))
  # Concatenate the convolutional features and the vector input
  concat_layer= Concatenate()([flat,vector_input])
  denseout = Dense(100)(concat_layer)
  denseout2 = Dense(50)(denseout)
  drop2=Dropout(0.5)(denseout2)
  output = Dense(1, activation='sigmoid')(drop2)

  # define a model with a list of two inputs
  model = Model(inputs=[input1, vector_input], outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=["accuracy", tf.keras.metrics.AUC(curve="ROC",name='auc_roc'),tf.keras.metrics.AUC(curve="PR",name='auc_pr'),tf.keras.metrics.Precision(name="precision"),tf.keras.metrics.Recall(name="recall")])
  return model

def define_CNN(l2_size,drop_size):

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
  model1=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,input1,activation,PoolSize,l2_size,drop_size)
  model2=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model1,activation,PoolSize,l2_size,drop_size)
  model3=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model2,activation,PoolSize,l2_size,drop_size)
  model4=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model3,activation,PoolSize,l2_size,drop_size)
  model5=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model4,activation,PoolSize,l2_size,drop_size)
  conv6=Conv1D(filtNo1,1)(model5)
  drop1=Dropout(0.25)(conv6)
  flat=Flatten()(drop1)

  denseout = Dense(denseSize)(flat)
  denseout2 = Dense(denseSize)(denseout)
  drop2=Dropout(0.5)(denseout2)
  output = Dense(1, activation='sigmoid')(drop2)
  # define a model with a list of two inputs
  model = Model(inputs=input1, outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy', metrics=['accuracy'])
  return model

def define_2DCNN():


  model = Sequential()
  model.add(Conv2D(8, (1, 3), input_shape=(18,1024,1)))
  # print(input_shape)
  model.add(Conv2D(8,(2, 1)))
  model.add(BatchNormalization(axis=-1))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(1,2)))


  model.add(Conv2D(8,(1, 3)))
  model.add(Conv2D(8,(2, 1)))
  model.add(BatchNormalization(axis=-1))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))


  model.add(Conv2D(8,(1, 3)))
  model.add(Conv2D(8,(2, 1)))
  model.add(BatchNormalization(axis=-1))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(1,2)))


  model.add(Conv2D(16,(1, 3)))
  model.add(Conv2D(16,(2, 1)))
  model.add(BatchNormalization(axis=-1))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))


  model.add(Conv2D(16,(1, 3)))
  model.add(Conv2D(16,(2, 1)))
  model.add(BatchNormalization(axis=-1))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))


  model.add(Conv2D(8,(1, 1)))
  model.add(Dropout(0.25))
  model.add(Flatten())

  model.add(Dense(8))
  model.add(Dense(8))
  model.add(Dropout(0.5))

  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  model.compile(optimizer=Adam(learning_rate=0.00001),loss='binary_crossentropy', metrics=['accuracy'])

  return model

def CNN_Spectrogram(filtNo1,filtNo2,filtsize,input_shape,activation,PoolSize,denseSize):



  input1 = Input(input_shape)



  BN1=BatchNormalization(axis=-1)(input1)
  conv1 = Conv1D(filtNo1,filtsize)(BN1)
  ActFunc1=Activation(activation)(conv1)

  BN2=BatchNormalization(axis=-1)(ActFunc1)
  conv2 = Conv1D(filtNo1, filtsize)(BN2)
  ActFunc2=Activation(activation)(conv2)
  pool1=MaxPooling1D(pool_size=PoolSize)(ActFunc2)

  BN3=BatchNormalization(axis=-1)(pool1)
  conv3 = Conv1D(filtNo2, filtsize)(BN3)
  ActFunc3=Activation(activation)(conv3)
  pool3=AveragePooling1D(pool_size=PoolSize)(ActFunc3)


  drop=Dropout(0.5)(pool3)
  flat=Flatten()(drop)
  dense=Dense(denseSize)(flat)
  output = Dense(1, activation='sigmoid')(dense)
  model = Model(inputs=input1, outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy', metrics=["accuracy", tf.keras.metrics.AUC(curve="ROC",name='auc_roc'),tf.keras.metrics.AUC(curve="PR",name='auc_pr'),tf.keras.metrics.Precision(name="precision"),tf.keras.metrics.Recall(name="recall")])
  return model

def define_CNNSMILEDiff(l2_size,drop_size):

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
  model1=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,input1,activation,PoolSize,l2_size,drop_size)
  model2=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model1,activation,PoolSize,l2_size,drop_size)
  model3=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model2,activation,PoolSize,l2_size,drop_size)
  model4=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model3,activation,PoolSize,l2_size,drop_size)
  model5=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model4,activation,PoolSize,l2_size,drop_size)
  conv6=Conv1D(filtNo1,1)(model5)
  drop1=Dropout(0.25)(conv6)
  flat=Flatten()(drop1)
# lly connected layer
  dense=Dense(denseSize)(flat)
################################################################
  dim_data =int(vectorsize*(vectorsize+1)/2)-18
  vector_input1 = Input((dim_data,))
  vector_input2 = Input((dim_data,))
  # Concatenate the convolutional features and the vector input
  concat_layer= Concatenate()([flat,vector_input1,vector_input2])
  denseout1 = Dense(100, activation='relu')(concat_layer)
  denseout2 = Dense(50, activation='relu')(denseout1)
  drop2=Dropout(0.5)(denseout2)
  output = Dense(1, activation='sigmoid')(drop2)
  # define a model with a list of two inputs
  model = Model(inputs=[input1, vector_input1,vector_input2], outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy', metrics=['accuracy'])
  return model

def define_CNNSpecSMILE(l2_size,drop_size):


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
  input2=Input(input_shape)
  model1=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,input1,activation,PoolSize,l2_size,drop_size)
  model2=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model1,activation,PoolSize,l2_size,drop_size)
  model3=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model2,activation,PoolSize,l2_size,drop_size)
  model4=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model3,activation,PoolSize,l2_size,drop_size)
  model5=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model4,activation,PoolSize,l2_size,drop_size)
  conv6=Conv1D(filtNo1,1)(model5)
  drop1=Dropout(0.25)(conv6)

  flat=Flatten()(drop1)
  dense=Dense(denseSize)(flat)
################################################################
  dim_data =int(vectorsize*(vectorsize+1)/2)-18
  vector_input = Input((dim_data,))
  flat2=Flatten()(input2)
  # Concatenate the convolutional features and the vector input
  concat_layer= Concatenate()([flat,flat2,vector_input])
  denseout = Dense(100)(concat_layer)
  denseout = Dense(50)(denseout)
  output = Dense(1, activation='sigmoid')(denseout)

  # define a model with a list of two inputs
  model = Model(inputs=[input1,input2, vector_input], outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=["accuracy", tf.keras.metrics.AUC(curve="ROC",name='auc_roc'),tf.keras.metrics.AUC(curve="PR",name='auc_pr'),tf.keras.metrics.Precision(name="precision"),tf.keras.metrics.Recall(name="recall")])
  return model

def define_model_CNNGRU(l2_size,drop_size):


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
  model1=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,input1,activation,PoolSize,l2_size,drop_size)
  model2=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model1,activation,PoolSize,l2_size,drop_size)
  model3=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model2,activation,PoolSize,l2_size,drop_size)
  model4=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model3,activation,PoolSize,l2_size,drop_size)
  model5=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model4,activation,PoolSize,l2_size,drop_size)
  conv6=Conv1D(filtNo1,1)(model5)
  drop1=Dropout(0.25)(conv6)
  flat=Flatten()(drop1)
  cnn=Model(inputs=input1,outputs=flat)
  encoded_frames = TimeDistributed(cnn)(inputGRU)
  encoded_sequence = Bidirectional(GRU(50, return_sequences=True))(encoded_frames)
  output=TimeDistributed(Dense(1,activation='sigmoid'))(encoded_sequence)

  model = Model(inputs=inputGRU, outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.00001),loss='binary_crossentropy', metrics=['accuracy'])
  return model


def create_sub_seq(nn_input, len_ss, labels=None):

  """
  This function creates all sub sequences for the batch
  """
  n_seq = nn_input.shape[0]
  len_seq = nn_input.shape[1]
  n_ss = len_seq - len_ss + 1
  new_labels = []
  if nn_input.ndim == 3:
    new_inp = np.zeros((n_ss*n_seq,len_ss,nn_input.shape[2]))
  elif nn_input.ndim == 4:
    new_inp = np.zeros((n_ss*n_seq,len_ss,nn_input.shape[2], nn_input.shape[3]))
  if labels is not None:
      dim_labels = labels.shape
      if len(dim_labels) == 2:
          new_labels = np.zeros((n_ss*n_seq, len_ss))
      elif len(dim_labels) == 3:
          new_labels = np.zeros((n_ss * n_seq, len_ss, dim_labels[2]))
  k = 0
  for i in range(n_seq):
      for j in range(n_ss):
          new_inp[k] = nn_input[i, j:j + len_ss, :]
          if labels is not None:
              if len(dim_labels) == 2:
                  new_labels[k, :] = labels[i, j:j + len_ss]
              elif len(dim_labels) == 3:
                  new_labels[k, :, :] = labels[i, j:j + len_ss, :]
          k += 1
  return new_inp, n_ss, new_labels




def SelectIndx(EDFNo,ind):

  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
  Name=PatientsName()

  indx=[]
  for j in range(len(ind)):
    # print(j)
    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
    indx.append(indices)

  indtest=[]
  indtrain=[]
  for i in range(len(indx)):

    for k in range(len(indx[i])):
      # print(len(indx[i]))

      if k==EDFNo:
        indtest.append(indx[i][k])
        # print(indtest)

      else:
        indtrain.append(indx[i][k])
        # print(indtrain)

  # indtest=np.concatenate(indtest,axis=0)
  # indtrain=np.concatenate(indtrain,axis=0)
  # print(len(indtest))
  return indtest,indtrain




def ReadMatFiles(dirname,dirname2,ind, seq_len=1,diff=None):

  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]
  spec=[]

  MI_all=[]
  X=[]
  Y=[]
  MI_diff_all=[]
  # print(ind)
  for k in range(len(ind)):

    print(EDF[ind[k]])
    matfile=loadmat(os.path.join(dirname,EDF[ind[k]]))

    Name2=EDF[ind[k]].split('.')
    # matfile2=loadmat(os.path.join(dirname2,Name2[0]+'_Spectogram.mat'))
    matfile2=loadmat(os.path.join(dirname2,EDF[ind[k]]))
    Spectrogram=matfile2['spectogram']
    x=matfile['X_4sec']
    y=matfile['Y_label_4sec']
    mi=matfile['estimated_MI']
    y=np.transpose(y)

    start_idx = np.argmax(y>0)
    a = y == 1
    end_idx = len(a) - np.argmax(np.flip(a)) - 1
    real_y = np.zeros_like(y)
    real_y[start_idx:end_idx+1] = 1
    MI=np.zeros((mi.shape[0],153))
    for j in range(mi.shape[0]):
      mi2=mi[j,:,:]
      mi_mod=list(mi2[np.triu_indices(18,k=1)])
      MI[j,:]=mi_mod


    MI_diff=[]
    if seq_len > 1:
      real_y = np.expand_dims(real_y, axis=0)
      x = np.expand_dims(x, axis=0)
      MI = np.expand_dims(MI, axis=0)
      # print(MI.shape)
      x, _ , real_y = create_sub_seq(x, seq_len, labels=real_y)
      MI, _, _ = create_sub_seq(MI, seq_len)
      # print(x.shape)
      # print(real_y.shape)
      # print(MI.shape)

    if diff is not None:

      for j in range(MI.shape[0]-1):

        MI_diff.append(MI[j+1]-MI[j])

      MI_diff=np.array(MI_diff)
      MI=MI[1:]
      x=x[1:]
      real_y=real_y[1:]
      Spectrogram=Spectrogram[1:]
    X.append(x)
    Y.append(real_y)
    MI_all.append(MI)
    MI_diff_all.append(MI_diff)
    spec.append(Spectrogram)


  X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  spec=np.concatenate(spec,axis=0)

  MI_all=np.concatenate(MI_all,axis=0)
  MI_diff_all=np.concatenate(MI_diff_all,axis=0)
  print(Y)
  print(X.shape)
  print(Y.shape)
  print(MI_all.shape)
  print(MI_diff_all.shape)

  return X, Y, MI_all, MI_diff_all,spec

def MeanStdVar(mylist):

  ListMean=np.mean(mylist,axis=0)
  ListStd=np.std(mylist)
  ListVar=np.var(mylist)

  return ListMean,ListStd,ListVar

def ModelTrain(dirname,dirname2,SaveResults,SaveHisResults,modelname,seq_len,cnn,smile,diff,twodcnn,spectrogram,cnnsmilespec,gru,epoch,epochLoad,l2_size,drop_size,batchSize):

  loss=[]
  loss_val=[]
  acc=[]
  acc_val=[]
  LR=[]

  f1=[]
  f1_val=[]

  f1beta=[]
  f1beta_val=[]

  ROC=[]
  ROC_val=[]

  PR=[]
  PR_val=[]

  PR_m=[]
  PR_m_val=[]

  recall=[]
  recall_val=[]

  FoldNum=6
  kfold = KFold(n_splits=FoldNum, shuffle=False)
  start_time = time.time()
  fold_no=1
  indx=range(0,24)

  for i in range(2):

    batchsize=batchSize

    testindx,trainindx=SelectIndx(i,indx)
    # print(range(len(testindx)))
    # for k in range(len(trainindx)):
    #
    #   print(k)
    x, y, mi,mi_diff,spec_train=ReadMatFiles(dirname,dirname2,trainindx,seq_len,diff)
    # time.sleep(2)
    xtest, ytest, mitest,mitest_diff,spec_test=ReadMatFiles(dirname,dirname2,testindx,seq_len,diff)
#
    if cnn==1:

      tf.keras.backend.clear_session()
      model=define_CNN(l2_size,drop_size)
      X_train=x
      X_test=xtest


    if smile==1:
      tf.keras.backend.clear_session()
      model=define_SMILE(l2_size,drop_size)
      X_train=[x,mi]
      X_test=[xtest,mitest]


    if diff==1:

      tf.keras.backend.clear_session()
      model=define_CNNSMILEDiff(l2_size,drop_size)
      X_train=[x,mi,mi_diff]
      X_test=[xtest,mitest,mitest_diff]


    if twodcnn==1:

      tf.keras.backend.clear_session()
      model=define_2DCNN()
      x=x.reshape(x.shape[0],x.shape[2],x.shape[1],1)
      xtest=xtest.reshape(xtest.shape[0],xtest.shape[2],xtest.shape[1],1)
      X_train=x
      X_test=xtest

#
    if spectrogram==1:

      tf.keras.backend.clear_session()
      model=CNN_Spectrogram(filtNo1=120,filtNo2=160,filtsize=5,input_shape=(1024,18),activation='relu',PoolSize=3,denseSize=8)

      X_train=spec_train
      X_test=spec_test

    if cnnsmilespec==1:

      tf.keras.backend.clear_session()
      model=define_CNNSpecSMILE(l2_size,drop_size)

      X_train=[x,spec_train,mi]
      X_test=[xtest,spec_test,mitest]



    ##################################################################
    from sklearn.utils import class_weight
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y),y=np.ravel(y))
    print(cw)
    class_weights = {0:cw[0], 1: cw[1]}
######################################
    # initial_learning_rate = 0.00001
    def lr_exp_decay(epoch, lr):
      # k = math.sqrt(2)

      k=2

      if lr>0.00001:
        return lr / k

      else:
        return lr



########################################
    if gru==1:

      tf.keras.backend.clear_session()
      model=define_model_CNNGRU(l2_size,drop_size)
      X_train=x
      X_test=xtest

      from sklearn.utils import class_weight
      cw = class_weight.compute_class_weight('balanced', np.unique(y),np.ravel(y))
      sample_weight = np.zeros_like(y)
      sample_weight[y == 0] = cw[0]
      sample_weight[y == 1] = cw[1]



    if gru==1:
      history=model.fit(X_train, y, validation_data=(X_test,ytest), batch_size=batchsize, epochs=epoch, sample_weight=sample_weight, callbacks=[LearningRateScheduler(lr_exp_decay, verbose=2)])
    else:

      history=model.fit(X_train, y, validation_data=(X_test,ytest), batch_size=batchsize, epochs=epoch, class_weight=class_weights, callbacks=[LearningRateScheduler(lr_exp_decay, verbose=2)])



    model.save(SaveResults+'/'+modelname+'_fold'+str(fold_no)+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.h5')
    print(history.history.keys())
    loss.append(history.history['loss'])
    loss_val.append(history.history['val_loss'])

    acc.append(history.history['accuracy'])
    acc_val.append(history.history['val_accuracy'])
    LR.append(history.history['lr'])


    #
    # f1beta.append(history.history['fbeta'])
    # f1beta_val.append(history.history['val_fbeta'])

    PR.append(history.history['auc_pr'])
    PR_val.append(history.history['val_auc_pr'])

    PR_m.append(history.history['precision'])
    PR_m_val.append(history.history['val_precision'])

    recall.append(history.history['recall'])
    recall_val.append(history.history['val_recall'])

    p=np.array(history.history['precision'])
    print("len of precision",len(p))
    r=np.array(history.history['recall'])
    p_val=np.array(history.history['val_precision'])
    r_val=np.array(history.history['val_recall'])



    f1.append(2 * ((p * r) / (p + r)))
    f1_val.append(2 * ((p_val * r_val) / (p_val + r_val)))

    ROC.append(history.history['auc_roc'])
    ROC_val.append(history.history['val_auc_roc'])
    fold_no=fold_no+1
    tf.keras.backend.clear_session()
#############################################
  loss_mean,loss_std,_=MeanStdVar(loss)
  loss_val_mean,loss_val_std,_=MeanStdVar(loss_val)
  acc_mean,acc_std,_=MeanStdVar(acc)
  acc_val_mean,acc_val_std,_=MeanStdVar(acc_val)

  LR_mean,LR_std,_=MeanStdVar(LR)

#############################################
  f1_mean,f1_std,_=MeanStdVar(f1)
  f1_val_mean,f1_val_std,_=MeanStdVar(f1_val)

  # f1beta_mean,f1beta_std,_=MeanStdVar(f1beta)
  # f1beta_val_mean,f1beta_val_std,_=MeanStdVar(f1beta_val)
  #
  PR_mean,PR_std,_=MeanStdVar(PR)
  PR_val_mean,PR_val_std,_=MeanStdVar(PR_val)

  PR_m_mean,PR_m_std,_=MeanStdVar(PR_m)
  PR_m_val_mean,PR_m_val_std,_=MeanStdVar(PR_m_val)

  ROC_mean,ROC_std,_=MeanStdVar(ROC)
  ROC_val_mean,ROC_val_std,_=MeanStdVar(ROC_val)

  recall_mean,recall_std,_=MeanStdVar(recall)
  recall_val_mean,recall_val_std,_=MeanStdVar(recall_val)

  np.savez(os.path.join(SaveHisResults, 'HistoryRes_'+modelname+'_epoch_'+str(epoch+epochLoad)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)), loss=loss, loss_val=loss_val, accuracy=acc, accuracy_val=acc_val, LR=LR, PR=PR, PR_val=PR_val,ROC=ROC, ROC_val=ROC_val, f1=f1, f1_val=f1_val)

  plt.plot(acc_mean)
  plt.plot(acc_val_mean)
  plt.title(modelname+'_accuracy'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize))
  plt.legend(['train', 'test'], loc='upper left')

  plt.fill_between(range(epoch), acc_mean-acc_std, acc_mean+acc_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), acc_val_mean-acc_val_std, acc_val_mean+acc_val_std, color='orange', alpha = 0.5)

  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_accuracy'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()

  plt.plot(loss_mean)
  plt.plot(loss_val_mean)
  plt.title(modelname+'_loss'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize))
  plt.legend(['train', 'test'], loc='upper left')

  plt.fill_between(range(epoch), loss_mean-loss_std, loss_mean+loss_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), loss_val_mean-loss_val_std, loss_val_mean+loss_val_std, color='orange', alpha = 0.5)

  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_loss'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()
  ###############3
  plt.plot(f1_mean)
  plt.plot(f1_val_mean)
  plt.title(modelname+'_F1 score'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize))
  plt.legend(['train', 'test'], loc='upper left')

  plt.fill_between(range(epoch), f1_mean-f1_std, f1_mean+f1_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), f1_val_mean-f1_val_std, f1_val_mean+f1_val_std, color='orange', alpha = 0.5)

  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_F1 score'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()
 #############
  # plt.plot(f1beta_mean)
  # plt.plot(f1beta_val_mean)
  # plt.title(modelname+'_Fbeta'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize))
  # plt.legend(['train', 'test'], loc='upper left')
  #
  # plt.fill_between(range(epoch), f1beta_mean-f1beta_std, f1beta_mean+f1beta_std, color='blue', alpha = 0.5)
  # plt.fill_between(range(epoch), f1beta_val_mean-f1beta_val_std, f1beta_val_mean+f1beta_val_std, color='orange', alpha = 0.5)
  #
  # plt.ylabel('%')
  # plt.xlabel('epoch')
  # plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_Fbeta'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
  # plt.clf()
#####
  plt.plot(PR_m_mean)
  plt.plot(PR_m_val_mean)
  plt.title(modelname+'_Precision'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize))
  plt.legend(['train', 'test'], loc='upper left')

  plt.fill_between(range(epoch), PR_m_mean-PR_m_std, PR_m_mean+PR_m_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), PR_m_val_mean-PR_m_val_std, PR_m_val_mean+PR_m_val_std, color='orange', alpha = 0.5)

  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_Precision'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()
#################3

  plt.plot(recall_mean)
  plt.plot(recall_val_mean)
  plt.title(modelname+'_Recall'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize))
  plt.legend(['train', 'test'], loc='upper left')

  plt.fill_between(range(epoch), recall_mean-recall_std, recall_mean+recall_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), recall_val_mean-recall_val_std, recall_val_mean+recall_val_std, color='orange', alpha = 0.5)

  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_Recall'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()
###############
  plt.plot(PR_mean)
  plt.plot(PR_val_mean)
  plt.title(modelname+'_PR'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize))
  plt.legend(['train', 'test'], loc='upper left')

  plt.fill_between(range(epoch), PR_mean-PR_std, PR_mean+PR_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), PR_val_mean-PR_val_std, PR_val_mean+PR_val_std, color='orange', alpha = 0.5)

  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_PR'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()
######
  plt.plot(ROC_mean)
  plt.plot(ROC_val_mean)
  plt.title(modelname+'_ROC'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize))
  plt.legend(['train', 'test'], loc='upper left')

  plt.fill_between(range(epoch), ROC_mean-ROC_std, ROC_mean+ROC_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), ROC_val_mean-ROC_val_std, ROC_val_mean+ROC_val_std, color='orange', alpha = 0.5)

  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_ROC'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()



  plt.plot(LR_mean)
  plt.title(modelname+'_LR'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize))
  plt.legend(['lr'], loc='upper left')
  plt.fill_between(range(epoch), LR_mean-LR_std, LR_mean+LR_std, color='orange', alpha = 0.5)

  plt.ylabel('lr')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_LR'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()

  print("--- %s seconds ---" % (time.time() - start_time))


i=0
k=None
j=256
dirname2='/media/datadrive/bsalafian/AllMatFiles'
dirname='/media/datadrive/bsalafian/6FoldCrossSMILE'
SaveResults='/home/baharsalafian/CNNSMILEGRUModels_JBHI_Metrics'
SaveHisResults='/home/baharsalafian/History_JBHI_LossTest_Keras_Metrics'

ModelTrain(dirname,dirname2,SaveResults,SaveHisResults,'CNNSMILESpectrogram10times_Decay.5_LR.001_24Testfiles_KerasMet',seq_len=1,cnn=0,smile=0,diff=0,twodcnn=0,spectrogram=0,cnnsmilespec=1,gru=0,epoch=1000,epochLoad=0,l2_size=k,drop_size=i,batchSize=j)
