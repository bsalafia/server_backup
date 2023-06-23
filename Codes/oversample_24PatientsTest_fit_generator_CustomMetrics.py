# -*- coding: utf-8 -*-
"""Oversample_allmodels_fit_generator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-Tcan2IgxDGbfnSy3i8i8EfOUeQOHn1V
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
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
from sklearn.metrics import plot_precision_recall_curve,roc_curve,roc_auc_score,auc
from sklearn.metrics import precision_recall_fscore_support,precision_recall_curve

from tensorflow import keras
from tensorflow.keras.regularizers import l2, l1_l2
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,TimeDistributed, GRU,Concatenate
from tensorflow.keras.optimizers import Adam
from keras.layers import BatchNormalization
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
# from keras.optimizers import Adam
# from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D, GlobalAveragePooling1D,Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
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
# datagen = ImageDataGenerator(zca_whitening=True)
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

def recall2(y_true, y_pred):

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall_m = true_positives / (possible_positives + K.epsilon())
  return recall_m

def precision(y_true, y_pred):

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision_m = true_positives / (predicted_positives + K.epsilon())
  return precision_m

def AUCPR(y_true, y_pred):

  precision, recall, _ = precision_recall_curve(y_true, y_pred)

  PR1=auc(recall, precision)

  return PR1
# def f1_score(y_true, y_pred):
  # precision_m = precision(y_true, y_pred)
  # recall_m = recall(y_true, y_pred)
  # return 2*((precision_m*recall_m)/(precision_m+recall_m+K.epsilon()))
def f1_score(y_true, y_pred,threshold_shift=0):

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

def auc_roc(y_true, y_pred):

  return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

def auc_pr(y_true, y_pred):

  return tf.py_function(AUCPR, (y_true, y_pred), tf.double)
# def auc_pr(y_true, y_pred):
#   precision1, recall1, _ = precision_recall_curve(y_true, y_pred)

#   PR1=auc(recall1, precision1)
#   return PR1

METRICS = [
      # keras.metrics.TruePositives(name='tp'),
      # keras.metrics.FalsePositives(name='fp'),
      # keras.metrics.TrueNegatives(name='tn'),
      # keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc_roc', curve='ROC'),
      keras.metrics.AUC(name='auc_pr', curve='PR'), # precision-recall curve
]


def  Conv_BN_Act_Pool(filtNo,filtsize1,filtsize2,input1,activation,PoolSize,l2_size,drop_size):

  conv1 = Conv1D(filtNo,filtsize1,kernel_regularizer=l2(l2_size))(input1)
  conv2 = Conv1D(filtNo, filtsize2,kernel_regularizer=l2(l2_size))(conv1)
  BN=BatchNormalization(axis=-1)(conv2)
  ActFunc=Activation(activation)(BN)
  pool1=MaxPooling1D(pool_size=PoolSize)(ActFunc)
  # out=Dropout(drop_size)(pool1)

  return pool1

def define_SMILE(drop_size):


  vectorsize=18
  input_shape=(1024,18)
  denseSize=8
  activation='relu'
  filtsize1=22
  filtNo1=8
  filtsize2=10
  filtNo2=16
  PoolSize=2
  l2_size=None
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
  model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=["accuracy",f1_score,auc_roc,precision,recall2,auc_pr])
  return model

def CNN_Spectrogram(filtNo1,filtNo2,filtsize,input_shape,activation,PoolSize,denseSize):

  # metrics=METRICS

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
  # model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy',metrics=metrics)
  model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy',metrics=["accuracy",f1_score,auc_roc,precision,recall2,auc_pr])

  return model

def CNN_SMILE_NewArch(filtNo1,filtNo2,filtsize,input_shape,activation,PoolSize,denseSize):

  # metrics=METRICS
  vectorsize=18
  dim_data =int(vectorsize*(vectorsize+1)/2)-18
  vector_input = Input((dim_data,))

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

  concat_layer= Concatenate()([flat,vector_input])
  dense=Dense(denseSize)(concat_layer)
  output = Dense(1, activation='sigmoid')(dense)
  model = Model(inputs=[input1, vector_input], outputs=output)
  # model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy',metrics=metrics)
  model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=["accuracy",f1_score,auc_roc,precision,recall2,auc_pr])

  return model

def CNN_SMILE_Spectrogram_NewArch(filtNo1,filtNo2,filtsize,input_shape,activation,PoolSize,denseSize):

  # metrics=METRICS
  vectorsize=18
  dim_data =int(vectorsize*(vectorsize+1)/2)-18
  vector_input = Input((dim_data,))

  input2=Input(input_shape)
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
  flat2=Flatten()(input2)
  concat_layer= Concatenate()([flat,flat2,vector_input])
  dense=Dense(denseSize)(concat_layer)
  output = Dense(1, activation='sigmoid')(dense)
  model = Model(inputs=[input1, input2, vector_input], outputs=output)
  # model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy',metrics=metrics)
  model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy',metrics=["accuracy",f1_score,auc_roc,precision,recall2,auc_pr])

  return model

def define_CNNSpecSMILE(drop_size):

  l2_size=None
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
  # model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=["accuracy", [tf.keras.metrics.AUC(curve="ROC",name='auc_roc')],[tf.keras.metrics.AUC(curve="PR",name='auc_pr')],[tf.keras.metrics.Precision(name="precision")],[tf.keras.metrics.Recall(name="recall")]])
  model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy',metrics=["accuracy",f1_score,auc_roc,precision,recall2,auc_pr])

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
  # mylist=np.array(mylist)
  ListMean=np.mean(mylist,axis=0)
  ListStd=np.std(mylist)
  ListVar=np.var(mylist)

  return ListMean,ListStd,ListVar

def lr_exp_decay(epoch, lr):
  # k = math.sqrt(2)
  k=2
  if lr>0.00001:
    return lr / k
  else:
    return lr

def split(x,y,mi,spectrogram):

  ind_one=[i for i,x in enumerate(y) if x== 1]
  ind_zero=[i for i,x in enumerate(y) if x== 0]

  y_zero=y[ind_zero]
  y_one=y[ind_one]

  x_zero=x[ind_zero]
  x_one=x[ind_one]

  mi_zero=mi[ind_zero]
  mi_one=mi[ind_one]

  spectrogram_zero=spectrogram[ind_zero]
  spectrogram_one=spectrogram[ind_one]

  return x_zero,y_zero,mi_zero,spectrogram_zero,x_one,y_one,mi_one,spectrogram_one

def Make_X(x,mi,spectrogram,cnn,smile,spect,cnnsmilespec,smile_new_arch,smile_spec_new_arch,twodcnn):

  if cnn==1:
    X_train=x

  if smile==1:
    X_train=[x,mi]

  if spect==1:
    X_train=spectrogram

  if cnnsmilespec==1:
    X_train=[x,spectrogram,mi]

  if twodcnn==1:
    x2d=x.reshape(x.shape[0],x.shape[2],x.shape[1],1)
    X_train=x2d
  if smile_new_arch==1:

    X_train=[x,mi]

  if smile_spec_new_arch==1:

    X_train=[x,spectrogram,mi]

  return X_train

def make_batches(x_one,x_zero,y_one,y_zero,mi_one,mi_zero,spectrogram_one,spectrogram_zero,mini_batch,cnn,smile,spect,cnnsmilespec,smile_new_arch,twodcnn):

  total_sample=x_one.shape[0]
  total_sample_zero=x_zero.shape[0]
  # print(len(x_zero) // 128)

  selected_idx = np.random.choice(total_sample,mini_batch)
  selected_idx_zero= np.random.choice(total_sample_zero,mini_batch)

  x_zero_batch= x_zero[selected_idx_zero]
  mi_zero_batch= mi_zero[selected_idx_zero]
  # print(X.shape)
  y_zero_batch = y_zero[selected_idx_zero]
  spectrogram_zero_batch = spectrogram_zero[selected_idx_zero]

  x_one_batch=x_one[selected_idx]
  y_one_batch=y_one[selected_idx]
  mi_one_batch=mi_one[selected_idx]
  spectrogram_one_batch=spectrogram_one[selected_idx]

  x_batch=np.concatenate((x_zero_batch, x_one_batch), axis=0)
  y_batch=np.concatenate((y_zero_batch, y_one_batch), axis=0)
  mi_batch=np.concatenate((mi_zero_batch, mi_one_batch), axis=0)
  spectrogram_batch=np.concatenate((spectrogram_zero_batch, spectrogram_one_batch), axis=0)
  X_main=Make_X( x_batch,mi_batch,spectrogram_batch,cnn,smile,spect,cnnsmilespec,smile_new_arch,smile_spec_new_arch,twodcnn)

  return X_main,y_batch

def validation_batches(x,y,mi,spectrogram,batch_size,cnn,smile,spect,cnnsmilespec,twodcnn):

  total_sample=x.shape[0]
  selected_idx = np.random.choice(total_sample,batch_size)
  x_one_batch=x[selected_idx]
  y_one_batch=y[selected_idx]
  mi_one_batch=mi[selected_idx]
  spectrogram_one_batch=spectrogram[selected_idx]

  X_main=Make_X(x_one_batch,mi_one_batch,spectrogram_one_batch,cnn,smile,spect,cnnsmilespec,smile_new_arch,smile_spec_new_arch,twodcnn)

  return X_main,y_one_batch

def validation_generator(x,y,mi,spectrogram,batch_size,cnn,smile,spect,cnnsmilespec,twodcnn):

  while True:
    x_batch,y_batch=validation_batches(x,y,mi,spectrogram,batch_size,cnn,smile,spect,cnnsmilespec,twodcnn)
    yield x_batch,y_batch

def validation_generator2(x,y,mi,spectrogram,batch_size,cnn,smile,spect,cnnsmilespec,twodcnn):

  while True:
    total_sample=x.shape[0]
    selected_idx = np.random.choice(total_sample,batch_size)
    x_one_batch=x[selected_idx]
    y_one_batch=y[selected_idx]
    mi_one_batch=mi[selected_idx]
    spectrogram_one_batch=spectrogram[selected_idx]
    yield [x_one_batch,mi_one_batch,spectrogram_one_batch],y_batch

def train_generator(x_one,x_zero,y_one,y_zero,mi_one,mi_zero,spectrogram_one,spectrogram_zero,mini_batch,cnn,smile,spect,cnnsmilespec,twodcnn):
  while True:
    x_batch,y_batch=make_batches(x_one,x_zero,y_one,y_zero,mi_one,mi_zero,spectrogram_one,spectrogram_zero,mini_batch,cnn,smile,spect,cnnsmilespec,smile_new_arch,twodcnn)
    yield x_batch,y_batch

def Plot_func(SaveHisResults,metric,metric_val,metric_name,modelname,epoch,batchSize):

  metric_mean,metric_std,_=MeanStdVar(metric)
  metric_val_mean,metric_val_std,_=MeanStdVar(metric_val)


  plt.plot(metric_mean)
  plt.plot(metric_val_mean)
  plt.title(modelname+'_'+metric_name+'epoch_'+str(epoch)+'_batchsize_'+str(batchSize))
  plt.legend(['train', 'test'], loc='upper left')

  plt.fill_between(range(epoch), metric_mean-metric_std, metric_mean+metric_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), metric_val_mean-metric_val_std, metric_val_mean+metric_val_std, color='orange', alpha = 0.5)

  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_'+metric_name+'_epoch_'+str(epoch)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()

def Select_Model(cnn,smile,spectrogram,cnnsmilespec,smile_new_arch,smile_spec_new_arch,twodcnn,diff,drop_size):

  if cnn==1:
    tf.keras.backend.clear_session()
    model=define_CNN(drop_size)

  if smile==1:
    tf.keras.backend.clear_session()
    model=define_SMILE(drop_size)

  if diff==1:
    tf.keras.backend.clear_session()
    model=define_CNNSMILEDiff(drop_size)

  if twodcnn==1:

    tf.keras.backend.clear_session()
    model=define_2DCNN()


  if spectrogram==1:
    tf.keras.backend.clear_session()
    model=CNN_Spectrogram(filtNo1=120,filtNo2=160,filtsize=5,input_shape=(1024,18),activation='relu',PoolSize=3,denseSize=8)

  if cnnsmilespec==1:
    tf.keras.backend.clear_session()
    model=define_CNNSpecSMILE(drop_size)

  if smile_new_arch==1:
    tf.keras.backend.clear_session()
    model=CNN_SMILE_NewArch(filtNo1=120,filtNo2=160,filtsize=5,input_shape=(1024,18),activation='relu',PoolSize=3,denseSize=8)

  if smile_spec_new_arch==1:

    tf.keras.backend.clear_session()
    model=CNN_SMILE_Spectrogram_NewArch(filtNo1=120,filtNo2=160,filtsize=5,input_shape=(1024,18),activation='relu',PoolSize=3,denseSize=8)
  return model

dirname2='/media/datadrive/bsalafian/AllMatFiles'
dirname='/media/datadrive/bsalafian/6FoldCrossSMILE'
SaveHisResults='/home/baharsalafian/CustomMetrics_results_fit_generator_Allcase'
ModelResults='/home/baharsalafian/CustomMetrics_model_fit_generator_Allcase'

modelname='CNN_SMILE_Spectrogram_24Testfiles_init_0.01_new_arch'
initLR=0.01
cnn=0
smile_new_arch=0
smile_spec_new_arch=1
smile=0
spectrogram=0
cnnsmilespec=0

twodcnn=0
diff=0
num_epochs=100
batch_size=256
SeqLen=1
l2_size=None
drop_size=0


EDFFiles=PatientsEDFFile(dirname)


# def Model_train(dirname,dirname2,ModelResults,SaveHisResults,modelname,initLR,cnn,smile,spectrogram,cnnsmilespec,twodcnn,diff,num_epochs,batch_size,SeqLen,l2_size,drop_size):
FoldNum=2


loss=[]
loss_val=[]

acc=[]
acc_val=[]

LR=[]
f1=[]
f1_val=[]

ROC=[]
ROC_val=[]

PR=[]
PR_val=[]

PR_m=[]
PR_m_val=[]

recall=[]
recall_val=[]





kfold = KFold(n_splits=FoldNum, shuffle=False)
# start_time = time.time()
fold_no=1
indx=range(0,24)
start_time = time.time()
mini_batch=int(batch_size/2)
for i in range(2):
  # batch_size=256
  # num_epochs = 2

  testindx,trainindx=SelectIndx(i,indx)
  X_train, y_train, mi,mi_diff,spec_train=ReadMatFiles(dirname,dirname2,trainindx,seq_len=SeqLen,diff=None)
  # x_zero,y_zero,mi_zero,spectrogram_zero,x_one,y_one,mi_one,spectrogram_one=split(X_train, y_train, mi,spec_train)
  X_test, y_test, mitest,mitest_diff,spec_test=ReadMatFiles(dirname,dirname2,testindx,seq_len=SeqLen,diff=None)

  x_zero,y_zero,mi_zero,spectrogram_zero,x_one,y_one,mi_one,spectrogram_one=split(X_train,y_train,mi,spec_train)

  train_steps=int(X_train.shape[0]//batch_size)
  test_steps=int(X_test.shape[0]//batch_size)

  tf.keras.backend.clear_session()
  model=Select_Model(cnn,smile,spectrogram,cnnsmilespec,smile_new_arch,smile_spec_new_arch,twodcnn,diff,drop_size)
  TrainGen=train_generator(x_one,x_zero,y_one,y_zero,mi_one,mi_zero,spectrogram_one,spectrogram_zero,mini_batch,cnn,smile,spectrogram,cnnsmilespec,twodcnn)
  # TestGen=validation_generator(X_test, y_test, mitest,spec_test,batch_size,cnn,smile,spectrogram,cnnsmilespec,twodcnn)
  # K.get_session().run(tf.local_variables_initializer())
  history=model.fit_generator(TrainGen,steps_per_epoch=train_steps,epochs=num_epochs,
                      verbose=2,
                      callbacks=[LearningRateScheduler(lr_exp_decay)],
                      validation_data=validation_generator(X_test, y_test, mitest,spec_test,batch_size,cnn,smile,spectrogram,cnnsmilespec,twodcnn),
                      validation_steps=test_steps)

  model.save(ModelResults+'/'+modelname+'_fold'+str(i+1)+'_epoch_'+str(num_epochs)+'_batchsize_'+str(batch_size)+'.h5')

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

  recall.append(history.history['recall2'])
  recall_val.append(history.history['val_recall2'])

  p=np.array(history.history['precision'])
  print("len of precision",len(p))
  r=np.array(history.history['recall2'])
  p_val=np.array(history.history['val_precision'])
  r_val=np.array(history.history['val_recall2'])
  #
  # f1.append(2 * ((p * r) / (p + r)))
  # f1_val.append(2 * ((p_val * r_val) / (p_val + r_val)))
  f1.append(history.history['f1_score'])
  f1_val.append(history.history['val_f1_score'])

  ROC.append(history.history['auc_roc'])
  ROC_val.append(history.history['val_auc_roc'])
# print("length of acc",acc_all[0].shape, acc_all[1].shape)
# print("length of acc_train", val_acc_all[0].shape, val_acc_all[1].shape)
Plot_func(SaveHisResults,acc,acc_val,'accuracy'+'_initLR_'+str(initLR)+'_',modelname,num_epochs,batch_size)
Plot_func(SaveHisResults,loss,loss_val,'loss'+'_initLR_'+str(initLR)+'_',modelname,num_epochs,batch_size)
Plot_func(SaveHisResults,f1,f1_val,'f1_score'+'_initLR_'+str(initLR)+'_',modelname,num_epochs,batch_size)
Plot_func(SaveHisResults,PR,PR_val,'AUC_PR'+'_initLR_'+str(initLR)+'_',modelname,num_epochs,batch_size)
Plot_func(SaveHisResults,ROC,ROC_val,'AUC_ROC'+'_initLR_'+str(initLR)+'_',modelname,num_epochs,batch_size)
Plot_func(SaveHisResults,PR_m,PR_m_val,'Precision'+'_initLR_'+str(initLR)+'_',modelname,num_epochs,batch_size)
Plot_func(SaveHisResults,recall,recall_val,'Recall'+'_initLR_'+str(initLR)+'_',modelname,num_epochs,batch_size)

print("--- %s seconds ---" % (time.time() - start_time))
