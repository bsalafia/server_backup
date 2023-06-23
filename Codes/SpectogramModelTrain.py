# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SIBykTwsWcs4r6_8QEnL1cHlVdgGVblJ
"""

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
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
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
# from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.datasets import cifar10
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D, GlobalAveragePooling1D,Bidirectional,AveragePooling1D
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

def Conv_BN_Act_Pool(filtNo1,filtNo2,filtsize,input_shape,activation,PoolSize,denseSize):



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
  model.compile(optimizer=Adam(learning_rate=0.00001),loss='binary_crossentropy', metrics=['accuracy'])
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

def ReadMatFiles(dirname,dirname2,indx):


  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
  EDF2=PatientsEDFFile(dirname2)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]
  X=[]
  Y=[]
  ind=[]
  # print(ind)
  for j in list(indx):

    # print(j)
    indices = [i for i, elem in enumerate(EDF2) if Name[j] in elem]
    ind.append(indices)

  ind=np.concatenate(ind,axis=0)
  for k in range(len(ind)):
    #
    # print(EDF2[ind[k]])
    #
    # time.sleep(2)

    matfile2=loadmat(os.path.join(dirname2,EDF2[ind[k]]))
    Name2=EDF2[ind[k]].split('.')
    matfile=loadmat(os.path.join(dirname,Name2[0]+'_Spectogram.mat'))
    x=matfile['spectogram']
    y=matfile2['Y_label_4sec']

    y=np.transpose(y)
    start_idx = np.argmax(y>0)
    a = y == 1
    end_idx = len(a) - np.argmax(np.flip(a)) - 1
    real_y = np.zeros_like(y)
    real_y[start_idx:end_idx+1] = 1
    X.append(x)
    Y.append(real_y)



  X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)

  print(X.shape)
  print(Y.shape)

  return X, Y

def MeanStdVar(mylist):

  ListMean=np.mean(mylist,axis=0)
  ListStd=np.std(mylist)
  ListVar=np.var(mylist)

  return ListMean,ListStd,ListVar

def ModelTrain(dirname,dirname2,SaveResults,SaveHisResults,modelname,batchsize,epoch):


  loss=[]
  loss_val=[]
  acc=[]
  acc_val=[]
  FoldNum=6
  kfold = KFold(n_splits=FoldNum, shuffle=False)
  start_time = time.time()
  fold_no=1
  for trainindx, testindx in kfold.split(range(24)):

    xtrain, ytrain=ReadMatFiles(dirname,dirname2,trainindx)

    from sklearn.utils import class_weight
    cw = class_weight.compute_class_weight('balanced', np.unique(ytrain),np.ravel(ytrain))
    print(cw)
    class_weights = {0:cw[0], 1: cw[1]}

    xtest, ytest=ReadMatFiles(dirname,dirname2,testindx)

    model=Conv_BN_Act_Pool(filtNo1=120,filtNo2=160,filtsize=5,input_shape=(1024,18),activation='relu',PoolSize=3,denseSize=8)

    history=model.fit(xtrain, ytrain, validation_data=(xtest,ytest), batch_size=batchsize, epochs=epoch, class_weight=class_weights,verbose = 2)
    model.save(SaveResults+'/'+modelname+'_fold'+str(fold_no)+'_epoch_'+str(epoch)+'.h5')
    loss.append(history.history['loss'])
    loss_val.append(history.history['val_loss'])
    acc.append(history.history['accuracy'])
    acc_val.append(history.history['val_accuracy'])
    fold_no=fold_no+1
    tf.keras.backend.clear_session()

###########################################
  loss_mean,loss_std,_=MeanStdVar(loss)
  loss_val_mean,loss_val_std,_=MeanStdVar(loss_val)
  acc_mean,acc_std,_=MeanStdVar(acc)
  acc_val_mean,acc_val_std,_=MeanStdVar(acc_val)
  np.savez(os.path.join(SaveHisResults, 'HistoryRes_'+modelname+'_Spectogram_ClassWeight'), loss=loss, loss_val=loss_val, accuracy=acc, accuracy_val=acc_val)
  plt.plot(acc_mean)
  plt.plot(acc_val_mean)
  plt.title(modelname+'_accuracy'+'_epoch_'+str(epoch))
  plt.legend(['train', 'test'], loc='upper left')
  plt.fill_between(range(epoch), acc_mean-acc_std, acc_mean+acc_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), acc_val_mean-acc_val_std, acc_val_mean+acc_val_std, color='orange', alpha = 0.5)
  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_accuracy'+'_Spectogram_ClassWeight'+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()
  plt.plot(loss_mean)
  plt.plot(loss_val_mean)
  plt.title(modelname+'_loss'+'_epoch_'+str(epoch))
  plt.legend(['train', 'test'], loc='upper left')
  plt.fill_between(range(epoch), loss_mean-loss_std, loss_mean+loss_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), loss_val_mean-loss_val_std, loss_val_mean+loss_val_std, color='orange', alpha = 0.5)
  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_loss'+'_Spectogram_ClassWeight'+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()
  print("--- %s seconds ---" % (time.time() - start_time))

dirname2='/media/datadrive/bsalafian/6FoldCrossSMILE'
dirname='/home/baharsalafian/SpectogramResultsSyncMI_Nobutter'
SaveResults='/home/baharsalafian/CNNSMILEGRUModels_JBHI_LossTest'
SaveHisResults='/home/baharsalafian/History_JBHI_LossTest'

ModelTrain(dirname,dirname2,SaveResults,SaveHisResults,'CNN_Spectogram10times_lr.00001',batchsize=256,epoch=100)
