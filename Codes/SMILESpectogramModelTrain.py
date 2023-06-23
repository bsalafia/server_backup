# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SIBykTwsWcs4r6_8QEnL1cHlVdgGVblJ
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
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


def  Conv_BN_Act_Pool(filtNo,filtsize1,filtsize2,input1,activation,PoolSize):

  conv1 = Conv1D(filtNo,filtsize1)(input1)
  conv2 = Conv1D(filtNo, filtsize2)(conv1)
  BN=BatchNormalization(axis=-1)(conv2)
  ActFunc=Activation(activation)(BN)
  pool1=MaxPooling1D(pool_size=PoolSize)(ActFunc)

  return pool1
def define_SMILE():


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
  model1=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,input1,activation,PoolSize)
  model2=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model1,activation,PoolSize)
  model3=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model2,activation,PoolSize)
  model4=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model3,activation,PoolSize)
  model5=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model4,activation,PoolSize)
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
  denseout = Dense(100, activation='relu')(concat_layer)
  denseout = Dense(50, activation='relu')(denseout)
  output = Dense(1, activation='sigmoid')(denseout)

  # define a model with a list of two inputs
  model = Model(inputs=[input1,input2, vector_input], outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy', metrics=['accuracy'])
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

def ReadMatFiles(dirname,dirname2,ind):


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
  MI_all=[]
  Xcnn=[]
  # print(ind)
  for k in range(len(ind)):

    # print(ind[k])


    matfile2=loadmat(os.path.join(dirname2,EDF2[ind[k]]))

    Name2=EDF2[ind[k]].split('.')

    matfile=loadmat(os.path.join(dirname,Name2[0]+'_Spectogram.mat'))
    x=matfile['spectogram']

    xcnn=matfile2['X_4sec']
    y=matfile2['Y_label_4sec']
    mi=matfile2['estimated_MI']

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

    X.append(x)
    Xcnn.append(xcnn)
    Y.append(real_y)
    MI_all.append(MI)



  X=np.concatenate(X,axis=0)
  Xcnn=np.concatenate(Xcnn,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI_all=np.concatenate(MI_all,axis=0)

  print(X.shape)
  print(Xcnn.shape)
  print(Y.shape)
  print(MI_all.shape)

  return Xcnn,X, Y,MI_all

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

    xcnn_train,xtrain, ytrain,mi_train=ReadMatFiles(dirname,dirname2,trainindx)
    #
    # print(xcnn_train.shape)
    # print(xtrain.shape)
    # print(ytrain.shape)
#
    from sklearn.utils import class_weight
    cw = class_weight.compute_class_weight('balanced', np.unique(ytrain),np.ravel(ytrain))
    print(cw)
    class_weights = {0:cw[0], 1: cw[1]}

    xcnn_test,xtest, ytest,mi_test=ReadMatFiles(dirname,dirname2,testindx)

    model=define_SMILE()

    history=model.fit([xcnn_train,xtrain,mi_train], ytrain, validation_data=([xcnn_test,xtest,mi_test],ytest), batch_size=batchsize, epochs=epoch, class_weight=class_weights,verbose = 2)
    model.save(SaveResults+'/'+modelname+'_fold'+str(fold_no)+'.h5')
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
  np.savez(os.path.join(SaveHisResults, 'HistoryRes_'+modelname), loss=loss, loss_val=loss_val, accuracy=acc, accuracy_val=acc_val)
  plt.plot(acc_mean)
  plt.plot(acc_val_mean)
  plt.title(modelname+'_loss'+'_Spectogram')
  plt.legend(['train', 'test'], loc='upper left')
  plt.fill_between(range(epoch), acc_mean-acc_std, acc_mean+acc_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), acc_val_mean-acc_val_std, acc_val_mean+acc_val_std, color='orange', alpha = 0.5)
  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_accuracy'+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()
  plt.plot(loss_mean)
  plt.plot(loss_val_mean)
  plt.title(modelname+'_loss'+'_epoch_'+str(epoch)+'_batchsize_'+str(batchsize))
  plt.legend(['train', 'test'], loc='upper left')
  plt.fill_between(range(epoch), loss_mean-loss_std, loss_mean+loss_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), loss_val_mean-loss_val_std, loss_val_mean+loss_val_std, color='orange', alpha = 0.5)
  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_loss'+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()
  print("--- %s seconds ---" % (time.time() - start_time))

dirname='/home/baharsalafian/SpectogramResultsSyncMI_Nobutter'
dirname2='/home/baharsalafian/6FoldCrossSMILE'
SaveResults='/home/baharsalafian/CNNSMILESpectrogramModels'
SaveHisResults='/home/baharsalafian/CNNSMILESpectrogramHist'

ModelTrain(dirname,dirname2,SaveResults,SaveHisResults,'CNN_Spectogram_SMILE',batchsize=256,epoch=100)
