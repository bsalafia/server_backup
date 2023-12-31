# -*- coding: utf-8 -*-
"""Copy of CNNSMILEGRU_Bs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Fgyv5Og_PPfexz3ih-4uEJqa-PfQsKSA
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
  # Concatenate the convolutional features and the vector input
  concat_layer= Concatenate()([flat,vector_input])
  denseout = Dense(100, activation='relu')(concat_layer)
  denseout = Dense(50, activation='relu')(denseout)
  output = Dense(1, activation='sigmoid')(denseout)

  # define a model with a list of two inputs
  model = Model(inputs=[input1, vector_input], outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['accuracy'])
  return model

def define_CNN():


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

def define_CNNSMILEDiff():
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
    dim_data =int(vectorsize*(vectorsize+1)/2)-18
    vector_input1 = Input((dim_data,))
    vector_input2 = Input((dim_data,))
    # Concatenate the convolutional features and the vector input
    concat_layer= Concatenate()([flat,vector_input1,vector_input2])
    denseout = Dense(100, activation='relu')(concat_layer)
    denseout = Dense(50, activation='relu')(denseout)
    output = Dense(1, activation='sigmoid')(denseout)

    # define a model with a list of two inputs
    model = Model(inputs=[input1, vector_input1,vector_input2], outputs=output)


    model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['accuracy'])

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

def ReadMatFiles(dirname,indx, seq_len=1,diff=None):

  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]
  ind=[]

  MI_all=[]
  X=[]
  Y=[]
  MI_diff_all=[]

  for j in list(indx):
    print(j)
    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
    ind.append(indices)

  ind=np.concatenate(ind,axis=0)

  for k in range(len(ind)):
    # print(ind[k])
    matfile=loadmat(os.path.join(dirname,EDF[int(ind[k])]))
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
    X.append(x)
    Y.append(real_y)
    MI_all.append(MI)
    MI_diff_all.append(MI_diff)

  #
  # X=np.concatenate(X,axis=0)
  # Y=np.concatenate(Y,axis=0)
  # MI_all=np.concatenate(MI_all,axis=0)
  # MI_diff_all=np.concatenate(MI_diff_all,axis=0)
  #
  # print(X.shape)
  # print(Y.shape)
  # print(MI_all.shape)
  # print(MI_diff_all.shape)

  return X, Y, MI_all, MI_diff_all

def Xtrain(cnn,smile,diff,twodcnn,x,y,mi,mi_diff):

  if cnn==1:
    X_train=x
  if smile==1:
    X_train=[x,mi]
  if diff==1:
    X_train=[x,mi,mi_diff]

  if twodcnn==1:
    x2d=x.reshape(x.shape[0],x.shape[2],x.shape[1],1)
    X_train=x2d

  return X_train







def ModelTrain(dirname,modeldir,SaveResults,SaveResultsTruePred,modelname,seq_len,cnn,smile,diff,twodcnn,indx,gru,Noepoch):

  FoldNum=6
  kfold = KFold(n_splits=FoldNum, shuffle=False)
  start_time = time.time()
  fold_no=0

  savenamef1='fscore_'+modelname+'_'+indx+'_'+str(Noepoch)
  cfname='cfmat_'+modelname+'_'+indx+'_'+str(Noepoch)
  fprname='fpr_'+modelname+'_'+indx+'_'+str(Noepoch)
  tprname='tpr_'+modelname+'_'+indx+'_'+str(Noepoch)
  prname='PR_'+modelname+'_'+indx+'_'+str(Noepoch)
  rocname='ROC_'+modelname+'_'+indx+'_'+str(Noepoch)

  for th in [0.3,0.4,0.5,0.6,0.7]:

    fold_no=0
    fpr=[]
    tpr=[]
    PR=[]
    ROC=[]
    pred=[]
    act=[]
    fscore=[]

    for trainindx, testindx in kfold.split(range(24)):


      if indx=='test':
        ind=testindx
      else:
        ind=trainindx

      x, y, mi,mi_diff=ReadMatFiles(dirname,ind,seq_len,diff)

      X=np.concatenate(x,axis=0)
      Y=np.concatenate(y,axis=0)
      MI_all=np.concatenate(mi,axis=0)
      MI_diff_all=np.concatenate(mi_diff,axis=0)
      fold_no=fold_no+1


      X_train=Xtrain(cnn,smile,diff,twodcnn,X,Y,MI_all,MI_diff_all)

      ModelName1=modelname+ '_fold'+str(fold_no)+'_epoch_'+str(Noepoch)+'.h5'

      model1=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))

      ypred=model1.predict(X_train)
      ypred_th = (ypred > th).astype(int)

      if gru==1:

        Y = Y[:, 2]
        ypred= ypred[:, 2]
        ypred_th= (ypred > th).astype(int)

      if th==0.7:

        fpr1, tpr1, _ = roc_curve(Y, ypred)
        precision, recall, _ = precision_recall_curve(Y, ypred)
        PR1=auc(recall, precision)
        ROC1=roc_auc_score(Y,ypred)
        fpr.append(fpr1)
        tpr.append(tpr1)
        PR.append(PR1)
        ROC.append(ROC1)

      precision, recall, f1, _ = precision_recall_fscore_support(Y, ypred_th, average='weighted')

      pred.append(list(ypred_th))
      act.append(list(Y))
      fscore.append(f1)
      pred1=np.concatenate(pred,axis=0)
      act1=np.concatenate(act,axis=0)
      cnf_matrix = confusion_matrix(act1, pred1)
      #
    np.save(os.path.join(SaveResults, savenamef1+'_'+str(th)),  fscore)
    np.save(os.path.join(SaveResults, cfname+'_'+str(th)),  cnf_matrix)

    if th==0.7:

      np.save(os.path.join(SaveResults, fprname),  fpr)
      np.save(os.path.join(SaveResults, tprname),  tpr)
      np.save(os.path.join(SaveResults, prname), PR)
      np.save(os.path.join(SaveResults, rocname),  ROC)


  ####################################
  for j in range(3):
      fold_no=0
      FoldNum=6
      kfold = KFold(n_splits=FoldNum, shuffle=False)

      for trainindx, testindx in kfold.split(range(24)):

        if indx=='test':
          ind=testindx
        else:
          ind=trainindx
        fold_no=fold_no+1
        x, y, mi,mi_diff=ReadMatFiles(dirname,ind,seq_len,diff)

        X_train=Xtrain(cnn,smile,diff,twodcnn,x[j],y[j],mi[j],mi_diff[j])

        ModelName1=modelname+ '_fold'+str(fold_no)+'_epoch_'+str(Noepoch)+'.h5'
        model1=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))

        ypred_plot=model1.predict(X_train)

        if gru==1:

          y = y[j][:, 2]
          ypred_plot= ypred_plot[:, 2]

        else:
          y = y[j]

        plt.subplot(3,2,fold_no)

        plt.plot(range(len(y)), y)
        plt.plot(range(len(y)),ypred_plot)


      plt.suptitle(modelname+'_EDFNo_'+str(j+1)+'_'+indx)

      plt.savefig(SaveResultsTruePred+'/'+modelname+'_EDFNo_'+str(j+1)+'_epoch_'+str(Noepoch)+'_'+indx+'.pdf', format='pdf', bbox_inches = 'tight')
      plt.clf()


  print("--- %s seconds ---" % (time.time() - start_time))

dirname='/home/baharsalafian/6FoldCrossSMILE'

modeldir='/home/baharsalafian/CNNSMILEGRU100Epoch_Pretrained'

SaveResults='/home/baharsalafian/Results100Epoch_Pretrained'

SaveResultsTruePred='/home/baharsalafian/TruePredPlots_Pretrained'

ModelTrain(dirname,modeldir,SaveResults,SaveResultsTruePred,'CNNSMILE10times'+'_Pretrained_GRU'+'10times',seq_len=3,cnn=0,smile=1,diff=0,twodcnn=0,indx='train',gru=1,Noepoch=100)

## '2DCNN10times' : name of the model that you wanna load
