# -*- coding: utf-8 -*-
"""Copy of CNNSMILEGRU_Bs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Fgyv5Og_PPfexz3ih-4uEJqa-PfQsKSA
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
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
from tensorflow.keras.regularizers import l2, l1_l2
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
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.datasets import cifar10
from keras.optimizers import Adam
from sklearn.metrics import precision_recall_fscore_support,precision_recall_curve
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import f1_score,plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve,roc_curve,roc_auc_score,auc
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D, GlobalAveragePooling1D,Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
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
from sklearn import metrics
import gc
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import math
datagen = ImageDataGenerator(zca_whitening=True)
from keras import backend as K
# from google.colab import drive
# drive.mount('/content/drive')


class MetricsHistory(tf.keras.callbacks.Callback):

  def __init__(self, train, validation=None):
      super(MetricsHistory, self).__init__()
      self.validation = validation
      self.train = train

  def on_epoch_end(self, batch, logs={}):

    logs['F1_score_train'] = float('-inf')
    logs['PR_train'] = float('-inf')
    logs['ROC_train'] = float('-inf')

    X_train, y_train = self.train[0], self.train[1]
    y_pred = self.model.predict(X_train)

    ypred_th = (y_pred > 0.5).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(y_train, ypred_th, average='weighted')

    precision, recall, _ = precision_recall_curve(y_train, y_pred)
    PR=auc(recall, precision)
    ROC=roc_auc_score(y_train,y_pred)

    if (self.validation):

      logs['F1_score_val'] = float('-inf')
      logs['PR_val'] = float('-inf')
      logs['ROC_val'] = float('-inf')

      X_valid, y_valid = self.validation[0], self.validation[1]
      y_val_pred = self.model.predict(X_valid)

      ypred_val_th = (y_val_pred > 0.5).astype(int)
      _, _, f1_val, _ = precision_recall_fscore_support(y_valid, ypred_val_th, average='weighted')

      precision_val, recall_val, _ = precision_recall_curve(y_valid,y_val_pred)

      PR_val=auc(recall_val, precision_val)
      ROC_val=roc_auc_score(y_valid,y_val_pred)

      logs['F1_score_train'] = np.round(f1, 3)
      logs['F1_score_val'] = np.round(f1_val,3)

      logs['PR_train'] = np.round(PR, 3)
      logs['ROC_train'] = np.round(ROC, 3)

      logs['PR_val'] = np.round(PR_val, 3)
      logs['ROC_val'] = np.round(ROC_val, 3)

    else:

      logs['F1_score_train'] = np.round(f1,3)
      logs['PR_train'] = np.round(PR, 3)
      logs['ROC_train'] = np.round(ROC, 3)

def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

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


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())



def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def PR(y_true, y_pred):
  pr=precision(y_true, y_pred)
  rec=recall(y_true, y_pred)
  # precision, recall, _ = precision_recall_curve(y_true, y_pred)
  PR1=auc(rec,pr)
  return PR1

def ROC(y_true, y_pred):

  ROC1=roc_auc_score(y_true, y_pred)

  return ROC

# def f1(y_true, y_pred):

#   ypred_th = (y_pred > th).astype(int)
#   _, _, f11, _ = precision_recall_fscore_support(y_true, ypred_th, average='weighted')

#   return f11

def auc(y_true, y_pred):

  return roc_auc_score(K.eval(y_true), K.eval(y_pred))







###########################
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
  denseout = Dense(50)(denseout)
  output = Dense(1, activation='sigmoid')(denseout)

  # define a model with a list of two inputs
  model = Model(inputs=[input1, vector_input], outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=["accuracy", tf.keras.metrics.AUC(curve="ROC",name='auc_roc'),tf.keras.metrics.AUC(curve="PR",name='auc_pr'),f1])
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
  model.compile(optimizer=Adam(learning_rate=0.00001),loss='binary_crossentropy', metrics=['accuracy'])
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


  X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI_all=np.concatenate(MI_all,axis=0)
  MI_diff_all=np.concatenate(MI_diff_all,axis=0)
  print(Y)
  print(X.shape)
  print(Y.shape)
  print(MI_all.shape)
  print(MI_diff_all.shape)

  return X, Y, MI_all, MI_diff_all

def MeanStdVar(mylist):

  ListMean=np.mean(mylist,axis=0)
  ListStd=np.std(mylist)
  ListVar=np.var(mylist)

  return ListMean,ListStd,ListVar

class ModelMetrics(tf.keras.callbacks.Callback):

  def on_train_begin(self,logs={}):
    self.precisions=[]
    self.recalls=[]
    self.f1_scores=[]
  def on_epoch_end(self, batch, logs={}):

    y_val_pred=self.model.predict(X_test)

    _precision,_recall,_f1,_sample=score(ytest,y_val_pred)

    self.precisions.append(_precision)
    self.recalls.append(_recall)
    self.f1_scores.append(_f1)

class MetricsHistory(tf.keras.callbacks.Callback):

  def __init__(self, model,train, validation=None):
      super(MetricsHistory, self).__init__()
      self.validation = validation
      self.train = train
      self.model=model

  def on_epoch_end(self, epoch, logs={}):

    logs['F1_score_train'] = float('-inf')
    logs['PR_train'] = float('-inf')
    logs['ROC_train'] = float('-inf')

    X_train, y_train = self.train[0], self.train[1]
    y_pred = self.model.predict(X_train)

    ypred_th = (y_pred > 0.5).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(y_train, ypred_th, average='weighted')

    precision, recall, _ = precision_recall_curve(y_train, y_pred)
    PR=auc(recall, precision)
    ROC=roc_auc_score(y_train,y_pred)

    if (self.validation):

      logs['F1_score_val'] = float('-inf')
      logs['PR_val'] = float('-inf')
      logs['ROC_val'] = float('-inf')

      X_valid, y_valid = self.validation[0], self.validation[1]
      y_val_pred = self.model.predict(X_valid)

      ypred_val_th = (y_val_pred > 0.5).astype(int)
      _, _, f1_val, _ = precision_recall_fscore_support(y_valid, ypred_val_th, average='weighted')

      precision_val, recall_val, _ = precision_recall_curve(y_valid,y_val_pred)

      PR_val=auc(recall_val, precision_val)
      ROC_val=roc_auc_score(y_valid,y_val_pred)

      logs['F1_score_train'] = np.round(f1, 3)
      logs['F1_score_val'] = np.round(f1_val,3)

      logs['PR_train'] = np.round(PR, 3)
      logs['ROC_train'] = np.round(ROC, 3)

      logs['PR_val'] = np.round(PR_val, 3)
      logs['ROC_val'] = np.round(ROC_val, 3)

    else:

      logs['F1_score_train'] = np.round(f1,3)
      logs['PR_train'] = np.round(PR, 3)
      logs['ROC_train'] = np.round(ROC, 3)


def ModelTrain(dirname,SaveResults,SaveHisResults,modeldir,modelname,modelnameLoad,seq_len,cnn,smile,diff,twodcnn,epoch, epochLoad,l2_size,drop_size,batchSize):

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

  FoldNum=6
  kfold = KFold(n_splits=FoldNum, shuffle=False)
  start_time = time.time()
  fold_no=1
  for trainindx, testindx in kfold.split(range(24)):

    batchsize=batchSize


    x, y, mi,mi_diff=ReadMatFiles(dirname,trainindx,seq_len,diff)
    xtest, ytest, mitest,mitest_diff=ReadMatFiles(dirname,testindx,seq_len,diff)

    if cnn==1:

      # tf.keras.backend.clear_session()
      ModelName1=modelnameLoad+ '_fold'+str(fold_no)+'_epoch_'+str(Noepoch)+'_l2Size_None_DropSize_0_batchsize_256'+'.h5'

      model=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))
      # model=define_CNN(l2_size,drop_size)
      X_train=x
      X_test=xtest

    if smile==1:
      tf.keras.backend.clear_session()
      model=define_SMILE(l2_size,drop_size)
      # ModelName1=modelnameLoad+ '_fold'+str(fold_no)+'_epoch_'+str(epochLoad)+'_l2Size_None_DropSize_0_batchsize_256'+'.h5'
      #
      # model=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))


      X_train=[x,mi]
      X_test=[xtest,mitest]

    if diff==1:

      ModelName1=modelnameLoad+ '_fold'+str(fold_no)+'_epoch_'+str(epochLoad)+'_l2Size_None_DropSize_0_batchsize_256'+'.h5'

      model=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))

      X_train=[x,mi,mi_diff]
      X_test=[xtest,mitest,mitest_diff]

    if twodcnn==1:

      ModelName1=modelnameLoad+ '_fold'+str(fold_no)+'_epoch_'+str(epochLoad)+'_l2Size_None_DropSize_0_batchsize_256'+'.h5'

      model=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))

      x=x.reshape(x.shape[0],x.shape[2],x.shape[1],1)
      xtest=xtest.reshape(xtest.shape[0],xtest.shape[2],xtest.shape[1],1)
      X_train=x
      X_test=xtest



    ##################################################################
    from sklearn.utils import class_weight
    cw = class_weight.compute_class_weight('balanced', np.unique(y),np.ravel(y))
    print(cw)
    class_weights = {0:cw[0], 1: cw[1]}
    def lr_exp_decay(epoch, lr):
      # k = math.sqrt(2)

      k=2

      if lr>0.00001:
        return lr / k

      else:
        return lr

  # print(y)
    # print(sample_weight)
    # metrics=ModelMetrics()
    history=model.fit(X_train, y, validation_data=(X_test,ytest), batch_size=batchsize, epochs=epoch, class_weight=class_weights, callbacks=[LearningRateScheduler(lr_exp_decay, verbose=2)])
    # print(metrics.precisions)
    print(history.history.keys())

    model.save(SaveResults+'/'+modelname+'_fold'+str(fold_no)+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.h5')

    loss.append(history.history['loss'])
    loss_val.append(history.history['val_loss'])

    acc.append(history.history['accuracy'])
    acc_val.append(history.history['val_accuracy'])
    LR.append(history.history['lr'])

    f1.append(history.history['f1'])
    f1_val.append(history.history['val_f1'])

    PR.append(history.history['auc_pr'])
    PR_val.append(history.history['val_auc_pr'])

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

  PR_mean,PR_std,_=MeanStdVar(PR)
  PR_val_mean,PR_val_std,_=MeanStdVar(PR_val)

  ROC_mean,ROC_std,_=MeanStdVar(ROC)
  ROC_val_mean,ROC_val_std,_=MeanStdVar(ROC_val)

  np.savez(os.path.join(SaveHisResults, 'HistoryRes_'+modelname+'_epoch_'+str(epoch+epochLoad)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)), loss=loss, loss_val=loss_val, accuracy=acc, accuracy_val=acc_val, LR=LR, PR=PR, PR_val=PR_val,ROC=ROC, ROC_val=ROC_val, f1=f1, f1_val=f1_val)

  plt.plot(acc_mean)
  plt.plot(acc_val_mean)
  plt.title(modelname+'_accuracy'+'_epoch_'+str(epoch+epochLoad)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize))
  plt.legend(['train', 'test'], loc='upper left')

  plt.fill_between(range(epoch), acc_mean-acc_std, acc_mean+acc_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), acc_val_mean-acc_val_std, acc_val_mean+acc_val_std, color='orange', alpha = 0.5)

  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_accuracy'+'_epoch_'+str(epoch+epochLoad)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
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
#######
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
#####
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

  ######
  plt.plot(LR_mean)
  plt.title(modelname+'_LR'+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize))
  plt.legend(['lr'], loc='upper left')
  plt.fill_between(range(epoch), LR_mean-LR_std, LR_mean+LR_std, color='orange', alpha = 0.5)

  plt.ylabel('lr')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_LR'+'_epoch_'+str(epoch+epochLoad)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()

  print("--- %s seconds ---" % (time.time() - start_time))

# for i in [0.1,0.25,0.5]:
#
#   for j in [128,256,512]:
#
#     for k in [None,0.0001,0.001,0.01]:

i=0
k=None
j=256

dirname='/media/datadrive/bsalafian/6FoldCrossSMILE'
modeldir='/home/baharsalafian/CNNSMILEGRUModels_JBHI_LossTest'
SaveResults='/home/baharsalafian/CNNSMILEGRUModels_JBHI_6FoldMetricsTest'
SaveHisResults='/home/baharsalafian/History_JBHI_LossTest_6FoldMetricsTest'

ModelTrain(dirname,SaveResults,SaveHisResults,modeldir,'CNNSMILE10times_Decay.5_lr.001_NoAct','CNNSMILE10times_lr.001_NoAct',seq_len=1,cnn=0,smile=1,diff=0,twodcnn=0,epoch=2, epochLoad=0,l2_size=k,drop_size=i,batchSize=j)

# dirname: directory where dataset is saved i.e. signals, MI estimation and labels
#SaveResults: directory you wish to save your results
#SaveHisResults: directory you wish to save the results of history for loss and accuracy during training the models
#'CNN10times':refers to the name that you wanna save for the trained model
# seq_len: refers to the memory length of the data, where in our implementation,for the models with GRU, it is 3 and for CNNs it's 1

# cnn, smile, diff and twodcnn: are control parameters that show which model is being trained, e.g. if you are training
# a 1D CNN model, the cnn should be set to 1. Therefore, cnn is for 1D CNN, smile for the CNN-SMILE architectures
# and twodcnn for 2D CNN models. diff is for the model that besides MI estimation, the difference of MI between consecutive blocks is also considered as
# additional features

# epoch and batchSize: refer to number of epoch and batch size you wanna use for training

# l2_size ,drop_size: the size for l2 regularization and drop layer, where in this model they are none and zero, respectively.
