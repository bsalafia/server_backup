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
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import f1_score,plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve,roc_curve,roc_auc_score,auc
from sklearn.metrics import precision_recall_fscore_support,precision_recall_curve
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
from keras.callbacks import LearningRateScheduler
import math
import time
from keras import backend as K
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
  model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=["accuracy", tf.keras.metrics.AUC(curve="ROC",name='auc_roc'),tf.keras.metrics.AUC(curve="PR",name='auc_pr'),f1])
  return model

def PatientsName():

  Name=['chb01','chb02','chb03','chb04','chb05','chb06','chb07','chb08','chb09','chb10',
  'chb11','chb12','chb13','chb14','chb15','chb16','chb17','chb18','chb19','chb20','chb21',
  'chb22','chb23','chb24']

  return Name


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
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))



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

#####################
class MetricsHistory(tf.keras.callbacks.Callback):

  def __init__(self, train, validation=None):
      super(MetricsHistory, self).__init__()
      self.validation = validation
      self.train = train

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

def ModelTrain(dirname,dirname2,SaveResults,SaveHisResults,modeldir,modelname,modelnameLoad,batchsize,epoch,epochLoad):


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

    xtrain, ytrain=ReadMatFiles(dirname,dirname2,trainindx)

    from sklearn.utils import class_weight
    cw = class_weight.compute_class_weight('balanced', np.unique(ytrain),np.ravel(ytrain))
    print(cw)
    class_weights = {0:cw[0], 1: cw[1]}

    def lr_exp_decay(epoch, lr):
      # k = math.sqrt(2)

      k=2

      if lr>0.00001:
        return lr / k

      else:
        return lr


    xtest, ytest=ReadMatFiles(dirname,dirname2,testindx)

    model=Conv_BN_Act_Pool(filtNo1=120,filtNo2=160,filtsize=5,input_shape=(1024,18),activation='relu',PoolSize=3,denseSize=8)
    # ModelName1=modelnameLoad+'_fold'+str(fold_no)+'_epoch_'+str(epochLoad)+'.h5'
    #
    # model=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))
    history=model.fit(xtrain, ytrain, validation_data=(xtest,ytest), batch_size=batchsize, epochs=epoch, class_weight=class_weights,callbacks=[LearningRateScheduler(lr_exp_decay, verbose=2)])
    model.save(SaveResults+'/'+modelname+'_fold'+str(fold_no)+'_epoch_'+str(epoch+epochLoad)+'.h5')
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

###########################################
  loss_mean,loss_std,_=MeanStdVar(loss)
  loss_val_mean,loss_val_std,_=MeanStdVar(loss_val)
  acc_mean,acc_std,_=MeanStdVar(acc)
  acc_val_mean,acc_val_std,_=MeanStdVar(acc_val)
  LR_mean,LR_std,_=MeanStdVar(LR)

  f1_mean,f1_std,_=MeanStdVar(f1)
  f1_val_mean,f1_val_std,_=MeanStdVar(f1_val)

  PR_mean,PR_std,_=MeanStdVar(PR)
  PR_val_mean,PR_val_std,_=MeanStdVar(PR_val)

  ROC_mean,ROC_std,_=MeanStdVar(ROC)
  ROC_val_mean,ROC_val_std,_=MeanStdVar(ROC_val)

  np.savez(os.path.join(SaveHisResults, 'HistoryRes_'+modelname+'_epoch_'+str(epoch+epochLoad)), loss=loss, loss_val=loss_val, accuracy=acc, accuracy_val=acc_val,LR=LR,LR1=LR, PR1=PR, PR_val1=PR_val,ROC1=ROC, ROC1_val=ROC_val, f11=f1, f11_val=f1_val)
  plt.plot(acc_mean)
  plt.plot(acc_val_mean)
  plt.title(modelname+'_accuracy'+'_epoch_'+str(epoch+epochLoad))
  plt.legend(['train', 'test'], loc='upper left')
  plt.fill_between(range(epoch), acc_mean-acc_std, acc_mean+acc_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), acc_val_mean-acc_val_std, acc_val_mean+acc_val_std, color='orange', alpha = 0.5)
  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_accuracy'+'_epoch_'+str(epoch+epochLoad)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()

  plt.plot(loss_mean)
  plt.plot(loss_val_mean)
  plt.title(modelname+'_loss'+'_epoch_'+str(epoch+epochLoad))
  plt.legend(['train', 'test'], loc='upper left')
  plt.fill_between(range(epoch), loss_mean-loss_std, loss_mean+loss_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), loss_val_mean-loss_val_std, loss_val_mean+loss_val_std, color='orange', alpha = 0.5)
  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_loss'+'_epoch_'+str(epoch+epochLoad)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()

#######
  plt.plot(f1_mean)
  plt.plot(f1_val_mean)
  plt.title(modelname+'_F1 score'+'_epoch_'+str(epoch))
  plt.legend(['train', 'test'], loc='upper left')

  plt.fill_between(range(epoch), f1_mean-f1_std, f1_mean+f1_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), f1_val_mean-f1_val_std, f1_val_mean+f1_val_std, color='orange', alpha = 0.5)

  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_F1 score'+'_epoch_'+str(epoch)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()
#####
  plt.plot(PR_mean)
  plt.plot(PR_val_mean)
  plt.title(modelname+'_PR'+'_epoch_'+str(epoch))
  plt.legend(['train', 'test'], loc='upper left')

  plt.fill_between(range(epoch), PR_mean-PR_std, PR_mean+PR_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), PR_val_mean-PR_val_std, PR_val_mean+PR_val_std, color='orange', alpha = 0.5)

  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_PR'+'_epoch_'+str(epoch)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()
######
  plt.plot(ROC_mean)
  plt.plot(ROC_val_mean)
  plt.title(modelname+'_ROC'+'_epoch_'+str(epoch))
  plt.legend(['train', 'test'], loc='upper left')

  plt.fill_between(range(epoch), ROC_mean-ROC_std, ROC_mean+ROC_std, color='blue', alpha = 0.5)
  plt.fill_between(range(epoch), ROC_val_mean-ROC_val_std, ROC_val_mean+ROC_val_std, color='orange', alpha = 0.5)

  plt.ylabel('%')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_ROC'+'_epoch_'+str(epoch)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()
########

  plt.plot(LR_mean)
  plt.title(modelname+'_LR'+'_epoch_'+str(epoch+epochLoad))
  plt.legend(['lr'], loc='upper left')
  plt.fill_between(range(epoch), LR_mean-LR_std, LR_mean+LR_std, color='orange', alpha = 0.5)

  plt.ylabel('lr')
  plt.xlabel('epoch')
  plt.savefig(SaveHisResults+'/'+'history_'+modelname+'_LR'+'_epoch_'+str(epoch+epochLoad)+'.pdf', format='pdf', bbox_inches = 'tight')
  plt.clf()

  print("--- %s seconds ---" % (time.time() - start_time))

dirname2='/media/datadrive/bsalafian/6FoldCrossSMILE'
dirname='/home/baharsalafian/SpectogramResultsSyncMI_Nobutter'
modeldir='/home/baharsalafian/CNNSMILEGRUModels_JBHI_LossTest'

SaveResults='/home/baharsalafian/CNNSMILEGRUModels_JBHI_6FoldMetricsTest'
SaveHisResults='/home/baharsalafian/History_JBHI_LossTest_6FoldMetricsTest'

ModelTrain(dirname,dirname2,SaveResults,SaveHisResults,modeldir,'CNN_Spectogram10times_Decay.5_lr.001','CNN_Spectogram10times_lr.00001',batchsize=256,epoch=100,epochLoad=0)