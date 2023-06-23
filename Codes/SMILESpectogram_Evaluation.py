# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zpZgsmLBEDcbNXdQvguvdzo_74Rbtfya
"""

# -*- coding: utf-8 -*-
"""Copy of CNNSMILEGRU_Bs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Fgyv5Og_PPfexz3ih-4uEJqa-PfQsKSA
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
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



  # X=np.concatenate(X,axis=0)
  # Xcnn=np.concatenate(Xcnn,axis=0)
  # Y=np.concatenate(Y,axis=0)
  # MI_all=np.concatenate(MI_all,axis=0)
  #
  # print(X.shape)
  # print(Xcnn.shape)
  # print(Y.shape)
  # print(MI_all.shape)

  return Xcnn,X, Y,MI_all








def ModelTrain(dirname,dirname2,modeldir,SaveResults,SaveResultsTruePred,modelname,indx,gru,Noepoch,batchSize):

  FoldNum=6
  kfold = KFold(n_splits=FoldNum, shuffle=False)
  start_time = time.time()
  fold_no=0

  savenamef1='fscore_'+modelname+'_'+indx+'_'+'epoch_'+str(Noepoch)+'_batchsize_'+str(batchSize)
  cfname='cfmat_'+modelname+'_'+indx+'_'+'epoch_'+str(Noepoch)+'_batchsize_'+str(batchSize)
  fprname='fpr_'+modelname+'_'+indx+'_'+'epoch_'+str(Noepoch)+'_batchsize_'+str(batchSize)
  tprname='tpr_'+modelname+'_'+indx+'_'+'epoch_'+str(Noepoch)+'_batchsize_'+str(batchSize)
  prname='PR_'+modelname+'_'+indx+'_'+'epoch_'+str(Noepoch)+'_batchsize_'+str(batchSize)
  rocname='ROC_'+modelname+'_'+indx+'_'+'epoch_'+str(Noepoch)+'_batchsize_'+str(batchSize)
  #
  for th in [0.5]:


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

      Xcnn,X, Y,MI_all=ReadMatFiles(dirname,dirname2,ind)

      Xcnn=np.concatenate(Xcnn,axis=0)
      X=np.concatenate(X,axis=0)
      Y=np.concatenate(Y,axis=0)
      MI_all=np.concatenate(MI_all,axis=0)

      fold_no=fold_no+1


      ModelName1=modelname+'_fold'+str(fold_no)+'.h5'

      model1=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))

      ypred=model1.predict([Xcnn,X,MI_all])
      ypred_th = (ypred > th).astype(int)

      if gru==1:


        Y = Y[:, 2]
        ypred= ypred[:, 2]
        ypred_th= (ypred > th).astype(int)



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



    np.save(os.path.join(SaveResults, fprname),  fpr)
    np.save(os.path.join(SaveResults, tprname),  tpr)
    np.save(os.path.join(SaveResults, prname), PR)
    np.save(os.path.join(SaveResults, rocname),  ROC)
  #

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
      xcnn,x,y,mi=ReadMatFiles(dirname,dirname2,ind)

      ModelName1=modelname+'_fold'+str(fold_no)+'.h5'
      model1=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))
      ypred_plot=model1.predict([xcnn[j],x[j],mi[j]])
      if gru==1:
        y = y[j][:, 2]
        ypred_plot= ypred_plot[:, 2]
      else:
        y = y[j]
      plt.subplot(3,2,fold_no)
      plt.plot(range(len(y)), y)
      plt.plot(range(len(y)),ypred_plot)
    plt.suptitle(modelname+'_EDFNo_'+str(j+1)+'_epoch_'+str(Noepoch)+'_batchsize_'+str(batchSize)+'_'+indx)
    plt.savefig(SaveResultsTruePred+'/'+modelname+'_EDFNo_'+str(j+1)+'_epoch_'+str(Noepoch)+'_batchsize_'+str(batchSize)+'_'+indx+'.pdf', format='pdf', bbox_inches = 'tight')
    plt.clf()
  print("--- %s seconds ---" % (time.time() - start_time))




dirname='/home/baharsalafian/SpectogramResultsSyncMI_Nobutter'
dirname2='/home/baharsalafian/6FoldCrossSMILE'
modeldir='/home/baharsalafian/CNNSMILESpectrogramModels'

SaveResults='/home/baharsalafian/Results_Evaluation_SMILESpectogram'
SaveResultsTruePred='/home/baharsalafian/TruePredPlot_SMILESpectogram'

ModelTrain(dirname,dirname2,modeldir,SaveResults,SaveResultsTruePred,'CNN_Spectogram_SMILE',indx='train',gru=0,Noepoch=100,batchSize=256)
