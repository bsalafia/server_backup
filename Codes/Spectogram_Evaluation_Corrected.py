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
os.environ["CUDA_VISIBLE_DEVICES"]="4"
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
#################################
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

  X_final=[]
  Y_final=[]
  MI_diff_final=[]
  MI_final=[]

  X_plot=[]
  Y_plot=[]
  MI_all_plot=[]
  MI_diff_all_plot=[]

  for j in list(indx):
    print(j)
    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
    ind.append(indices)

    print(indices)

    # time.sleep(2)

  # ind=np.concatenate(ind,axis=0)
  print(len(ind[0]))
  for k in range(len(ind)):
    for q in range(len(ind[k])):

    # print(ind[k])
      matfile=loadmat(os.path.join(dirname,EDF[int(ind[k][q])]))
      print(EDF[int(ind[k][q])])
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

      X_plot.append(x)
      Y_plot.append(real_y)
      MI_all_plot.append(MI)
      MI_diff_all_plot.append(MI_diff)

    X=np.concatenate(X,axis=0)
    Y=np.concatenate(Y,axis=0)
    MI_all=np.concatenate(MI_all,axis=0)
    MI_diff_all=np.concatenate(MI_diff_all,axis=0)

    X_final.append(X)
    Y_final.append(Y)
    MI_diff_final.append(MI_all)
    MI_final.append(MI_diff_all)

    MI_all=[]
    X=[]
    Y=[]
    MI_diff_all=[]


  #
  # print(X.shape)
  # print(Y.shape)
  # print(MI_all.shape)
  # print(MI_diff_all.shape)

  return X_final, Y_final, MI_final, MI_diff_final,X_plot,Y_plot,MI_all_plot,MI_diff_all_plot






#####################
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

  X_final=[]
  Y_final=[]

  X_plot=[]
  Y_plot=[]

  # print(ind)
  for j in list(indx):

    # print(j)
    indices = [i for i, elem in enumerate(EDF2) if Name[j] in elem]
    ind.append(indices)

  # ind=np.concatenate(ind,axis=0)
  for k in range(len(ind)):

    for q in range(len(ind[k])):


      matfile2=loadmat(os.path.join(dirname2,EDF2[int(ind[k][q])]))
      Name2=EDF2[ind[k][q]].split('.')

      # print(Name2)
      # time.sleep(2)
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

      X_plot.append(x)
      Y_plot.append(real_y)

    X=np.concatenate(X,axis=0)
    Y=np.concatenate(Y,axis=0)

    X_final.append(X)
    Y_final.append(Y)

    X=[]
    Y=[]




  # X=np.concatenate(X,axis=0)
  # Y=np.concatenate(Y,axis=0)

  # print(X.shape)
  # print(Y.shape)

  return X_final, Y_final,X_plot,Y_plot








def ModelTrain(dirname,dirname2,modeldir,SaveResults,SaveResultsTruePred,modelname,indx,gru,Noepoch,batchSize):

  FoldNum=6
  kfold = KFold(n_splits=FoldNum, shuffle=False)
  start_time = time.time()
  fold_no=0

  savenamef1='fscore_'+modelname+'_'+indx+'_'+'epoch_'+str(Noepoch)
  cfname='cfmat_'+modelname+'_'+indx+'_'+'epoch_'+str(Noepoch)
  fprname='fpr_'+modelname+'_'+indx+'_'+'epoch_'+str(Noepoch)
  tprname='tpr_'+modelname+'_'+indx+'_'+'epoch_'+str(Noepoch)
  prname='PR_'+modelname+'_'+indx+'_'+'epoch_'+str(Noepoch)
  rocname='ROC_'+modelname+'_'+indx+'_'+'epoch_'+str(Noepoch)

  precname='Precision_'+modelname+'_'+indx+'_'+str(Noepoch)

  recname='Recall_'+modelname+'_'+indx+'_'+str(Noepoch)

  #
  for th in [0.5]:


    fold_no=0
    AUC_PR=[]
    AUC_ROC=[]
    Precision=[]
    Recall=[]
    f1score=[]
    cnf_matrix=[]

    for trainindx, testindx in kfold.split(range(24)):



      if indx=='test':
        ind=testindx
      else:
        ind=trainindx

      x, y,_,_=ReadMatFiles(dirname,dirname2,ind)

      fold_no=fold_no+1
      fpr=[]
      tpr=[]
      PR=[]
      ROC=[]
      pred=[]
      act=[]
      fscore=[]
      prec=[]
      rec=[]

      ModelName1=modelname+'_fold'+str(fold_no)+'_initLR_0.01'+'_epoch_'+str(Noepoch)+'_batchsize_'+str(batchSize)+'.h5'
      model1=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))

      for k in range(len(x)):
        X=x[k]
        Y=y[k]

        ypred=model1.predict(X)
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
        prec.append(precision)
        rec.append(recall)

      AUC_PR.append(np.mean(np.array(PR),axis=0))
      AUC_ROC.append(np.mean(np.array(ROC),axis=0))

      Precision.append(np.mean(np.array(prec),axis=0))
      Recall.append(np.mean(np.array(rec),axis=0))
      f1score.append(np.mean(np.array(fscore),axis=0))
      pred1=np.concatenate(pred,axis=0)
      act1=np.concatenate(act,axis=0)
      cnf_matrix.append(confusion_matrix(act1, pred1))

  np.save(os.path.join(SaveResults, savenamef1+str(th)),  f1score)
  np.save(os.path.join(SaveResults, cfname+str(th)),  cnf_matrix)

  # np.save(os.path.join(SaveResults, fprname),  fpr)
  # np.save(os.path.join(SaveResults, tprname),  tpr)
  np.save(os.path.join(SaveResults, prname), PR)
  np.save(os.path.join(SaveResults, rocname),  ROC)
  np.save(os.path.join(SaveResults, precname),Precision)
  np.save(os.path.join(SaveResults, recname), Recall)

  #
  #
  # ####################################



  #

  ####################################
  fold_no=0
  FoldNum=6
  cnt=0
  kfold = KFold(n_splits=FoldNum, shuffle=False)
  for trainindx, testindx in kfold.split(range(24)):
    if indx=='test':
      ind=testindx
    else:
      ind=trainindx
    fold_no=fold_no+1
    for j in [0,8,11]:

      _,_,x, y=ReadMatFiles(dirname,dirname2,ind)

      ModelName1=modelname+'_fold'+str(fold_no)+'_initLR_0.01'+'_epoch_'+str(Noepoch)+'_batchsize_'+str(batchSize)+'.h5'
      model1=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))
      ypred_plot=model1.predict(x[j])
      if gru==1:
        y1 = y[j][:, 2]
        ypred_plot= ypred_plot[:, 2]
      else:
        y1 = y[j]
      cnt=cnt+1
      plt.subplot(6,3,cnt)
      plt.plot(range(len(y1)), y1)
      plt.plot(range(len(y1)),ypred_plot)
  plt.suptitle(modelname+'_EDFNo_'+str(j+1)+'_epoch_'+str(Noepoch)+'_batchsize_'+str(batchSize)+'_'+indx+'_fold_'+str(fold_no))
  plt.savefig(SaveResultsTruePred+'/'+modelname+'_epoch_'+str(Noepoch)+'_'+indx+'.pdf', format='pdf', bbox_inches = 'tight')
    # plt.clf()
  print("--- %s seconds ---" % (time.time() - start_time))

  #
  #




dirname='/home/baharsalafian/SpectogramResultsSyncMI_Nobutter'

dirname2='/media/datadrive/bsalafian/6FoldCrossSMILE'
modeldir='/home/baharsalafian/Custom_model_class_weight_6fold'

SaveResults='/home/baharsalafian/Results_JBHI_6FoldMetrics_classweight'

SaveResultsTruePred='/home/baharsalafian/TruePredPlots_JBHI_6FoldMetrics_classweight'



ModelTrain(dirname,dirname2,modeldir,SaveResults,SaveResultsTruePred,'Spectrogram_6Fold_class_weight',indx='test',gru=0,Noepoch=150,batchSize=256)
