# -*- coding: utf-8 -*-
"""Copy of CNNSMILEGRU_Bs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Fgyv5Og_PPfexz3ih-4uEJqa-PfQsKSA
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import statistics
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

#
# def SelectIndx(EDFNo,ind):
#
#   EDF=[]
#   EDFFiles=[]
#   Name=[]
#   EDF=PatientsEDFFile(dirname)
#   Name=PatientsName()
#
#   indx=[]
#   for j in range(len(ind)):
#     # print(j)
#     indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
#     indx.append(indices)
#
#   indtest=[]
#   indtrain=[]
#   for i in range(len(indx)):
#
#     for k in range(len(indx[i])):
#       # print(len(indx[i]))
#
#       if k==EDFNo:
#         indtest.append(indx[i][k])
#         # print(indtest)
#
#       else:
#         indtrain.append(indx[i][k])
#         # print(indtrain)
#
#   # indtest=np.concatenate(indtest,axis=0)
#   # indtrain=np.concatenate(indtrain,axis=0)
#   # print(len(indtest))
#   return indtest,indtrain
#
def dataloader(x,mi,y,spectrogram, seq_len, cnn,smile,diff):

  spec=[]
  MI_all=[]
  X=[]
  Y=[]
  MI_diff_all=[]
  # print('flag')
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
    print('flag')
    x, _ , real_y = create_sub_seq(x, seq_len, labels=real_y)
    MI, _, _ = create_sub_seq(MI, seq_len)

  if diff is not None:
    for j in range(MI.shape[0]-1):
      MI_diff.append(MI[j+1]-MI[j])
    MI_diff=np.array(MI_diff)
    MI=MI[1:]
    x=x[1:]
    real_y=real_y[1:]
    spectrogram=spectrogram[1:]



  return x,real_y,MI,MI_diff,spectrogram


def ReadMatFiles(dirname,dirname2,SaveResults,indx, testindx,fold_no,modelname,batchsize,epoch ,cnn,smile,diff,seq_len):



  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]
  ind=[]

  Ytest_all=[]
  Ypred_all=[]

  Xtest=[]
  Ytest=[]
  MItest=[]
  MItest_diff=[]
  print(Name[testindx])
    # print(j)
  ind = [i for i, elem in enumerate(EDF) if Name[testindx] in elem]
  #   ind.append(indices)
  print(ind)
  # ind=np.concatenate(ind,axis=0)
  cnt=0

  X_train=[]
  Y_train=[]
  MI_train=[]
  MI_train_diff=[]
  Spec_train=[]
  shapes=[]
  # print(EDF[ind[k]])
  for z in ind:
    mat=loadmat(os.path.join(dirname,EDF[int(z)]))
    x=mat['X_4sec']
    shape1=x.shape[0]
    shapes.append(shape1)
  MediumShape=min(shapes, key=lambda x:abs(x-statistics.median(shapes)))
  index = shapes.index(MediumShape)
  matfile_test=loadmat(os.path.join(dirname,EDF[int(ind[index])]))
  print('test ', EDF[int(ind[index])])
  # time.sleep(1)
  matfile_test2=loadmat(os.path.join(dirname2,EDF[int(ind[index])]))
  spectrogram_test=matfile_test2['spectogram']
  # print(EDF[int(ind[k][index])])
  xtest=matfile_test['X_4sec']
  ytest=matfile_test['Y_label_4sec']
  mitest=matfile_test['estimated_MI']
  X_test,Y_test,MI_test,MI_test_diff,Spec_test=dataloader(xtest,mitest,ytest,spectrogram_test,seq_len,cnn,smile,diff)
  # print(X_test.shape)
  for q in [p for p in ind if p != ind[index]]:
    matfile=loadmat(os.path.join(dirname,EDF[q]))
    matfile_test2=loadmat(os.path.join(dirname2,EDF[q]))
    spectrogram_train=matfile_test2['spectogram']
    print('train ',EDF[q])
    xtrain=matfile['X_4sec']
    ytrain=matfile['Y_label_4sec']
    mitrain=matfile['estimated_MI']
    Xtrain,Ytrain,MItrain,MI_difftrain,spectrogramtrain=dataloader(xtrain,mitrain,ytrain,spectrogram_train,seq_len,cnn,smile,diff)
    # print(Ytrain.shape)
    X_train.append(Xtrain)
    MI_train.append(MItrain)
    MI_train_diff.append(MI_difftrain)
    Y_train.append(Ytrain)
    Spec_train.append(spectrogramtrain)
  X_train=np.concatenate(X_train,axis=0)
  Y_train=np.concatenate(Y_train,axis=0)
  MI_train=np.concatenate(MI_train,axis=0)
  MI_train_diff=np.concatenate(MI_train_diff,axis=0)
  Spec_train=np.concatenate(Spec_train,axis=0)
  # ypred,model=FineTuning(SaveResults,modelname,fold_no,x1,y1,mi1,mi_diff1,batchsize,epoch,Xtest,MItest,MItest_diff,cnn,smile,diff)
  # Ypred_all.append(ypred)
  # Ytest_all.append(Ytest)
  return X_train,Y_train,MI_train, Spec_train,X_test,Y_test,MI_test,Spec_test

def Xtrain(cnn,smile,diff,twodcnn,spectrogram,gru,x,y,mi,spec):

  if cnn==1:
    X_train=x

  if gru==1:
    X_train=x

  if smile==1:
    X_train=[x,mi]

  # if diff==1:
  #   X_train=[x,mi,mi_diff]

  if twodcnn==1:
    x2d=x.reshape(x.shape[0],x.shape[2],x.shape[1],1)
    X_train=x2d

  if spectrogram==1:

    X_train=spec

  return X_train







def ModelTrain(dirname,dirname2,modeldir,SaveResults,SaveResultsTruePred,modelname,seq_len,cnn,smile,diff,twodcnn,spectrogram,indx,gru,epoch,l2_size,drop_size,batchSize):

  FoldNum=6
  kfold = KFold(n_splits=FoldNum, shuffle=False)
  start_time = time.time()
  fold_no=0

  savenamef1='fscore_'+modelname+'_'+indx+'_'+'epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)
  cfname='cfmat_'+modelname+'_'+indx+'_'+'epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)
  fprname='fpr_'+modelname+'_'+indx+'_'+'epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)
  tprname='tpr_'+modelname+'_'+indx+'_'+'epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)
  prname='PR_'+modelname+'_'+indx+'_'+'epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)
  rocname='ROC_'+modelname+'_'+indx+'_'+'epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)

  ind1=range(0,24)
  for th in [0.5]:

    fold_no=0
    fpr=[]
    tpr=[]
    PR=[]
    ROC=[]
    pred=[]
    act=[]
    fscore=[]

    for testindx in range(24):

      # testindx,trainindx=SelectIndx(i,ind1)

      if indx=='test':
        ind=testindx
      else:
        ind=trainindx

      Xt,Yt,MIt, Spectrain,Xtest,Ytest,MItest,Spectest=ReadMatFiles(dirname,dirname2,SaveResults,indx,testindx, fold_no,modelname,batchSize,epoch ,cnn,smile,diff,seq_len)


      # X=np.concatenate(x,axis=0)
      # Y=np.concatenate(y,axis=0)
      # MI_all=np.concatenate(mi,axis=0)
      # MI_diff_all=np.concatenate(mi_diff,axis=0)
      # spec=np.concatenate(spec,axis=0)
      fold_no=fold_no+1


      X_train=Xtrain(cnn,smile,diff,twodcnn,spectrogram,gru,Xtest,Ytest,MItest,Spectest)

      ModelName1=modelname+'_fold'+str(fold_no)+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.h5'

      model1=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))

      ypred=model1.predict(X_train)
      ypred_th = (ypred > th).astype(int)

      if gru==1:

        Ytest = Ytest[:, 2]
        ypred= ypred[:, 2]
        ypred_th= (ypred > th).astype(int)



      fpr1, tpr1, _ = roc_curve(Ytest, ypred)
      precision, recall, _ = precision_recall_curve(Ytest, ypred)
      PR1=auc(recall, precision)
      ROC1=roc_auc_score(Ytest,ypred)
      fpr.append(fpr1)
      tpr.append(tpr1)
      PR.append(PR1)
      ROC.append(ROC1)

      precision, recall, f1, _ = precision_recall_fscore_support(Ytest, ypred_th, average='weighted')

      pred.append(list(ypred_th))
      act.append(list(Ytest))
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
  ind1=range(0,24)
  ####################################
  for j in range(24):


      fold_no=0
      FoldNum=6
      kfold = KFold(n_splits=FoldNum, shuffle=False)

      fold_no=fold_no+1

      Xt,Yt,MIt, Spectrain,Xtest,Ytest,MItest,Spectest=ReadMatFiles(dirname,dirname2,SaveResults,indx,j, fold_no,modelname,batchSize,epoch ,cnn,smile,diff,seq_len)

      X_train=Xtrain(cnn,smile,diff,twodcnn,spectrogram,gru,Xtest,Ytest,MItest,Spectest)

      ModelName1=modelname+'_fold'+str(fold_no)+'_epoch_'+str(epoch)+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.h5'

      model1=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))

      ypred_plot=model1.predict(X_train)

      if gru==1:

        Ytest= Ytest[:, 2]
        ypred_plot= ypred_plot[:, 2]

      else:
        Ytest = Ytest

      # plt.subplot(12,2,fold_no)

      plt.plot(range(len(Ytest)), Ytest)
      plt.plot(range(len(Ytest)),ypred_plot)


      plt.suptitle(modelname+'_PatientNo_'+str(j+1)+'_epoch_'+str(epoch)+'_'+indx+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize))

      plt.savefig(SaveResultsTruePred+'/'+modelname+'_PatientNo_'+str(j+1)+'_epoch_'+str(epoch)+'_'+indx+'_l2Size_'+str(l2_size)+'_DropSize_'+str(drop_size)+'_batchsize_'+str(batchSize)+'.pdf', format='pdf', bbox_inches = 'tight')
      plt.clf()


  print("--- %s seconds ---" % (time.time() - start_time))

# for i in [0.1,0.25,0.5]:
#
#   for j in [128,256,512]:
#
#     for k in [None,0.0001,0.001,0.01]:

dirname='/media/datadrive/bsalafian/6FoldCrossSMILE'
dirname2='/media/datadrive/bsalafian/AllMatFiles'
modeldir='/home/baharsalafian/CNNSMILEGRUModels_24Models_JBHI_Losstest'

SaveResults='/home/baharsalafian/Results_24Models_JBHI_Losstest'
SaveResultsTruePred='/home/baharsalafian/TruePredPlots_24Models_JBHI_Losstest'

i=0
k=None
j=256

ModelTrain(dirname,dirname2,modeldir,SaveResults,SaveResultsTruePred,'2DCNN10times_lr.0001',seq_len=1,cnn=0,smile=0,diff=0,twodcnn=1,spectrogram=0,indx='test',gru=0,epoch=100,l2_size=k,drop_size=i,batchSize=j)

## '2DCNN10times' : name of the model that you wanna load
