# -*- coding: utf-8 -*-
"""CorrelationCheck.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_E_Nec8S7PCpZog8eY6ucGzoRLB7cmfF
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
# from pycit.estimators import ksg_mi,bi_ksg_cmi,mixed_mi,mixed_cmi,bi_ksg_mi
# from pycit.preprocessing import low_amplitude_noise
import sys
import scipy
import h5py
import glob, os
from scipy.io import loadmat,savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn import preprocessing
from keras import regularizers
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
from numpy.random import default_rng
import torch.nn.functional as F

import gc
import time
# from numpy.lib.stride_tricks import sliding_window_view
from numba import jit, njit, prange
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr,spearmanr
# from google.colab import drive
# drive.mount('/content/drive')

# seed(1)
# # prepare data
# data1 = 20 * randn(1000) + 100
# data2 = data1 + (10 * randn(1000) + 50)
# # calculate Pearson's correlation
# corr, _ = pearsonr(data1, data2)
# print('Pearsons correlation: %.3f' % corr)

# ####################
# corr, _ = spearmanr(data1, data2)
# print('Spearmans correlation: %.3f' % corr)

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

# def LoadData(dirname,Name):
#
#   matFile1 = loadmat(os.path.join(dirname, Name))
#   DenoisedSig=matFile1['DenoisedSig']
#   Fs=matFile1['Fs']
#   Siezure_start=matFile1['Siezure_start']
#   Siezure_end=matFile1['Siezure_end']
#   Sig_start=matFile1['Sig_start']
#   Sig_end=matFile1['Sig_end']
#   DenoisedSigSeizure= np.transpose(DenoisedSig)
#
#   return DenoisedSig,Fs[0][0],Siezure_start[0][0],Siezure_end[0][0],Sig_start[0][0],Sig_end[0][0]

def LoadData(dirname,indx):

    os.chdir(dirname)
    a=[]
    X=[]
    Y=[]
    k=0
    for file in glob.glob("*.mat"):

        a.append(file)
    print(a)
    Name=[a[x] for x in indx]
    print(Name)
    for i in range(len(Name)):

        matFile1 = h5py.File(os.path.join(dirname, Name[i]),'r')
        # matFile1 = loadmat(os.path.join(dirname, Name[i]))
        print(matFile1)
        xx=matFile1.get('input')
        yy=matFile1.get('target')
        x1=np.array(xx)
        y1=np.array(yy)
        print(x1.shape)
        x= np.transpose(x1)
        y=np.transpose(y1)
        # step=x.shape[0]
        X.append(x)
        Y.append(y)
        # k=k+step

    xx = np.concatenate(X, axis=0)
    yy = np.concatenate(Y, axis=0)
    return xx,yy

def Label(dirname,Name,WindowSize1,WindowSize2):

  DenoisedSig,Fs,Siezure_start,Siezure_end,Sig_start,Sig_end = LoadData(dirname,Name)

  n_channels= DenoisedSig.shape[0]

  # X=np.zeros((np.int64(Sig_end)-np.int64(Sig_start)-WindowSize+1,n_channels,WindowSize*Fs))

  # Ylabel=np.zeros((np.int64(Sig_end)-np.int64(Sig_start)-WindowSize+1,1))

  n1=1
  n2=1+WindowSize1
  s=Siezure_start-Sig_start+1
  e=Siezure_end-Sig_start+1
  t1 = 0
  t2 = WindowSize1*Fs
  X=[]
  Ylabel=[]
  Ylabel_4sec=[]
  Sup=DenoisedSig.shape[1]
  k=0

  while t2 <= Sup:


    X.append(DenoisedSig[:,t1:min(t2,Sup)])

    if  (n2<=s or n1>=e):
      Ylabel.append(0)

    elif (n1<s and n2<e):

      Ylabel.append(min(n2-s,e-s,WindowSize1)/WindowSize1)

    elif (n1>=s and n2<=e):
      Ylabel.append(1)

    elif n2>=e:
      Ylabel.append(min(e-n1,e-s)/WindowSize1)

    n11=n2-WindowSize2

    if  (n2<=s or n11>=e):

      Ylabel_4sec.append(0)

    elif (n11<s and n2<e):

      Ylabel_4sec.append(min(n2-s,e-s,WindowSize2)/WindowSize2)

    elif (n11>=s and n2<=e):

      Ylabel_4sec.append(1)

    elif n2>=e:

      Ylabel_4sec.append(min(e-n11,e-s)/WindowSize2)

    if t2+Fs > Sup:
      X=np.array(X)
      X_4sec=X[:,:,-1024::]

    t2 = t2 + Fs
    t1 = t1 + Fs
    n2=n2+1
    n1=n1+1
    k  = k + 1

  X=np.array(X)
  Ylabel=np.array(Ylabel)
  Ylabel_4sec=np.array(Ylabel_4sec)
  return X,Ylabel,X_4sec,Ylabel_4sec

from scipy.io import savemat
dirname='/media/datadrive/bsalafian/1DCNNDataset'
savedir='/home/baharsalafian/PearsonMean/'

WindowSize1=32

WindowSize2=4
labels = [0, 1]

EDFFiles=PatientsEDFFile(dirname)
start_time = time.time()
for file_idx in [0,1,2,3,4,5,6,7,8]:
  X_train,Y_train=LoadData(dirname, [file_idx])
  signal_max = np.max(X_train,axis=(0,1))
  # X_train,Y_train,_,_=Label(dirname,EDFFiles[file_idx],WindowSize1,WindowSize2)

  # Y_train = (Y_train > 0.03).astype(int)
  # Ylabel_4sec=(Ylabel_4sec>0.03).astype(int)

  # X_train=np.transpose(X_train,(0, 2, 1))

  estimated_MI = np.zeros((len(labels), X_train.shape[2], X_train.shape[2]))

  for l in labels:

   for j in range(X_train.shape[2]-1):

    for k in range(j+1, X_train.shape[2]):

      estimates=[]
      # all_x = []
      # all_y = []

      for i in range(X_train.shape[0]):

        if Y_train[i] != l:
          continue

        X = X_train[i, :, j]/signal_max[j]
        Y = X_train[i, :, k]/signal_max[k]

        corr, _ = pearsonr(X,Y)
        estimates.append(corr)

      # estimates = np.concatenate(estimates)
      estimated_MI[l,j,k] = np.mean(np.array(estimates))
      # time.sleep(0.5)
      print("-----File_idx {}, Label {}, Chan1 {},  Chan2 {}, Estimated MI {:.4f}".format(file_idx, l, j, k, estimated_MI[l,j,k]))

  # Name=EDFFiles[file_idx].split('.')
  savemat(os.path.join(savedir, 'MI_RollWindow_PearsonMean_AllFiles_Normal'+str(file_idx)+'.mat'), {"estimated_MI": estimated_MI})
  # print("-----File_idx {}, Label {}, Chan1 {},  Chan2 {}, Estimated MI {:.4f}".format(file_idx, l, j, k, estimated_MI[l,j,k]))

  print("--- %s seconds ---" % (time.time() - start_time))