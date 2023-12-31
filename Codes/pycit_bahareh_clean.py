# -*- coding: utf-8 -*-
"""Pycit_Bahareh_Clean.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11-xTKCvME3tWvEgv4eTyaZIlQWLHM_5c
"""

# Commented out IPython magic to ensure Python compatibility.
# import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
from pycit.estimators import ksg_mi,bi_ksg_cmi,mixed_mi,mixed_cmi,bi_ksg_mi
from pycit.preprocessing import low_amplitude_noise
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
# import torch
# import torch.nn as nn
from tqdm.auto import tqdm, trange
from numpy.random import default_rng
# import torch.nn.functional as F
# from google.colab import drive
import gc
import time
# gc.collect()
# drive.mount('/content/drive')
# print(np.version.version)



def LoadData(dirname,indx):

    os.chdir(dirname)
    a=[]
    X=[]
    Y=[]
    k=0
    for file in glob.glob("*.mat"):

        a.append(file)
    # print(a)
    Name=[a[x] for x in indx]
    for i in range(len(Name)):

        matFile1 = h5py.File(os.path.join(dirname, Name[i]),'r')
        xx=matFile1.get('input')
        yy=matFile1.get('target')
        x1=np.array(xx)
        y1=np.array(yy)
        x= np.transpose(x1)
        y=np.transpose(y1)
        step=x.shape[0]
        X.append(x)
        Y.append(y)
        k=k+step

    xx = np.concatenate(X, axis=0)
    yy = np.concatenate(Y, axis=0)
    return xx,yy

def normalize(data):
  signal_max = np.max(data,axis=(0,1))
  ##########
  signal_min=np.min(data,axis=(0,1))
  den=signal_max-signal_min
  if data.ndim > 1:
    den[den == 0] = 1.
  elif den == 0:
    den = 1.

  return signal_max,signal_min,den

def standardize(data):
  mean1 = np.mean(data,axis=(0,1))
  stdv = np.std(data,axis=(0,1))

  if data.ndim > 1:
      stdv[stdv == 0] = 1.
  elif stdv == 0:
      stdv = 1.
  return mean1,stdv

# mi_test=bi_ksg_mi(x,y,5)
# print(mi_test)

from numpy.lib.stride_tricks import sliding_window_view

dirname='/home/baharsalafian/1DCNNDataset'

dir_mi_results ='/home/baharsalafian/KNNClean1Results'


vec_dim = 1
knn_k = 5
start_time = time.time()
labels = [0, 1]
for file_idex in range(24):
  X_train, Y_train = LoadData(dirname, [file_idex])

  Y_train = (Y_train > 0.2).astype(int)

  signal_max, signal_min, den= normalize(X_train)
##########
 # mean1,stdv=standardize(X_train)
###############

  #X_train=low_amplitude_noise(X_train, eps=1e-10)

  estimated_MI = np.zeros((len(labels), X_train.shape[2], X_train.shape[2]))
  estimated_MI_std = np.zeros((len(labels), X_train.shape[2], X_train.shape[2]))
  estimated_MI_per4Sec = np.zeros((len(labels), X_train.shape[0],X_train.shape[2], X_train.shape[2]))


  for j in range(X_train.shape[2]-1):
    for k in range(j+1, X_train.shape[2]):
        for l in labels:
          estimated_mi = []
          for i in tqdm(range(X_train.shape[0])):
            if Y_train[i] != l:
              continue

            X = X_train[i, :, j]/signal_max[j]
            Y = X_train[i, :, k]/signal_max[k]
            vec_X = sliding_window_view(X, vec_dim)
            vec_Y = sliding_window_view(Y, vec_dim)
            #print(all_x.shape)
            estimated_mi.append(bi_ksg_mi(vec_X, vec_Y, knn_k))
            estimated_MI_per4Sec[l,i,j,k]=estimated_mi[-1]





          estimated_MI[l,j,k]=np.mean(estimated_mi)
          estimated_MI_std[l,j,k]=np.std(estimated_mi)

          file_name = 'Pycit_MeanMI_Est_1NNperChan_fileidx_{}.mat'.format(file_idex)
          savemat(os.path.join(dir_mi_results, file_name), {"estimated_MI": estimated_MI})

          file_name2 = 'Pycit_StdMI_Est_1NNperChan_fileidx_{}.mat'.format(file_idex)
          savemat(os.path.join(dir_mi_results, file_name2), {"estimated_MI": estimated_MI_std})

          file_name3 = 'Pycit_MI_Est_per4Sec_1NNperChan_fileidx_{}.mat'.format(file_idex)
          savemat(os.path.join(dir_mi_results, file_name3), {"estimated_MI": estimated_MI_per4Sec})

          print("-----File_idx {}, Label {}, Chan1 {},  Chan2 {}, Mean Est MI {:.4f}, Std Est MI {:.5f} ".format
                (file_idex, l, j, k, estimated_MI[l,j,k], estimated_MI_std[l,j,k]))


  plt.subplot(1,2,1)
  plt.imshow(estimated_MI[0])
  plt.title('No Seizure')
  plt.subplot(1,2,2)
  plt.imshow(estimated_MI[1])
  plt.title('Seizure')
  plt.show()


print("--- %s seconds ---" % (time.time() - start_time))
