# -*- coding: utf-8 -*-
from cProfile import label
from cmath import infj, pi
from distutils.util import split_quoted
import os
import itertools
from posixpath import split
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
from collections import Counter
import tensorflow as tf
import scipy
import h5py
import glob, os
from scipy.io import loadmat
import math
from keras.models import Model
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# from keras.datasets import mnist
from sklearn.metrics import plot_precision_recall_curve,roc_curve,roc_auc_score,auc
from sklearn.metrics import precision_recall_fscore_support,precision_recall_curve
import re
from tensorflow import keras
from tensorflow.keras.regularizers import l2, l1_l2
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,TimeDistributed, GRU,Concatenate
from tensorflow.keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, GlobalAveragePooling1D,MaxPooling1D,AveragePooling1D
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
# from keras.optimizers import Adam
# from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D, GlobalAveragePooling1D,Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
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
import gc
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import csv
import wandb
from wandb.keras import WandbCallback

import collections

from scipy.io import savemat,loadmat



def PatientsEDFFile(dirname):

  os.chdir(dirname)
  a=[]
  X=[]
  Y=[]
  b=[]
  pid = []
  k=0
  for file in glob.glob("*.mat"):


    path=os.path.join(dirname,file)
    a.append(path)

  return a
def segment_data(loaddir):

  loaddir = '/media/datadrive/bsalafian/EEGDataLongDuration'
  info_mat = loadmat(os.path.join(loaddir, 'chb01_03_1.mat'))

  data = info_mat['data']
  DenoisedSig = info_mat['DenoisedSig']
  DenoisedSigSeizure = info_mat['DenoisedSigSeizure']

  print("shape of data is ", data.shape)
  print("shape of DenoisedSig is ", DenoisedSig.shape)


  SigEnd = info_mat['Sig_end'][0][0]
  SigStart = info_mat['Sig_start'][0][0]
  SiezureStart = info_mat['Siezure_start'][0][0]
  SiezureEnd = info_mat['Siezure_end'][0][0]
  n_channels = 18
  Fs = 256

  data_denoised = DenoisedSig[:,0:DenoisedSig.shape[1]-1]

  n = data_denoised.shape[1]//Fs

  output=[data_denoised[:,i:i + Fs] for i in range(0,data_denoised.shape[1], Fs)]

  X = np.array(output)
  print("out shape", X.shape)

  Ylabel=np.zeros((SigEnd-SigStart,1))
  Seizure_durarion = SiezureEnd - SiezureStart + 1

  Seizure_start_label = SiezureStart - SigStart 
  print(Seizure_start_label)
  Ylabel[Seizure_start_label:Seizure_start_label+Seizure_durarion,0] =  1

  return X, Ylabel

  