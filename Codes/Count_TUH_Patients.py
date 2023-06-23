
from cmath import pi
from distutils.util import split_quoted
import os
from posixpath import split
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import tensorflow as tf
import scipy
import h5py
import glob, os
from scipy.io import loadmat
import math
from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
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


def PatientsEDFFile(dirname):

  os.chdir(dirname)
  a=[]
  X=[]
  Y=[]
  k=0
  for file in glob.glob("*.pkl"):

      split_file=file.split('.')
      a.append(split_file[0])
      # print(a)
  return a

dirname='/home/baharsalafian/TUH_Bahareh_experiment/v1.5.2/raw_seizures'
a= PatientsEDFFile(dirname)
pid=[]
header=[]
p_count=0
FNSZ=0
GNSZ=0
CPSZ=0
TNSZ=0
ABSZ=0
SPSZ=0
MYSZ=0
TCSZ=0
ATSZ=0
CNSZ=0
header_focal=[]
header_generalized=[]

for i in range(len(a)):

    split_a=a[i].split('_')
    
    x=split_a[3] not in pid
    
    if x:

        pid.append(split_a[3])
        p_count+=1

        if split_a[5]=='FNSZ':
            FNSZ+=1
            header_focal.append('pid_'+split_a[3]+'_type_'+split_a[5])

        elif split_a[5]=='GNSZ':
            GNSZ+=1
            header_generalized.append('pid_'+split_a[3]+'_type_'+split_a[5])

        elif split_a[5]=='CPSZ':
            CPSZ+=1
            header_focal.append('pid_'+split_a[3]+'_type_'+split_a[5])

        elif split_a[5]=='TNSZ':
            TNSZ+=1
            header_generalized.append('pid_'+split_a[3]+'_type_'+split_a[5])

        elif split_a[5]=='ABSZ':
            ABSZ+=1
            header_generalized.append('pid_'+split_a[3]+'_type_'+split_a[5])

        elif split_a[5]=='SPSZ':
            SPSZ+=1
            header_focal.append('pid_'+split_a[3]+'_type_'+split_a[5])

        elif split_a[5]=='MYSZ':
            MYSZ+=1
            header_generalized.append('pid_'+split_a[3]+'_type_'+split_a[5])

        elif split_a[5]=='TCSZ':
            TCSZ+=1
            header_generalized.append('pid_'+split_a[3]+'_type_'+split_a[5])

        elif split_a[5]=='ATSZ':
            ATSZ+=1
            header_generalized.append('pid_'+split_a[3]+'_type_'+split_a[5])


        elif split_a[5]=='CNSZ':
            CNSZ+=1
            header_generalized.append('pid_'+split_a[3]+'_type_'+split_a[5])

print("num of focal", len(header_focal))
print("num of gen", len(header_generalized))
header=header_focal
with open('/home/baharsalafian/TUH_statistics/Focal.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

header=header_generalized
with open('/home/baharsalafian/TUH_statistics/Generalized.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

my_dict={'p_count':p_count, 'pid':pid , 'FNSZ':FNSZ, 'GNSZ':GNSZ, 'ABSZ':ABSZ ,'TNSZ':TNSZ ,'CNSZ':CNSZ, 'TCSZ':TCSZ ,'ATSZ':ATSZ, 'MYSZ':MYSZ, 'CPSZ':CPSZ, 'SPSZ':SPSZ, 

'header_focal':header_focal,'header_generalized':header_generalized}

np.save('/home/baharsalafian/TUH_statistics/All_info.npy', my_dict)


new_dict = np.load('/home/baharsalafian/TUH_statistics/All_info.npy', allow_pickle='TRUE')
print(new_dict.item())
print("sss")