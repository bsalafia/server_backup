
from cProfile import label
import multiprocessing
import os
from tabnanny import check
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
import collections
from parse_label import parse_event_labels
# %matplotlib inline
# from keras.datasets import mnist
from sklearn.metrics import plot_precision_recall_curve,roc_curve,roc_auc_score,auc
from sklearn.metrics import precision_recall_fscore_support,precision_recall_curve
import itertools
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
import pickle
import re 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import InputLayer
from keras.layers import Input
import time
import gc
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import math
import wandb
from wandb.keras import WandbCallback
# datagen = ImageDataGenerator(zca_whitening=True)
# from google.colab import drive
# drive.mount('/content/drive')

def PatientsEDFFile(dirname):

  os.chdir(dirname)
  a=[]
  X=[]
  Y=[]
  b=[]
  k=0
  for file in glob.glob("*.pkl"):

      split_file=file.split('.')
      split_file2=split_file[0].split('_')
      b.append(split_file2[2]+'_'+split_file2[3]+'_'+split_file2[4]+'_'+split_file2[5])
      a.append(file)

      # print(a)
  return a,b
def recall2(y_true, y_pred):

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall_m = true_positives / (possible_positives + K.epsilon())
  return recall_m

def precision(y_true, y_pred):

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision_m = true_positives / (predicted_positives + K.epsilon())
  return precision_m

def AUCPR(y_true, y_pred):

  precision, recall, _ = precision_recall_curve(y_true, y_pred)

  PR1=auc(recall, precision)

  return PR1
# def f1_score(y_true, y_pred):
  # precision_m = precision(y_true, y_pred)
  # recall_m = recall(y_true, y_pred)
  # return 2*((precision_m*recall_m)/(precision_m+recall_m+K.epsilon()))
def f1_score(y_true, y_pred,threshold_shift=0):

  y_pred = K.clip(y_pred, 0, 1)
  # shifting the prediction threshold from .5 if needed
  y_pred_bin = K.round(y_pred + threshold_shift)
  tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
  fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
  fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
  precision1 = tp / (tp + fp)
  recall1 = tp / (tp + fn)
  p = precision1
  r = recall1
  return 2 * ((p * r) / (p + r))

def auc_roc(y_true, y_pred):

  return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

def auc_pr(y_true, y_pred):

  return tf.py_function(AUCPR, (y_true, y_pred), tf.double)


def  Conv_BN_Act_Pool(filtNo,filtsize1,filtsize2,input1,activation,PoolSize,l2_size,drop_size):

  conv1 = Conv1D(filtNo,filtsize1,kernel_regularizer=l2(l2_size))(input1)
  conv2 = Conv1D(filtNo, filtsize2,kernel_regularizer=l2(l2_size))(conv1)
  BN=BatchNormalization(axis=-1)(conv2)
  ActFunc=Activation(activation)(BN)
  pool1=MaxPooling1D(pool_size=PoolSize)(ActFunc)
  # out=Dropout(drop_size)(pool1)

  return pool1

def define_CNN(drop_size,initLR):
  l2_size=None
  input_shape=(1000,20)
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

  denseout = Dense(100)(flat)
  denseout2 = Dense(50)(denseout)
  drop2=Dropout(0.5)(denseout2)
  output = Dense(1, activation='sigmoid')(drop2)
  # define a model with a list of two inputs
  model = Model(inputs=input1, outputs=output)
  model.compile(optimizer=Adam(learning_rate=initLR),loss='binary_crossentropy',metrics=["accuracy",f1_score,auc_roc,precision,recall2,auc_pr])
  return model


def MeanStdVar(mylist):

  ListMean=np.mean(mylist,axis=0)
  ListStd=np.std(mylist)
  ListVar=np.var(mylist)

  return ListMean,ListStd,ListVar

def lr_exp_decay(epoch, lr):
  # k = math.sqrt(2)
  k=2
  if epoch<100:
    return lr
  elif lr>0.00001 & epoch>100:
    return lr / k



def train_test_split(new_dict,focal_No,gen_No):

    train=[]
    test=[]

    focal_list=new_dict.item()['header_focal']
    
    generalized_list=new_dict.item()['header_generalized']
    

    test=[*focal_list[-10:],*generalized_list[-10:]]


    train=[*focal_list[0:focal_No],*generalized_list[0:gen_No]]

    return train,test

def find_indices(pid,name):
    index=[]
    for j in range(len(name)):
        index.append([i for i, e in enumerate(pid) if e == name[j]])
    
    return index

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



def ReadPklfiles(dirname,index, seq_len=1):

    main,pid= PatientsEDFFile(dirname)

    ind=find_indices(pid,index)
    index_v2=list(itertools.chain.from_iterable(ind))


    data=[]
    label=[]
    for i in range(len(index_v2)):
        
        path=os.path.join(dirname,main[index_v2[i]])
        #print(path)
        pkl = pickle.load(open(path , 'rb'))
        #print(path)
        x=pkl.data_segment
        y=pkl.tse_label_segment

        
        start_idx = np.argmax(y>0)
        a = y == 1
        end_idx = len(a) - np.argmax(np.flip(a)) - 1
        real_y = np.zeros_like(y)

        real_y[start_idx:end_idx+1] = 1


        if seq_len > 1:
            real_y = np.expand_dims(real_y, axis=0)
            x = np.expand_dims(x, axis=0)
     
            # print(MI.shape)
            x, _ , real_y = create_sub_seq(x, seq_len, labels=real_y)
           

        data.append(x)
        label.append(real_y)
    
    data=np.concatenate(data,axis=0)
    label=np.concatenate(label,axis=0)

    return data,label
