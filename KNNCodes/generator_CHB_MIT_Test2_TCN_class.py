# -*- coding: utf-8 -*-
"""Copy of TCN_Example_Bahareh.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TcSy2-BXUmAPwJeWSyDy84Wq6-ZWfR-5
"""

from cProfile import label
import multiprocessing
import os
import sys
from tabnanny import check
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
from sklearn.utils import shuffle
import collections
# from parse_label import parse_event_labels
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
import ast
from wandb.keras import WandbCallback

# datagen = ImageDataGenerator(zca_whitening=True)
# from google.colab import drive
# drive.mount('/content/drive')

from keras_tcn import TCN
from SBRNN_Keras import ModelConfig, SBRNN_Detector, ModelConfigTCN
from data_gen_chb import SeizureDataGenerator


def PatientsName():

  Name=['chb01','chb02','chb03','chb04','chb05','chb06','chb07','chb08','chb09','chb10',
  'chb11','chb12','chb13','chb14','chb15','chb16','chb17','chb18','chb19','chb20','chb21',
  'chb22','chb23','chb24']
  return Name

def segment_data(loaddir,mat_name):

  
  info_mat = loadmat(os.path.join(loaddir, mat_name))

  data = info_mat['data']
  DenoisedSig = info_mat['DenoisedSig']
  DenoisedSigSeizure = info_mat['DenoisedSigSeizure']

  #print("shape of data is ", data.shape)
  #print("shape of DenoisedSig is ", DenoisedSig.shape)


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
  #print("out shape", X.shape)

  Ylabel=np.zeros((SigEnd-SigStart,1))
  Seizure_durarion = SiezureEnd - SiezureStart + 1

  Seizure_start_label = SiezureStart - SigStart 
  #print(Seizure_start_label)
  Ylabel[Seizure_start_label:Seizure_start_label+Seizure_durarion,0] =  1

  return X, Ylabel



def PatientsEDFFile(dirname):

  os.chdir(dirname)
  a=[]
  X=[]
  Y=[]
  b=[]
  c = []
  k=0
  for file in glob.glob("*.mat"):
    # print("this is file",file)
    split_file=file.split('.')
    # print(split_file[0])
    # split_file2=split_file[0].split('_')
    # b.append(split_file2[2]+'_'+split_file2[3]+'_'+split_file2[4]+'_'+split_file2[5])
    a.append(file)
    c.append(split_file[0])

      # print(a)
  return a,c
  
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

def ReadMatFiles(dirname,dir_csv,ind,category_name):

  EDF = []
  EDFFiles = []
  Name = []
  EDF,names = PatientsEDFFile(dirname)
  Name = PatientsName()
  Xfile=[]
  Yfile=[]
  c = 0


  X=[]
  Y=[]

  # print(ind)
  for k in range(len(ind)):

  
    csv_name = EDF[ind[k]].split('.')
    x, y = segment_data(dirname,EDF[ind[k]])

    print("shape of x",x.shape)
    print("shape of y", y.shape)
    path = os.path.join(dirname,EDF[ind[k]])

    df = pd.DataFrame(columns=['FileName', 'Label', 'index'])
    for j in range(x.shape[0]):
      df = df.append({'FileName':path , 'Label': y[j,:][0],'index': np.int64(j)},ignore_index=True)
    # df.to_csv(os.path.join(os.path.join(dir_csv,f'data_files_chb/{category_name}'), f'{csv_name[0]}.csv'))
    c = c + len(df) - 7
  
  print("this is the whole length", c)

    # X.append(x)
    # Y.append(y)
    #matfile=loadmat(os.path.join(dirname,EDF[ind[k]]))

  
  # X = np.concatenate(X,axis=0)
  # Y = np.concatenate(Y,axis=0)
  # Y = Y.T
  # print(X.shape)
  # print(Y.shape)
  # return X, Y

def SelectIndx(EDFNo,ind,dirname):

  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF,names=PatientsEDFFile(dirname)
  Name=PatientsName()

  indx=[]
  for j in range(len(ind)):
    # print(j)
    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
    indx.append(indices)

  indtest=[]
  indtrain=[]
  for i in range(len(indx)):

    for k in range(len(indx[i])):
      # print(len(indx[i]))

      if k==EDFNo:
        indtest.append(indx[i][k])
        # print(indtest)

      else:
        indtrain.append(indx[i][k])
        # print(indtrain)

  # indtest=np.concatenate(indtest,axis=0)
  # indtrain=np.concatenate(indtrain,axis=0)
  # print(len(indtest))
  return indtest,indtrain

class DataGenerator(keras.utils.Sequence):

  def __init__(self,data,batch_size,shuffle):
  #Initializing the values
    self.data  = data
    self.batch_size = batch_size
    self.list_IDs = np.arange(len(data))
    self.shuffle = shuffle
    self.on_epoch_end()

  def on_epoch_end(self):
    self.indexes = self.list_IDs
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __len__(self):
    return int(np.floor(len(self.data)/self.batch_size))

  def __getitem__(self, index):
    list_IDs_temp = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    # list_IDs_temp = [self.list_IDs[k] for k in index]
    X,y = self.__data_generation(list_IDs_temp)
    return X,y


  # def __segment_data(self,loaddir):

  #   info_mat = loadmat(loaddir)
  #   # data = info_mat['data']
  #   data_denoised = info_mat['DenoisedSig']
  #   # DenoisedSigSeizure = info_mat['DenoisedSigSeizure']

  #   #print("shape of data is ", data.shape)
  #   #print("shape of DenoisedSig is ", DenoisedSig.shape)


  #   # SigEnd = info_mat['Sig_end'][0][0]
  #   # SigStart = info_mat['Sig_start'][0][0]
  #   # SiezureStart = info_mat['Siezure_start'][0][0]
  #   # SiezureEnd = info_mat['Siezure_end'][0][0]
  #   # n_channels = 18
  #   Fs = 256

    

  #   # n = data_denoised.shape[1]//Fs

  #   output=[data_denoised[:,i:i + Fs] for i in range(0,data_denoised.shape[1], Fs)]

  #   X = np.array(output)
  #   # print("out shape", X.shape)

  #   # Ylabel=np.zeros((SigEnd-SigStart,1))
  #   # Seizure_durarion = SiezureEnd - SiezureStart + 1

  #   # Seizure_start_label = SiezureStart - SigStart 
  #   # #print(Seizure_start_label)
  #   # Ylabel[Seizure_start_label:Seizure_start_label+Seizure_durarion,0] =  1

  #   return X

  def __data_generation(self,list_IDs_temp):
    
    X = self.data.iloc[list_IDs_temp,0]
    X = np.array(X.values.tolist())

    y = self.data.iloc[list_IDs_temp,1]
    y = np.array(y.values.tolist())

    # print("this is x shape ", X.shape)
    # print("this is y shape", y.shape)

    return X, y


## The code for training TCN
def TCN_Model(len_seq, one_sec_samp, num_channel, num_filters, LR):

  seq_len = len_seq

  input_size = one_sec_samp

  num_chan = num_channel

  inps = keras.layers.Input(shape=(seq_len, num_chan, input_size), name="inp")

  n = seq_len * 2
  a = [i for i in range(1, n+1) if (math.log(i)/math.log(2)).is_integer()]
  dilation_list = tuple(a)

  tcn_out = TCN(nb_filters=num_filters, kernel_size=3,
                        nb_stacks=1, dilations=dilation_list,
                        padding='same', use_skip_connections=True,
                        dropout_rate=0.2, return_sequences=True,
                        activation='wavenet')(inps)

  #latent_large = Flatten()(tcn_out)
  #tcn_out = keras.layers.Lambda(lambda x: x[:, :, 0, :])(tcn_out)
  shape = tcn_out.get_shape().as_list()
  print(shape)
  tcn_out = tf.reshape(tcn_out, [-1, shape[1] , shape[2] * shape[3]])
  latent_small = TimeDistributed(Dense(128,activation='relu'))(tcn_out)
  probs = TimeDistributed(Dense(1,activation='sigmoid'))(latent_small)
  preds = tf.cast(probs > 0.5, tf.int16)

  model = keras.models.Model(inputs=inps, outputs=probs)
  model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=LR), metrics=["accuracy",f1_score,precision,recall2])
  #print(model)
  return model



def lr_exp_decay(epoch, lr):
  # k = math.sqrt(2)
  k=2

  if lr>0.00001:
    return lr / k
  
  else:
    return lr

#################### Train the model #############################
# def segment_data(loaddir):

#     info_mat = loadmat(loaddir)
#     data = info_mat['data']
#     DenoisedSig = info_mat['DenoisedSig']
#     DenoisedSigSeizure = info_mat['DenoisedSigSeizure']

#     #print("shape of data is ", data.shape)
#     #print("shape of DenoisedSig is ", DenoisedSig.shape)


#     SigEnd = info_mat['Sig_end'][0][0]
#     SigStart = info_mat['Sig_start'][0][0]
#     SiezureStart = info_mat['Siezure_start'][0][0]
#     SiezureEnd = info_mat['Siezure_end'][0][0]
#     n_channels = 18
#     Fs = 256

#     data_denoised = DenoisedSig[:,0:DenoisedSig.shape[1]-1]

#     n = data_denoised.shape[1]//Fs

#     output=[data_denoised[:,i:i + Fs] for i in range(0,data_denoised.shape[1], Fs)]

#     X = np.array(output)
#     #print("out shape", X.shape)

#     # Ylabel=np.zeros((SigEnd-SigStart,1))
#     # Seizure_durarion = SiezureEnd - SiezureStart + 1

#     # Seizure_start_label = SiezureStart - SigStart 
#     # #print(Seizure_start_label)
#     # Ylabel[Seizure_start_label:Seizure_start_label+Seizure_durarion,0] =  1

#     return X


modelname = 'TCN_CHB_MIT_On_Epoch_pkl_test2'
initLR = 0.001
num_epochs = 30
batch_size = 256

len_seq = 8
one_sec_samp = 256
num_channel = 18
num_filters = 16
LR = 0.001
tcn = 1
cnn = 0
two_dcnn = 0
gru = 0
len_seq_gru = 3

indx=range(0,24)

# print(model_tcn)

dir_csv = '/home/baharsalafian'
if not os.path.exists(os.path.join(dir_csv,'data_files_chb')):
    os.mkdir(os.path.join(dir_csv,'data_files_chb'))
if not os.path.exists(os.path.join(dir_csv,'data_files_chb/train')):
    os.mkdir(os.path.join(dir_csv,'data_files_chb/train')) 
if not os.path.exists(os.path.join(dir_csv,'data_files_chb/test')):
    os.mkdir(os.path.join(dir_csv,'data_files_chb/test')) 

dirname = '/media/datadrive/bsalafian/EEGDataLongDuration'
dir_csv = '/home/baharsalafian'

i = 0

testindx,trainindx=SelectIndx(i,indx,dirname)

# ReadMatFiles(dirname, dir_csv, trainindx, 'train')

# ReadMatFiles(dirname, dir_csv, testindx, 'test')

df_train = pd.read_pickle(os.path.join(os.path.join(dir_csv,'data_files_chb/train/train_list'), 'data_list.pkl'))
print(len(df_train))
df_test = pd.read_pickle(os.path.join(os.path.join(dir_csv,'data_files_chb/test/test_list'), 'data_list.pkl'))


start_time = time.time()
model = TCN_Model(len_seq, one_sec_samp, num_channel, num_filters, LR)
print(model)
params = {
'batch_size':256,
'shuffle':True}

train_gen = DataGenerator(df_train,**params)
valid_gen = DataGenerator(df_test,**params)


wandb.init(project = modelname+'_len_seq_'+str(len_seq)+'_num_filters_'+str(num_filters)+'_batch_size_'+str(batch_size)+'_num_epochs_'
                                +str(num_epochs), entity="bsalafian")



# model.fit_generator(train_generator,epochs=num_epochs,steps_per_epoch=321,
#                     callbacks=[WandbCallback(),LearningRateScheduler(lr_exp_decay)],
#                     validation_data=test_generator,validation_steps=121,
#                     use_multiprocessing=True, workers=64)
model.fit_generator(train_gen,epochs=num_epochs, callbacks=[WandbCallback()],
                    validation_data=valid_gen, verbose = 2, use_multiprocessing=True,workers=16)
wandb.finish()

print("--- %s seconds ---" % (time.time() - start_time))

