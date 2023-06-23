# -*- coding: utf-8 -*-
"""Copy of TCN_Example_Bahareh.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TcSy2-BXUmAPwJeWSyDy84Wq6-ZWfR-5
"""

from cProfile import label
from distutils.command.bdist import show_formats
import multiprocessing
from operator import index
import os
import sys
from tabnanny import check
from matplotlib import test
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow as tf
import scipy
import h5py
import glob, os
from scipy.io import loadmat,savemat
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
from wandb.keras import WandbCallback
from collections import deque
from scipy.io import loadmat
import copy

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

def segment_data(loaddir):


  info_mat = loadmat(loaddir)

  # data = info_mat['data']
  DenoisedSig = info_mat['DenoisedSig']
  # DenoisedSigSeizure = info_mat['DenoisedSigSeizure']

  #print("shape of data is ", data.shape)
  # print("shape of DenoisedSig is ", DenoisedSig.shape)


  SigEnd = info_mat['Sig_end'][0][0]
  SigStart = info_mat['Sig_start'][0][0]
  SiezureStart = info_mat['Siezure_start'][0][0]
  SiezureEnd = info_mat['Siezure_end'][0][0]
  n_channels = 18
  Fs = 256

  data_denoised = DenoisedSig[:,0:DenoisedSig.shape[1]-1]

  # n = data_denoised.shape[1]//Fs

  output=[data_denoised[:,i:i + Fs] for i in range(0,data_denoised.shape[1], Fs)]

  X = np.array(output)
  print("out shape", X.shape)

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
  model.compile(optimizer=Adam(learning_rate=initLR),loss='binary_crossentropy',metrics=["accuracy",f1_score,precision,recall2,auc_pr])

  return model

def MeanStdVar(mylist):


  ListMean=np.mean(mylist,axis=0)
  ListStd=np.std(mylist)
  ListVar=np.var(mylist)

  return ListMean,ListStd,ListVar


def lr_exp_decay(epoch, lr):
  # k = math.sqrt(2)
  k=2
  if lr>0.00001:
    return lr / k
  else:
    return lr

def ReadMatFiles(dirname,all_img_dir,dir_csv,ind,category_name):

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
  df = pd.DataFrame(columns=['FileName', 'Label', 'index'])
  for k in range(len(ind)):

  
    csv_name = EDF[ind[k]].split('.')

    matfile = loadmat(os.path.join(dirname,EDF[ind[k]]))
 
    x = matfile['X_4sec']
    y = matfile['Y_label_4sec']
    # print(x[0,:,:].shape)
    y=np.transpose(y)

    start_idx = np.argmax(y>0)
    a = y == 1
    end_idx = len(a) - np.argmax(np.flip(a)) - 1
    real_y = np.zeros_like(y)
    real_y[start_idx:end_idx+1] = 1


    # x, y = segment_data(os.path.join(dirname,EDF[ind[k]]))

    print("shape of x",x.shape)
    print("shape of y", real_y.shape)
    # path = os.path.join(dirname,EDF[ind[k]])

    
    for j in range(x.shape[0]):
      savemat(os.path.join(all_img_dir,csv_name[0]+'_'+str(j)+'.mat'), {"data":x[j,:,:]})
      df = df.append({'FileName':csv_name[0]+'_'+str(j), 'Label': real_y[j,:][0]},ignore_index=True)
  df.to_pickle(os.path.join(os.path.join(dir_csv,f'data_files_chb_old/{category_name}'), f'{category_name}.pkl'))
    # c = c + len(df) - 7
  
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




def shuffle_data(samples):

  data = shuffle(samples)
  return data


def define_CNN(drop_size,initLR):
  l2_size=None
  vectorsize=18
  input_shape=(1024,18)
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

  denseout = Dense(denseSize)(flat)
  denseout2 = Dense(denseSize)(denseout)
  drop2=Dropout(0.5)(denseout2)
  output = Dense(1, activation='sigmoid')(drop2)
  # define a model with a list of two inputs
  model = Model(inputs=input1, outputs=output)
  model.compile(optimizer=Adam(learning_rate=initLR),loss='binary_crossentropy',metrics=["accuracy",f1_score,precision,recall2])
  return model

class My_Custom_Generator(keras.utils.Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size,shuffle) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.on_epoch_end()
    
  def on_epoch_end(self):
    # self.indexes = self.list_IDs
    if self.shuffle == True:
      self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    # print("this is type of y",type(batch_y))
    # # print("this is dtype of y",dtype(batch_y))
    # print("this is batch y",batch_y)
    x = np.array([loadmat('/home/baharsalafian/chb_all_images/' + str(file_name)+'.mat')['data'] for file_name in batch_x])
    # print("this is during data generator",idx)
    y = np.array(batch_y)
    # print("this is type of y",type(y))
    # print("this is dtype of y",dtype(batch_y))
    # print("this is batch y",y)
    y = np.expand_dims(y, axis=1)
    

    return x, y


############## train_generator oversample

class My_Custom_Generator_train(keras.utils.Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size,shuffle) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.on_epoch_end()
    
  def on_epoch_end(self):
    # self.indexes = self.list_IDs
    if self.shuffle == True:
      self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)
  
  
  def __getitem__(self, idx) :

    ind_one = [i for i,x in enumerate(self.labels) if x== 1]
    ind_zero = [i for i,x in enumerate(self.labels) if x== 0]

    x_one = self.image_filenames[ind_one]
    x_zero = self.image_filenames[ind_zero]

    y_one = self.labels[ind_one]
    y_zero = self.labels[ind_zero]
    

    selected_idx_one = np.random.choice(len(x_one),self.batch_size//2)
 
    batch_x_zero = x_zero[idx * self.batch_size//2 : (idx+1) * self.batch_size//2]
    batch_x_one = x_one.loc[x_one.index[selected_idx_one]]
    

    batch_y_one = y_one.loc[y_one.index[selected_idx_one]]
    batch_y_zero = y_zero[idx * self.batch_size//2 : (idx+1) * self.batch_size//2]
  
    x_one_total = np.array([loadmat('/home/baharsalafian/chb_all_images/' + str(file_name)+'.mat')['data'] for file_name in batch_x_one])
    x_zero_total = np.array([loadmat('/home/baharsalafian/chb_all_images/' + str(file_name)+'.mat')['data'] for file_name in batch_x_zero])
    x_total = np.concatenate((x_zero_total,x_one_total),axis=0)
    
    # print("this is during data generator",idx)
    y_one_total = np.array(batch_y_one)
    y_zero_total = np.array(batch_y_zero)

    y_one_total = np.expand_dims(y_one_total, axis=1)
    y_zero_total = np.expand_dims(y_zero_total, axis=1)
    y_total = np.concatenate((y_zero_total,y_one_total),axis=0)

    return x_total, y_total

#################### Train the model #############################


modelname = 'CNN_CHB_MIT_new_loader_oversample_both_shuffle'

indx=range(0,24)

# print(model_tcn)

dir_csv = '/home/baharsalafian'
if not os.path.exists(os.path.join(dir_csv,'data_files_chb_old')):
    os.mkdir(os.path.join(dir_csv,'data_files_chb_old'))
if not os.path.exists(os.path.join(dir_csv,'data_files_chb_old/train')):
    os.mkdir(os.path.join(dir_csv,'data_files_chb_old/train'))
if not os.path.exists(os.path.join(dir_csv,'data_files_chb_old/test')):
    os.mkdir(os.path.join(dir_csv,'data_files_chb_old/test'))

if not os.path.exists(os.path.join(dir_csv,'data_files_chb/test/test_list')):
    os.mkdir(os.path.join(dir_csv,'data_files_chb/test/test_list'))
if not os.path.exists(os.path.join(dir_csv,'data_files_chb/test/test_list_v2')):
    os.mkdir(os.path.join(dir_csv,'data_files_chb/test/test_list_v2'))


if not os.path.exists(os.path.join(dir_csv,'data_files_chb/train/train_list_v2')):
    os.mkdir(os.path.join(dir_csv,'data_files_chb/train/train_list_v2'))




i = 0
all_img_dir = '/home/baharsalafian/chb_all_images'

dirname = '/media/datadrive/bsalafian/6FoldCrossSMILE'
dir_csv = '/home/baharsalafian'
testindx,trainindx=SelectIndx(i,indx,dirname)

# load_samples_modified_V2_seperate(dir_csv,data_cat='train',data_list='train_list',shuffle=True,temporal_length=8,temporal_stride=1)



initLR =  0.0001
cnn=0
smile=0
smile_new_arch=1
spectrogram=0
cnnsmilespec=0
twodcnn=0
diff=0
num_epochs = 100
batch_size=256
SeqLen=1
l2_size=None
drop_size=0

model = define_CNN(drop_size,initLR)
start_time = time.time()

df_train = pd.read_pickle(os.path.join(os.path.join(dir_csv,'data_files_chb_old/train'), 'train.pkl'))
train_file_name = df_train.iloc[:,0] 
# print(type(np.array(train_file_name.tolist())))


train_label = df_train.iloc[:,1]


df_test = pd.read_pickle(os.path.join(os.path.join(dir_csv,'data_files_chb_old/test'), 'test.pkl'))
# print(df_test.index.tolist())
test_file_name = df_test.iloc[:,0] 
test_label = df_test.iloc[:,1]


# train_file_name = np.save(os.path.join(dir_csv,'data_files_chb_old/train'), 'X_train_file_name.npy')

# # print(type(np.array(train_file_name.tolist())))
# train_label = np.save(os.path.join(dir_csv,'data_files_chb_old/train'), 'Y_train.npy')

# test_file_name = np.save(os.path.join(dir_csv,'data_files_chb_old/test'), 'X_test_file_name.npy')

# # print(type(np.array(train_file_name.tolist())))
# test_label = np.save(os.path.join(dir_csv,'data_files_chb_old/test'), 'Y_test.npy')
# np.save(os.path.join(os.path.join(dir_csv,'data_files_chb_old/train'), 'X_train_file_name.npy'), np.array(train_file_name.tolist()))
# np.save(os.path.join(os.path.join(dir_csv,'data_files_chb_old/train'), 'Y_train.npy'), np.array(train_label.tolist()))
# print(len(df_train.iloc[:,1]))


# ind_one=[i for i,x in enumerate(test_label) if x== 1]
# out = test_label[ind_one]
# print(out)
# print(ind_one)
# np.save(os.path.join(os.path.join(dir_csv,'data_files_chb_old/test'), 'X_test_file_name.npy'), np.array(test_file_name.tolist()))
# np.save(os.path.join(os.path.join(dir_csv,'data_files_chb_old/test'), 'Y_test.npy'), np.array(test_label.tolist()))

train_gen = My_Custom_Generator_train(train_file_name, train_label, batch_size,shuffle=True)
validation_gen = My_Custom_Generator(test_file_name, test_label, batch_size,shuffle=True)

train_steps = int(len(df_train.iloc[:,1])//batch_size)
test_steps = int(len(df_test.iloc[:,1])//batch_size)

print("this is train steps",train_steps)
wandb.init(project = modelname+'_batch_size_'+str(batch_size)+'_num_epochs_'
                                +str(num_epochs)+'LR'+str(initLR), entity="bsalafian")




model.fit_generator(train_gen,steps_per_epoch=train_steps,
epochs=num_epochs,
callbacks=[WandbCallback(),LearningRateScheduler(lr_exp_decay)],
validation_data=validation_gen,validation_steps=test_steps,
verbose = 2, use_multiprocessing=True,workers=16)

# model.fit_generator(train_gen,
# epochs=num_epochs,
# callbacks=[WandbCallback(),LearningRateScheduler(lr_exp_decay)],
# validation_data=validation_gen,
# verbose = 2, use_multiprocessing=True,workers=16)

wandb.finish()

print("--- %s seconds ---" % (time.time() - start_time))
# ### let's combine all samples #######

# x_test = np.array(df_test.iloc[:,0])
# print(x_test.shape)
# print('test')
# data_list = load_samples_modified(dir_csv,data_cat='test',data_list='test_list',shuffle=True,temporal_length=8,temporal_stride=1)

# print(len(data_list))
# df = pd.DataFrame(columns=['FileName', 'Label', 'index'])
# for i in range(len(data_list)):
#   print(i)
#   # print(data_list[i][2])
#   x = segment_data(data_list[i][0][0])
#   x = x[data_list[i][2]]
#   # print(x.shape)
#   df = df.append({'FileName': x , 'Label': data_list[i][1],'ind': data_list[i][2]},ignore_index=True)
# df.to_pickle(os.path.join(os.path.join(dir_csv,f'data_files_chb/test/test_list'), 'data_list.pkl'))



# print(data_list)

# ReadMatFiles(dirname, all_img_dir, dir_csv,  trainindx, 'train')
# ReadMatFiles(dirname, all_img_dir, dir_csv, testindx, 'test')