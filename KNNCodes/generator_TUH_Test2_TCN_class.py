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
from wandb.keras import WandbCallback

# datagen = ImageDataGenerator(zca_whitening=True)
# from google.colab import drive
# drive.mount('/content/drive')

from keras_tcn import TCN
from SBRNN_Keras import ModelConfig, SBRNN_Detector, ModelConfigTCN
from data_gen import SeizureDataGenerator

def PatientsEDFFile(dirname):

  os.chdir(dirname)
  a=[]
  X=[]
  Y=[]
  b=[]
  c = []
  k=0
  for file in glob.glob("*.pkl"):

    split_file=file.split('.')
    print(split_file[0])
    split_file2=split_file[0].split('_')
    b.append(split_file2[2]+'_'+split_file2[3]+'_'+split_file2[4]+'_'+split_file2[5])
    a.append(file)
    c.append(split_file[0])

      # print(a)
  return a,b,c
  
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


    test=[*focal_list[-5:],*generalized_list[-5:]]


    train=[*focal_list[0:focal_No],*generalized_list[0:gen_No]]
    #train=[*focal_list[-5:],*generalized_list[-5:]]

    return train,test

def find_indices(pid,name):

  index=[]
  for j in range(len(name)):
      index.append([i for i, e in enumerate(pid) if e == name[j]])

  return index

def Readpklfiles(dirname,index,n_chan,one_sec_samp,len_seq,len_seq_gru,tcn,cnn,two_dcnn,gru):

    main,pid= PatientsEDFFile(dirname)

    ind=find_indices(pid,index)
    index_v2=list(itertools.chain.from_iterable(ind))
    
    names = []
    size = []

    data=[]
    label=[]
    for i in range(len(index_v2)):

      path=os.path.join(dirname,main[index_v2[i]])
      names.append(path)
      
      pkl = pickle.load(open(path , 'rb'))

      x=pkl.data
      size.append(x)
    
    


    return names,size


## The code for training TCN
def TCN_Model(len_seq, one_sec_samp, num_channel, num_filters, LR):

  seq_len = len_seq

  input_size = one_sec_samp

  num_chan = num_channel

  inps = keras.layers.Input(shape=(seq_len, num_chan, input_size), name="inp")

  n = seq_len*2
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
new_dict = np.load('/home/baharsalafian/TUH_statistics/All_info.npy', allow_pickle='TRUE')
#focal_No=130
#gen_No=70

#focal_No=60
#gen_No=40

#focal_No=30
#gen_No=20

# focal_No = 30
# gen_No = 20




# print(model_tcn)

seizure_type_data = collections.namedtuple('seizure_type_data',
                                           ['patient_id', 'seizure_type', 'seizure_start', 'seizure_end', 'data',
                                           'new_sig_start', 'new_sig_end', 'original_sample_frequency','TSE_channels',
                                           'label_matrix', 'tse_label_matrix','lbl_channels','data_preprocess',
                                           'TSE_channels_preprocess','lable_matrix_preprocess','lbl_channels_preprocess',
                                           'data_segment','tse_label_segment','tse_timepoints'])





new_dict = np.load('/home/baharsalafian/TUH_statistics/All_info.npy', allow_pickle='TRUE')
#focal_No=130
#gen_No=70

#focal_No=60
#gen_No=40

#focal_No=30
#gen_No=20

focal_No = 15
gen_No = 5

train,test=train_test_split(new_dict,focal_No,gen_No)




dirname='/home/baharsalafian/TUH_Bahareh_experiment/v1.5.2/raw_seizures'
SaveHisResults='/home/baharsalafian/TUH_experiments_TCN_history'
ModelResults='/home/baharsalafian/TUH_experiments_TCN_models'


modelname = 'TCN_20_train_10_test_class_gen_8_workers'
initLR = 0.001
num_epochs = 3
batch_size = 256

len_seq = 8
one_sec_samp = 250
num_channel = 20
num_filters = 16
LR = 0.001
tcn = 1
cnn = 0
two_dcnn = 0
gru = 0
len_seq_gru = 3


# print(model_tcn)

seizure_type_data = collections.namedtuple('seizure_type_data',
                                           ['patient_id', 'seizure_type', 'seizure_start', 'seizure_end', 'data',
                                           'new_sig_start', 'new_sig_end', 'original_sample_frequency','TSE_channels',
                                           'label_matrix', 'tse_label_matrix','lbl_channels','data_preprocess',
                                           'TSE_channels_preprocess','lable_matrix_preprocess','lbl_channels_preprocess',
                                           'data_segment','tse_label_segment','tse_timepoints'])



dir_csv = '/home/baharsalafian'
if not os.path.exists(os.path.join(dir_csv,'data_files')):
    os.mkdir(os.path.join(dir_csv,'data_files'))
if not os.path.exists(os.path.join(dir_csv,'data_files/train')):
    os.mkdir(os.path.join(dir_csv,'data_files/train')) 
if not os.path.exists(os.path.join(dir_csv,'data_files/test')):
    os.mkdir(os.path.join(dir_csv,'data_files/test')) 


# create_csv_files(dirname,dir_csv,test,'test',num_channel,one_sec_samp,len_seq,len_seq_gru,tcn,cnn,two_dcnn,gru)
# create_csv_files(dirname,dir_csv,train,'train',num_channel,one_sec_samp,len_seq,len_seq_gru,tcn,cnn,two_dcnn,gru)

# train_data = load_samples(dir_csv,data_cat='train',temporal_stride=1,temporal_length=len_seq)
# test_data = load_samples(dir_csv,data_cat='test',temporal_stride=1,temporal_length=len_seq)

# print(len(train_data))
# print(len(test_data))

data_gen_obj = SeizureDataGenerator(dir_csv,temporal_stride=1,temporal_length=len_seq,shuffle=True)

train_generator = data_gen_obj.data_generator_modified(data_cat='train',num_csv=10,batch_size=256,shuffle=True)


test_generator = data_gen_obj.data_generator_modified(data_cat='test',num_csv=10,batch_size=256,shuffle=True)
# train_generator = data_generator(test_data,batch_size=batch_size,shuffle=True)

# test_generator = data_generator(test_data,batch_size=batch_size,shuffle=True)
start_time = time.time()

model = TCN_Model(len_seq, one_sec_samp, num_channel, num_filters, LR)
# wandb.init(project=modelname+'_len_seq_'+str(len_seq)+'_num_filters_'+str(num_filters)+'_batch_size_'+str(batch_size)+'_num_epochs_'
                                # +str(num_epochs), entity="bsalafian")
model.fit_generator(train_generator,epochs=num_epochs,steps_per_epoch=61,
                    callbacks=[LearningRateScheduler(lr_exp_decay)],
                    validation_data=test_generator,validation_steps=2,
                    use_multiprocessing=True, workers=8)

# model.fit_generator(train_generator,epochs=num_epochs,steps_per_epoch=61,
#                     callbacks=[WandbCallback(),LearningRateScheduler(lr_exp_decay)],
#                     validation_data=test_generator,validation_steps=2,
#                     use_multiprocessing=True, workers=8)
# model.save(ModelResults+'/'+modelname+'_len_seq_'+str(len_seq)+'_num_filters_'+str(num_filters)+'_batch_size_'+str(batch_size)+'_num_epochs_'
                                # +str(num_epochs)+'_Nb_patients_'+str(focal_No+gen_No)+'.h5')
# wandb.finish()

print("--- %s seconds ---" % (time.time() - start_time))

#print(train_steps)
#data_generator_check(names_test,batch_size,num_channel,tcn,cnn,two_dcnn,gru,len_seq,len_seq_gru)