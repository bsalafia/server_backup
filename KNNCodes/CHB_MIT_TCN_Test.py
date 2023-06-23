from cProfile import label
import multiprocessing
import os
import sys
from tabnanny import check

#from oversample_24PatientsTest_CustomMetrics_W_B import train_generator
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
import time
#from wandb.keras import WandbCallback

# datagen = ImageDataGenerator(zca_whitening=True)
# from google.colab import drive
# drive.mount('/content/drive')

from keras_tcn import TCN
from SBRNN_Keras import ModelConfig, SBRNN_Detector, ModelConfigTCN

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
  #y_true = y_true[:, 3]
  #y_pred = y_pred[:, 3]

  return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

def auc_pr(y_true, y_pred):

  #y_true = y_true[:, 3]
  #y_pred = y_pred[:, 3]

  return tf.py_function(AUCPR, (y_true, y_pred), tf.double)



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


def create_sub_seq(nn_input, len_ss, labels=None):
    """
    This function creates all sub sequences for the batch
    """
    # print(nn_input.shape)
    n_seq = nn_input.shape[0]

    len_seq = nn_input.shape[1]

    # print(n_seq)
    # print(len_seq)

    n_ss = len_seq - len_ss + 1
    new_labels = []
    if nn_input.ndim == 3:
      new_inp = np.zeros((n_ss*n_seq,len_ss,nn_input.shape[2]))
    elif nn_input.ndim == 4:

      new_inp = np.zeros((n_ss*n_seq,len_ss,nn_input.shape[2], nn_input.shape[3]))

    # print(new_inp.shape)
    if labels is not None:
        dim_labels = labels.shape
        if len(dim_labels) == 2:
            new_labels = np.zeros((n_ss*n_seq, len_ss))

            # print(new_labels.shape)
        elif len(dim_labels) == 3:
            new_labels = np.zeros((n_ss * n_seq, len_ss, dim_labels[2]))
    k = 0
    for i in range(n_seq):
        for j in range(n_ss):
            new_inp[k, :, :] = nn_input[i, j:j + len_ss, :]
            if labels is not None:
                if len(dim_labels) == 2:
                    new_labels[k, :] = labels[i, j:j + len_ss]
                elif len(dim_labels) == 3:
                    new_labels[k, :, :] = labels[i, j:j + len_ss, :]
            k += 1

    return new_inp, n_ss, new_labels

def creat_sub_seq_models(x,y,n_chan,tcn,cnn,two_dcnn,gru,len_seq,len_seq_gru):

  if tcn == 1:

    x_new = np.reshape(x, (1, x.shape[0], -1))
    x_new, _,y_new = create_sub_seq(x_new, len_seq, y)
    x_new = np.reshape(x_new, (x_new.shape[0], x_new.shape[1], n_chan, one_sec_samp))
    last_element = y_new.shape[1]

  if cnn == 1:

    x_new = np.reshape(x, (1, x.shape[0], -1))
    x_new, _,y_new = create_sub_seq(x_new, len_seq, y)
    x_new = np.reshape(x_new, (x_new.shape[0], x_new.shape[1], n_chan, one_sec_samp))
    last_element = y_new.shape[1]
    x_new = np.reshape(x_new, (x_new.shape[0], x_new.shape[1]*one_sec_samp,n_chan))
    y_new = y_new[:,last_element-1]
    y_new = np.expand_dims(y_new, axis=1)

  if gru == 1:

    x_new = np.reshape(x_new, (1,x_new.shape[0], x_new.shape[1]*one_sec_samp,n_chan))
    y_new = y_new[:,last_element-1]
    y_new = np.expand_dims(y_new, axis=0)
    x_new, _,y_new = create_sub_seq(x_new, len_seq_gru, y_new)

  if two_dcnn == 1:

    x_new = np.reshape(x, (1, x.shape[0], -1))
    x_new, _,y_new = create_sub_seq(x_new, len_seq, y)
    x_new = np.reshape(x_new, (x_new.shape[0], x_new.shape[1], n_chan, one_sec_samp))
    last_element = y_new.shape[1]
    x_new = np.reshape(x_new, (x_new.shape[0], n_chan, x_new.shape[1]*one_sec_samp))
    y_new = y_new[:,last_element-1]
    y_new = np.expand_dims(y_new, axis=1)

  return x_new, y_new

def SelectIndx(EDFNo,ind,dirname):

  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
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

def ReadMatFiles(dirname,ind):

  EDF = []
  EDFFiles = []
  Name = []
  EDF = PatientsEDFFile(dirname)
  Name = PatientsName()
  Xfile=[]
  Yfile=[]



  X=[]
  Y=[]

  # print(ind)
  for k in range(len(ind)):

    print(EDF[ind[k]])
    x, y = segment_data(loaddir,EDF[ind[k]])
    X.append(x)
    Y.append(y)
    #matfile=loadmat(os.path.join(dirname,EDF[ind[k]]))

  
  X = np.concatenate(X,axis=0)
  Y = np.concatenate(Y,axis=0)
  Y = Y.T
  return X, Y

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

def validation_batches(x,y,batch_size):

  total_sample=x.shape[0]
  selected_idx = np.random.choice(total_sample,batch_size)
  x_one_batch=x[selected_idx]
  y_one_batch=y[selected_idx]
  return x_one_batch,y_one_batch

def validation_generator(x,y,batch_size):

  while True:
    x_batch,y_batch=validation_batches(x,y,batch_size)
    yield x_batch,y_batch

def lr_exp_decay(epoch, lr):
  # k = math.sqrt(2)
  k=2

  if lr>0.00001:
    return lr / k
  
  else:
    return lr
def split(x,y):

  ind_one=[i for i,x in enumerate(y) if (x== 1).all()]
  ind_zero=[i for i,x in enumerate(y) if (x== 0).all()]

  y_zero=y[ind_zero]
  y_one=y[ind_one]

  x_zero=x[ind_zero]
  x_one=x[ind_one]

  
  return x_zero,y_zero,x_one,y_one

def make_batches(x_one,x_zero,y_one,y_zero,mini_batch):

  total_sample=x_one.shape[0]
  total_sample_zero=x_zero.shape[0]
  # print(len(x_zero) // 128)

  selected_idx = np.random.choice(total_sample,mini_batch)
  selected_idx_zero= np.random.choice(total_sample_zero,mini_batch)

  x_zero_batch= x_zero[selected_idx_zero]
  
  y_zero_batch = y_zero[selected_idx_zero]
  
  x_one_batch=x_one[selected_idx]
  y_one_batch=y_one[selected_idx]
 
  x_batch=np.concatenate((x_zero_batch, x_one_batch), axis=0)
  y_batch=np.concatenate((y_zero_batch, y_one_batch), axis=0)
  

  return x_batch, y_batch


def train_generator(x_one,x_zero,y_one,y_zero,mini_batch):
  while True:
    x_batch,y_batch=make_batches(x_one,x_zero,y_one,y_zero,mini_batch)
    yield x_batch,y_batch


loaddir = '/media/datadrive/bsalafian/EEGDataLongDuration'

#x, y = segment_data(loaddir)

initLR = 0.001
num_epochs = 20
num_filters = 16
len_seq = 4
one_sec_samp = 256
num_channel = 18

LR = 0.001
tcn = 1
cnn = 0
two_dcnn = 0
gru = 0
len_seq_gru = 3
drop_size = 0
model = TCN_Model(len_seq, one_sec_samp, num_channel, num_filters, LR)
batch_size = 256
#x_new, y_new = creat_sub_seq_models(x,y,num_channel,tcn,cnn,two_dcnn,gru,len_seq,len_seq_gru)

indx=range(0,24)
start_time = time.time()
mini_batch=int(batch_size//2)

modelname = 'TCN_CHB_MIT'

i = 0
testindx,trainindx=SelectIndx(i,indx,loaddir)

x_train, y_train = ReadMatFiles(loaddir,trainindx)
x_test, y_test = ReadMatFiles(loaddir,testindx)

x_new_train, y_new_train = creat_sub_seq_models(x_train,y_train,num_channel,tcn,cnn,two_dcnn,gru,len_seq,len_seq_gru)
x_new_test, y_new_test = creat_sub_seq_models(x_test,y_test,num_channel,tcn,cnn,two_dcnn,gru,len_seq,len_seq_gru)

train_steps=int(x_new_train.shape[0]//batch_size)
test_steps=int(x_new_test.shape[0]//batch_size)

x_zero, y_zero, x_one, y_one = split(x_new_train, y_new_train)
wandb.init(project=modelname+'_len_seq_'+str(len_seq)+'_num_filters_'+str(num_filters)+'_batch_size_'+str(batch_size)+'_num_epochs_'
                                +str(num_epochs), entity="bsalafian")

TrainGen = train_generator(x_one,x_zero,y_one,y_zero,mini_batch)
start_time = time.time()
history = model.fit_generator(TrainGen,steps_per_epoch=train_steps,epochs=num_epochs,
                      verbose=2,
                      callbacks=[WandbCallback(),LearningRateScheduler(lr_exp_decay)],
                      validation_data=validation_generator(x_new_test, y_new_test,batch_size),
                      validation_steps=test_steps)

print(x_new_test.shape)
print(y_new_test.shape)
print("--- %s seconds ---" % (time.time() - start_time))