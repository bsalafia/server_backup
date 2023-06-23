
#%% 
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



#print('min and max ', min(nb_files_per_patient), max(nb_files_per_patient))

def PatientsEDFFile(dirname):

  os.chdir(dirname)
  a=[]
  X=[]
  Y=[]
  b=[]
  pid = []
  k=0
  for file in glob.glob("*.pkl"):

      split_file=file.split('.')
      split_file2=split_file[0].split('_')
      b.append(split_file2[2]+'_'+split_file2[3]+'_'+split_file2[4]+'_'+split_file2[5])
      pid.append(split_file2[3])

      


      path=os.path.join(dirname,file)
      a.append(path)
  counter_object = Counter(pid)
  keys = counter_object.keys()
  num_values = len(keys)
  
        # print(a)
  return a, num_values, keys, b

dirname='/home/baharsalafian/TUH_Bahareh_experiment/v1.5.2/raw_seizures'
files, num_patients, pids, file_name = PatientsEDFFile(dirname)

seizure_type_data = collections.namedtuple('seizure_type_data',
                                           ['patient_id', 'seizure_type', 'seizure_start', 'seizure_end', 'data', 
                                           'new_sig_start', 'new_sig_end', 'original_sample_frequency','TSE_channels', 
                                           'label_matrix', 'tse_label_matrix','lbl_channels','data_preprocess',
                                           'TSE_channels_preprocess','lable_matrix_preprocess','lbl_channels_preprocess',
                                           'data_segment','tse_label_segment','tse_timepoints'])


savedir = '/home/baharsalafian/TUH_statistics'

Focal_list = ['FNSZ','CPSZ','SPSZ']

Generalized_list = ['GNSZ','TNSZ','ABSZ','MYSZ','TCSZ','ATSZ','CNSZ']


##### plot files per patient
#print(files2)
nb_files_per_patient = []
seizure_min = []
seizure_max = []
focal_no = []
general_no = []
for pid in pids:
  seizure_duration = []
  lists = [i for i in file_name if pid in i]
  nb_files_per_patient.append(len(lists))
  f_c = 0 
  g_c = 0
  for j in range(len(lists)):

    split_name = lists[j].split('_')

    if split_name[3] in Focal_list:

      f_c = f_c + 1

    else:
      g_c = g_c + 1 
  
  focal_no.append(f_c)
  general_no.append(g_c)


#savemat(os.path.join(savedir, 'focal_generalized_info.mat'), {'focal_no': focal_no,
 #'general_no': general_no })

print('min , max , total', min(nb_files_per_patient), max(nb_files_per_patient),
sum(nb_files_per_patient))

info = loadmat(os.path.join(savedir, 'focal_generalized_info.mat'))

focal_no = info['focal_no']
general_no = info['general_no']

print(focal_no.T[:].shape)

r = np.arange(len(focal_no.T))
width = 0.75

print(focal_no.T[:,0])
  
plt.plot(focal_no.T[:,0])
plt.xlabel("Patient ID")
plt.ylabel('Focal Number')
plt.title("Focal seizures/patients")


plt.show()
res_focal = [i for i, val in enumerate(focal_no.T[:,0]>0) if val]

print("min, max of focal, patients with focal ", min(focal_no.T[:,0]), 
max(focal_no.T[:,0]),len(res_focal))


plt.plot(general_no.T[:,0])
plt.xlabel("Patient ID")
plt.ylabel('Generalized Number')
plt.title("Generalized seizures/patients")


plt.show()



res_gen = [j for j, val in enumerate(general_no.T[:,0]>0) if val]
print("min, max of generalized, patients with gen ", min(general_no.T[:,0]), 
max(general_no.T[:,0]),len(res_gen))
#  {'seizure_min': seizure_min,
 #'seizure_max': seizure_max , 'nb_files_per_patient':nb_files_per_patient})


counts, edges, bars  = plt.hist(general_no.T[:,0], bins=7, edgecolor='black')



plt.title('Focal number histogram ')
plt.xlabel('Number of files')
plt.ylabel('Patient count')


plt.show()


info_seizure = loadmat(os.path.join(savedir, 'seizure_duration_info_CHB_MIT.mat'))
nb_files_per_patient = info_seizure['nb_files_per_patient']
plt.plot(nb_files_per_patient.T[:,0])
plt.title('Number of files per patients CHB_MIT')
plt.show()

########### seizure information ############

seizure_min = info_seizure['seizure_min']
seizure_min = seizure_min.T[:,0]
seizure_max = info_seizure['seizure_max']
seizure_max = seizure_max.T[:,0]

plt.plot(seizure_min, label='min of seizure')
plt.plot(seizure_max, label='max of seizure')
plt.title('Min and max of seizure duration CHB_MIT')
plt.legend()
plt.xlabel('Patient ID')
plt.ylabel('seizure duration')
plt.show()
print("seizure shape is ", min(seizure_min))

# %%
