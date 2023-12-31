











# np.save(os.path.join(SaveResults,'accuracyCNNSMILELOO'),score)
# np.save(os.path.join(SaveResults,'lossCNNSMILELOO'),loss1)

# print(EDFFiles[indices[0]])

#   print(indices)
#   test.append(test_index)

# # print(test[2])

# for z in test:
#   indices=ReadMatFiles(dirname,z)
#   print(EDFFiles)
# -*- coding: utf-8 -*-
"""1DCNNServer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XIaeyIZmYN-0bMvxywlL5M5nTjaRBOJe
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
from keras import optimizers, regularizers
import keras.backend as K
from keras import regularizers
from tensorflow.keras.layers import InputLayer
from keras.layers import Input
import time
import tensorflow as tf
import os
import scipy
import h5py
import glob, os
# import BaseLineModel
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Concatenate
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import f1_score,plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve,roc_curve,roc_auc_score,auc
from sklearn.metrics import precision_recall_fscore_support,precision_recall_curve
import matplotlib.pyplot as plt;
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D, GlobalAveragePooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from keras import regularizers
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import InputLayer
from keras.layers import Input
# from google.colab import drive
from sklearn.model_selection import LeaveOneOut
import gc
gc.collect()
# drive.mount('/content/drive')

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


def TestDataLoadCNN(model,x_test,y_test,threshold):

   # classes=np.unique(y_test)
  y_pred=model.predict(x_test)
  pred_prob = model.predict_proba(x_test)
  predicted_class=np.zeros(y_pred.shape)
  y_test_thresh=np.zeros(y_test.shape)

  predicted_class[y_pred>threshold]=1
  y_test_thresh[y_test>threshold]=1

    ## The output of the predictor sometimes has an extra dimention of size 1
    ## np.squeeze removes this extra empty dimention
  predicted_class = np.squeeze(predicted_class)
  y_test_thresh = np.squeeze(y_test_thresh)
  pred_prob = np.squeeze(pred_prob)

  return x_test,y_test,y_pred,y_test_thresh,predicted_class,pred_prob


def TestDataLoadmerge(model,x_test,y_test,threshold):

   # classes=np.unique(y_test)
  y_pred=model.predict(x_test)
  pred_prob = model.predict_on_batch(x_test)
  predicted_class=np.zeros(y_pred.shape)
  y_test_thresh=np.zeros(y_test.shape)

  predicted_class[y_pred>threshold]=1
  y_test_thresh[y_test>threshold]=1

    ## The output of the predictor sometimes has an extra dimention of size 1
    ## np.squeeze removes this extra empty dimention
  predicted_class = np.squeeze(predicted_class)
  y_test_thresh = np.squeeze(y_test_thresh)
  pred_prob = np.squeeze(pred_prob)

  return x_test,y_test,y_pred,y_test_thresh,predicted_class,pred_prob

def  Conv_BN_Act_Pool(filtNo,filtsize1,filtsize2,input1,activation,PoolSize):
    conv1 = Conv1D(filtNo,filtsize1)(input1)
    conv2 = Conv1D(filtNo, filtsize2)(conv1)
    BN=BatchNormalization(axis=-1)(conv2)
    ActFunc=Activation(activation)(BN)
    pool1=MaxPooling1D(pool_size=PoolSize)(ActFunc)

    return pool1
    # model = Model(inputs = input1, outputs = pool1)

def define_model():
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

    model1=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,input1,activation,PoolSize)
    model2=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model1,activation,PoolSize)
    model3=Conv_BN_Act_Pool(filtNo1,filtsize2,filtsize1,model2,activation,PoolSize)
    model4=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model3,activation,PoolSize)
    model5=Conv_BN_Act_Pool(filtNo2,filtsize2,filtsize1,model4,activation,PoolSize)


    conv6=Conv1D(filtNo1,1)(model5)
    drop1=Dropout(0.25)(conv6)
    flat=Flatten()(drop1)

# Fully connected layer

    dense=Dense(denseSize)(flat)
##################################################################
    dim_data =int(vectorsize*(vectorsize+1)/2)
    vector_input = Input((dim_data,))

    # Concatenate the convolutional features and the vector input
    concat_layer= Concatenate()([vector_input, flat])
    denseout = Dense(100, activation='relu')(concat_layer)
    denseout = Dense(50, activation='relu')(denseout)
    output = Dense(1, activation='sigmoid')(denseout)

    # define a model with a list of two inputs
    model = Model(inputs=[input1, vector_input], outputs=output)


    model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['accuracy'])

    return model

model=define_model()
model.summary()

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

def ReadMatFiles(dirname,indx):

  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]
  ind=[]
  MI=[]
  X=[]
  Y=[]

  for j in list(indx):
    print(j)
    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
    ind.append(indices)

  ind=np.concatenate(ind,axis=0)

  for k in range(len(ind)):
    # print(ind[k])
    matfile=loadmat(os.path.join(EDF[int(ind[k])]))
    x=matfile['X_4sec']
    y=matfile['Y_label_4sec']
    mi=matfile['estimated_MI']

    for j in range(mi.shape[0]):

      mi2=mi[j,:,:]
      mi_mod=list(mi2[np.triu_indices(18)])
      MI.append(mi_mod)

    y=np.transpose(y)
    X.append(x)
    Y.append(y)

  X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI=np.array(MI)

  return X,Y,MI

def Data(dirname):

  X=[]
  Y=[]
  MI=[]
  EDF=PatientsEDFFile(dirname)

  for i in range(len(EDF)):

    matfile=loadmat(os.path.join(dirname, EDF[i]))
    x=matfile['X_4sec']
    y=matfile['Y_label_4sec']
    mi=matfile['estimated_MI']

    for j in range(mi.shape[0]):
      mi2=mi[j,:,:]
      mi_mod=list(mi2[np.triu_indices(18)])
      MI.append(mi_mod)

    y=np.transpose(y)
    X.append(x)
    Y.append(y)


  X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI=np.array(MI)

  return X,Y,MI

# version 1: filters = 5 kernelsize = 9
## Version2
#     filter1=8
#     filter2=16

#     kernelsize1=7
#     kernelsize2=9
## Version3
#     filter1=8
#     filter2=16

#     kernelsize1=10
#     kernelsize2=22

## Version4
#     filter1=8 ---> at the first CNN and only two dense
#     filter2=16

#     kernelsize1=10
#     kernelsize2=22

## Version5
#     filter1=8 ---> at the first CNN and only two dense
#     filter2=16

#     kernelsize1=22
#     kernelsize2=10

def ReadMatFilesDiff(dirname,indx):
  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]
  ind=[]
  MI=[]
  X=[]
  Y=[]

  for j in list(indx):

    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
    ind.append(indices)

  ind=np.concatenate(ind,axis=0)

  for k in range(len(ind)):

    matfile=loadmat(os.path.join(dirname,EDF[int(ind[k])]))
    x=matfile['X_4sec']
    y=matfile['Y_label_4sec']
    mi=matfile['estimated_MI']

    for j in range(mi.shape[0]):

      mi2=mi[j,:,:]
      mi_mod=list(mi2[np.triu_indices(18)])
      MI.append(mi_mod)

    y=np.transpose(y)
    X.append(x)
    Y.append(y)

  X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI=np.array(MI)
  MI_diff=np.zeros((MI.shape[0]-1,MI.shape[1]))

  for j in range(MI.shape[0]-1):
    MI_diff[j,:]=MI[j+1,:]-MI[j,:]




  return X[1:,:,:],Y[1:,:],MI[1:,:],MI_diff

def ReadMatFilesSMILEonly(dirname1,dirname2,indx):

  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname1)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]
  ind=[]
  MI=[]
  X=[]
  Y=[]
  for j in list(indx):
    print(j)
    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
    ind.append(indices)
  ind=np.concatenate(ind,axis=0)
  for k in range(len(ind)):
    # print(ind[k])
    matfile=loadmat(os.path.join(dirname1,EDF[int(ind[k])]))
    matfile2=loadmat(os.path.join(dirname2,EDF[int(ind[k])]))
    ylabel=matfile2['Y_label']
    x=matfile['X_4sec']
    y=matfile['Y_label_4sec']
    mi=matfile['estimated_MI']
    for j in range(mi.shape[0]):
      mi2=mi[j,:,:]
      mi_mod=list(mi2[np.triu_indices(18)])
      MI.append(mi_mod)
    # y=np.transpose(y)
    ylabel=np.transpose(ylabel)
    # X.append(x)
    Y.append(ylabel)
  # X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI=np.array(MI)
  MI=MI.reshape(MI.shape[0],MI.shape[1],1)
  return MI,Y

def ReadMatFilesSMILEDiffonly(dirname1,dirname2,indx):

  EDF=[]
  EDFFiles=[]
  Name=[]
  EDF=PatientsEDFFile(dirname1)

  Name=PatientsName()
  Xfile=[]
  Yfile=[]
  ind=[]
  MI=[]
  X=[]
  Y=[]

  for j in list(indx):
    print(j)
    indices = [i for i, elem in enumerate(EDF) if Name[j] in elem]
    ind.append(indices)

  ind=np.concatenate(ind,axis=0)

  for k in range(len(ind)):
    # print(ind[k])
    matfile=loadmat(os.path.join(dirname1,EDF[int(ind[k])]))
    matfile2=loadmat(os.path.join(dirname2,EDF[int(ind[k])]))
    ylabel=matfile2['Y_label']
    x=matfile['X_4sec']
    y=matfile['Y_label_4sec']
    mi=matfile['estimated_MI']

    for j in range(mi.shape[0]):

      mi2=mi[j,:,:]
      mi_mod=list(mi2[np.triu_indices(18)])
      MI.append(mi_mod)

    # y=np.transpose(y)
    ylabel=np.transpose(ylabel)
    # X.append(x)
    Y.append(ylabel)

  # X=np.concatenate(X,axis=0)
  Y=np.concatenate(Y,axis=0)
  MI=np.array(MI)

  MI_diff=np.zeros((MI.shape[0]-1,MI.shape[1]))

  for j in range(MI.shape[0]-1):

    MI_diff[j,:]=MI[j+1,:]-MI[j,:]

  MI=MI.reshape(MI.shape[0],MI.shape[1],1)
  MI_diff=MI_diff.reshape(MI_diff.shape[0],MI_diff.shape[1],1)

  return MI[1:,:,:],Y[1:,:],MI_diff


def define_model2():



    model = Sequential()

    filter1=8
    filter2=16

    kernelsize1=22
    kernelsize2=10

    model.add(Conv1D(filter1, kernelsize2, input_shape=(1024,18)))
    model.add(Conv1D(filter1, kernelsize1))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filter1, kernelsize2))
    model.add(Conv1D(filter1, kernelsize1))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filter1, kernelsize2))
    model.add(Conv1D(filter1, kernelsize1))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filter2, kernelsize2))
    model.add(Conv1D(filter2, kernelsize1))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filter2, kernelsize2))
    model.add(Conv1D(filter2, kernelsize1))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filter1, 1))

    model.add(Dropout(0.25))
    model.add(Flatten())

# Fully connected layer

    model.add(Dense(8))
    model.add(Dense(8))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['accuracy'])

    return model

def Calculation(y_test_thresh, predicted_class,pred_prob):

    precision, recall, fscore, _ = precision_recall_fscore_support(y_test_thresh, predicted_class, average='weighted')
    fpr, tpr, _ = roc_curve(y_test_thresh, pred_prob)

    precision, recall, _ = precision_recall_curve(y_test_thresh, pred_prob)
    PR=auc(recall, precision)
    ROC=roc_auc_score(y_test_thresh, pred_prob)

    return fpr,tpr,PR,ROC

dirname='/home/baharsalafian/6FoldCrossSMILE'
dirname2='/home/baharsalafian/SMILELabel'
SaveResults='/home/baharsalafian/CNNSMILE6FoldRes'
modeldir='/home/baharsalafian/CNNSMILE6FoldRes'


threshold=0.2
threshold2=0.03

fpr_smile=[]
tpr_smile=[]
PR_smile=[]
ROC_smile=[]

fpr_smileonly=[]
tpr_smileonly=[]
PR_smileonly=[]
ROC_smileonly=[]

fpr_smilediff=[]
tpr_smilediff=[]
PR_smilediff=[]
ROC_smilediff=[]

fpr_smileonlydiff=[]
tpr_smileonlydiff=[]
PR_smileonlydiff=[]
ROC_smileonlydiff=[]


fpr_cnn=[]
tpr_cnn=[]
PR_cnn=[]
ROC_cnn=[]

fold_no=1
batchsize=128
epoch=10
start_time = time.time()
FoldNum=6

kfold = KFold(n_splits=FoldNum, shuffle=False)
for trainindx, testindx in kfold.split(range(24)):

    X_test,Y_test,mi_test = ReadMatFiles(dirname,testindx)

    X_test1,Y_test1,mi_test1,mi_testdiff1=ReadMatFilesDiff(dirname,testindx)

    X_test2,Y_test2=ReadMatFilesSMILEonly(dirname,dirname2,testindx)

    X_test3,Y_test3,mi_testdiff3=ReadMatFilesSMILEDiffonly(dirname,dirname2,testindx)

    # model.fit([X_train,mi_train],Y_train,validation_split=0.2,batch_size=batchsize , epochs=epoch,verbose = 2)
    ModelName1='CNNSMILE6foldV2'+ str(fold_no)+'.h5'
    model1=tf.keras.models.load_model(os.path.join(modeldir,ModelName1))

    ModelName2='CNN6fold'+ str(fold_no)+'.h5'
    model2=tf.keras.models.load_model(os.path.join(modeldir,ModelName2))

    ModelName3='CNNSMILE6foldDiff'+str(fold_no)+'.h5'
    model3=tf.keras.models.load_model(os.path.join(modeldir,ModelName3))

    ModelName4='CNN6foldsmileonly'+str(fold_no)+'.h5'
    model4=tf.keras.models.load_model(os.path.join(modeldir,ModelName4))

    ModelName5='CNN6foldsmilediffonly'+str(fold_no)+'.h5'
    model5=tf.keras.models.load_model(os.path.join(modeldir,ModelName5))
    # X_train = None
    # Y_train = None
    # gc.collect()

    _,_,_,y_test_thresh1,predicted_class1,predicted_prob1 =TestDataLoadmerge(model1,[X_test,mi_test],Y_test,threshold)

    _,_,_,y_test_thresh2,predicted_class2,predicted_prob2=TestDataLoadCNN(model2,X_test,Y_test,threshold)

    _,_,_,y_test_thresh3,predicted_class3,predicted_prob3=TestDataLoadmerge(model3,[X_test1,mi_test1,mi_testdiff1],Y_test1,threshold)

    _,_,_,y_test_thresh4,predicted_class4,predicted_prob4=TestDataLoadCNN(model4,X_test2,Y_test2,threshold2)

    _,_,_,y_test_thresh5,predicted_class5,predicted_prob5=TestDataLoadmerge(model5,[X_test3,mi_testdiff3],Y_test3,threshold2)

    fpr1,tpr1,PR1,ROC1=Calculation(y_test_thresh1, predicted_class1,predicted_prob1)


    fpr_smile.append(fpr1)
    tpr_smile.append(tpr1)
    ROC_smile.append(ROC1)
    PR_smile.append(PR1)


    fpr2,tpr2,PR2,ROC2=Calculation(y_test_thresh2, predicted_class2,predicted_prob2)

    fpr_cnn.append(fpr2)
    tpr_cnn.append(tpr2)
    ROC_cnn.append(ROC2)
    PR_cnn.append(PR2)


    fpr3,tpr3,PR3,ROC3=Calculation(y_test_thresh3, predicted_class3,predicted_prob3)
    fpr_smilediff.append(fpr3)
    tpr_smilediff.append(tpr3)
    ROC_smilediff.append(ROC3)
    PR_smilediff.append(PR3)

    fpr3,tpr3,PR3,ROC3=Calculation(y_test_thresh4, predicted_class4,predicted_prob4)
    fpr_smileonly.append(fpr3)
    tpr_smileonly.append(tpr3)
    ROC_smileonly.append(ROC3)
    PR_smileonly.append(PR3)

    fpr3,tpr3,PR3,ROC3=Calculation(y_test_thresh5, predicted_class5,predicted_prob5)
    fpr_smileonlydiff.append(fpr3)
    tpr_smileonlydiff.append(tpr3)
    ROC_smileonlydiff.append(ROC3)
    PR_smileonlydiff.append(PR3)



    fold_no=fold_no+1


    X_test = None
    Y_test = None
    gc.collect()

np.save(os.path.join(SaveResults, 'fpr_smile'),  fpr_smile)
np.save(os.path.join(SaveResults, 'tpr_smile'), tpr_smile)
np.save(os.path.join(SaveResults, 'PR_smile'), PR_smile)
np.save(os.path.join(SaveResults, 'ROC_smile'), ROC_smile)

np.save(os.path.join(SaveResults, 'fpr_smileonly'),  fpr_smileonly)
np.save(os.path.join(SaveResults, 'tpr_smileonly'), tpr_smileonly)
np.save(os.path.join(SaveResults, 'PR_smileonly'), PR_smileonly)
np.save(os.path.join(SaveResults, 'ROC_smileonly'), ROC_smileonly)

np.save(os.path.join(SaveResults, 'fpr_smileonlydiff'),  fpr_smileonlydiff)
np.save(os.path.join(SaveResults, 'tpr_smileonlydiff'), tpr_smileonlydiff)
np.save(os.path.join(SaveResults, 'PR_smileonlydiff'), PR_smileonlydiff)
np.save(os.path.join(SaveResults, 'ROC_smileonlydiff'), ROC_smileonlydiff)

np.save(os.path.join(SaveResults, 'fpr_smilediff'),  fpr_smilediff)
np.save(os.path.join(SaveResults, 'tpr_smilediff'), tpr_smilediff)
np.save(os.path.join(SaveResults, 'PR_smilediff'), PR_smilediff)
np.save(os.path.join(SaveResults, 'ROC_smilediff'), ROC_smilediff)


np.save(os.path.join(SaveResults, 'fpr_cnn'),  fpr_cnn)
np.save(os.path.join(SaveResults, 'tpr_cnn'), tpr_cnn)
np.save(os.path.join(SaveResults, 'PR_cnn'), PR_cnn)
np.save(os.path.join(SaveResults, 'ROC_cnn'), ROC_cnn)
print("--- %s seconds ---" % (time.time() - start_time))

# np.save(os.path.join(SaveResults,'accuracyCNNSMILELOO'),score)
# np.save(os.path.join(SaveResults,'lossCNNSMILELOO'),loss1)

# print(EDFFiles[indices[0]])

#   print(indices)
#   test.append(test_index)

# # print(test[2])

# for z in test:
#   indices=ReadMatFiles(dirname,z)
#   print(EDFFiles)
