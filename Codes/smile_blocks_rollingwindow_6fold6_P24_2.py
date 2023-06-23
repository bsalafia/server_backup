# -*- coding: utf-8 -*-
"""SMILE_Blocks_RollingWindow_6Fold.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ETADg77e6DPxVhh1xRtizL5orB6_rPDH
"""

# !pip install pycit
# !pip install --upgrade numpy

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# from pycit.estimators import ksg_mi,bi_ksg_cmi,mixed_mi,mixed_cmi,bi_ksg_mi
# from pycit.preprocessing import low_amplitude_noise
import sys
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
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
from numpy.random import default_rng
import torch.nn.functional as F
from sklearn.model_selection import KFold
# from google.colab import drive
import gc
import time
# from numpy.lib.stride_tricks import sliding_window_view
from numba import jit, njit, prange
# drive.mount('/content/drive')
# gc.collect()

def LoadData(dirname,Name):

  matFile1 = loadmat(os.path.join(dirname, Name))
  DenoisedSig=matFile1['DenoisedSig']
  Fs=matFile1['Fs']
  Siezure_start=matFile1['Siezure_start']
  Siezure_end=matFile1['Siezure_end']
  Sig_start=matFile1['Sig_start']
  Sig_end=matFile1['Sig_end']
  DenoisedSigSeizure= np.transpose(DenoisedSig)

  return DenoisedSig,Fs[0][0],Siezure_start[0][0],Siezure_end[0][0],Sig_start[0][0],Sig_end[0][0]

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

def Label(dirname,Name,WindowSize1,WindowSize2):

  DenoisedSig,Fs,Siezure_start,Siezure_end,Sig_start,Sig_end = LoadData(dirname,Name)

  n_channels= DenoisedSig.shape[0]

  # X=np.zeros((np.int64(Sig_end)-np.int64(Sig_start)-WindowSize+1,n_channels,WindowSize*Fs))

  # Ylabel=np.zeros((np.int64(Sig_end)-np.int64(Sig_start)-WindowSize+1,1))

  n1=1
  n2=1+WindowSize1
  s=Siezure_start-Sig_start+1
  e=Siezure_end-Sig_start+1
  t1 = 0
  t2 = WindowSize1*Fs
  X=[]
  Ylabel=[]
  Ylabel_4sec=[]
  Sup=DenoisedSig.shape[1]
  k=0

  while t2 <= Sup:


    X.append(DenoisedSig[:,t1:min(t2,Sup)])

    if  (n2<=s or n1>=e):
      Ylabel.append(0)

    elif (n1<s and n2<e):

      Ylabel.append(min(n2-s,e-s,WindowSize1)/WindowSize1)

    elif (n1>=s and n2<=e):
      Ylabel.append(1)

    elif n2>=e:
      Ylabel.append(min(e-n1,e-s)/WindowSize1)

    n11=n2-WindowSize2

    if  (n2<=s or n11>=e):

      Ylabel_4sec.append(0)

    elif (n11<s and n2<e):

      Ylabel_4sec.append(min(n2-s,e-s,WindowSize2)/WindowSize2)

    elif (n11>=s and n2<=e):

      Ylabel_4sec.append(1)

    elif n2>=e:

      Ylabel_4sec.append(min(e-n11,e-s)/WindowSize2)

    if t2+Fs > Sup:
      X=np.array(X)
      X_4sec=X[:,:,-1024::]

    t2 = t2 + Fs
    t1 = t1 + Fs
    n2=n2+1
    n1=n1+1
    k  = k + 1

  X=np.array(X)
  Ylabel=np.array(Ylabel)
  Ylabel_4sec=np.array(Ylabel_4sec)
  return X,Ylabel,X_4sec,Ylabel_4sec

def SavePatientswithEDFList(dirnameAllMats,savedir,WindowSize,indx):


  EDF=[]
  Name=[]
  EDF=PatientsEDFFile(dirnameAllMats)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]


  indices = [i for i, elem in enumerate(EDF) if Name[indx] in elem]
  # print(indices)

  for k in range(len(indices)):
      #print(EDF[indices[k]])
      X,Ylabel = Label(dirname,EDF[indices[k]],WindowSize)
      Xfile.append(X)
      Yfile.append(Ylabel)
      X=[]
      Ylabel=[]
  class Antoine:

    pass
  Data = Antoine()
  Xfile=np.concatenate(Xfile,axis=0)
  Xfile=np.transpose(Xfile,(0, 2, 1))
  Yfile=np.concatenate(Yfile,axis=0)
  Data.Xfile=Xfile
  Data.Yfile=Yfile
  np.save(os.path.join(savedir, Name[indx]+'X_'+str(WindowSize)+'.npy'), Xfile)
  np.save(os.path.join(savedir, Name[indx]+'Y_'+str(WindowSize)+'.npy'), Yfile)
  return Xfile,Yfile

def ReadMatFiles(dirname,indx):

  EDF=[]
  Name=[]
  EDF=PatientsEDFFile(dirname)
  Name=PatientsName()
  Xfile=[]
  Yfile=[]


  indices = [i for i, elem in enumerate(EDF) if Name[indx] in elem]

  return indices,EDF

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

class MI_Est_Losses():
  def __init__(self, estimator, device):
    """Estimate variational lower bounds on mutual information based on
      a function T(X,Y) represented by a NN using variational lower bounds.
    Args:
      estimator: string specifying estimator, one of:
        'smile', 'nwj', 'infonce', 'tuba', 'js', 'interpolated'
      device: the device to use (CPU or GPU)
    """
    self.device = device
    self.estimator = estimator



  def logmeanexp_diag(self, x):
    """Compute logmeanexp over the diagonal elements of x."""
    batch_size = x.size(0)

    logsumexp = torch.logsumexp(x.diag(), dim=(0,))
    num_elem = batch_size

    return logsumexp - torch.log(torch.tensor(num_elem).float()).to(self.device)


  def logmeanexp_nodiag(self, x, dim=None):
      batch_size = x.size(0)
      if dim is None:
          dim = (0, 1)

      logsumexp = torch.logsumexp(
          x - torch.diag(np.inf * torch.ones(batch_size).to(self.device)), dim=dim)

      try:
          if len(dim) == 1:
              num_elem = batch_size - 1.
          else:
              num_elem = batch_size * (batch_size - 1.)
      except ValueError:
          num_elem = batch_size - 1
      return logsumexp - torch.log(torch.tensor(num_elem)).to(self.device)


  def tuba_lower_bound(self, scores, log_baseline=None):
      if log_baseline is not None:
          scores -= log_baseline[:, None]

      # First term is an expectation over samples from the joint,
      # which are the diagonal elmements of the scores matrix.
      joint_term = scores.diag().mean()

      # Second term is an expectation over samples from the marginal,
      # which are the off-diagonal elements of the scores matrix.
      marg_term = self.logmeanexp_nodiag(scores).exp()
      return 1. + joint_term - marg_term


  def nwj_lower_bound(self, scores):
      return self.tuba_lower_bound(scores - 1.)


  def infonce_lower_bound(scores):
      nll = scores.diag().mean() - scores.logsumexp(dim=1)
      # Alternative implementation:
      # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
      mi = torch.tensor(scores.size(0)).float().log() + nll
      mi = mi.mean()
      return mi


  def js_fgan_lower_bound(self, f):
      """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
      f_diag = f.diag()
      first_term = -F.softplus(-f_diag).mean()
      n = f.size(0)
      second_term = (torch.sum(F.softplus(f)) -
                    torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
      return first_term - second_term


  def js_lower_bound(self, f):
      """Obtain density ratio from JS lower bound then output MI estimate from NWJ bound."""
      nwj = self.nwj_lower_bound(f)
      js = self.js_fgan_lower_bound(f)

      with torch.no_grad():
          nwj_js = nwj - js

      return js + nwj_js


  def dv_upper_lower_bound(self, f):
      """
      Donsker-Varadhan lower bound, but upper bounded by using log outside.
      Similar to MINE, but did not involve the term for moving averages.
      """
      first_term = f.diag().mean()
      second_term = self.logmeanexp_nodiag(f)

      return first_term - second_term


  def mine_lower_bound(self, f, buffer=None, momentum=0.9):
      """
      MINE lower bound based on DV inequality.
      """
      if buffer is None:
          buffer = torch.tensor(1.0).to(self.device)
      first_term = f.diag().mean()

      buffer_update = self.logmeanexp_nodiag(f).exp()
      with torch.no_grad():
          second_term = self.logmeanexp_nodiag(f)
          buffer_new = buffer * momentum + buffer_update * (1 - momentum)
          buffer_new = torch.clamp(buffer_new, min=1e-4)
          third_term_no_grad = buffer_update / buffer_new

      third_term_grad = buffer_update / buffer_new

      return first_term - second_term - third_term_grad + third_term_no_grad, buffer_update


  def smile_lower_bound(self,f, clip=None):
      if clip is not None:
          f_ = torch.clamp(f, -clip, clip)
      else:
          f_ = f
      z = self.logmeanexp_nodiag(f_, dim=(0, 1))
      dv = f.diag().mean() - z

      js = self.js_fgan_lower_bound(f)

      with torch.no_grad():
          dv_js = dv - js

      return js + dv_js


  def _ent_js_fgan_lower_bound(self, vec, ref_vec):
      """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
      first_term = -F.softplus(-vec).mean()
      second_term = torch.sum(F.softplus(ref_vec)) / ref_vec.size(0)
      return first_term - second_term

  def _ent_smile_lower_bound(self, vec, ref_vec, clip=None):
      if clip is not None:
          ref = torch.clamp(ref_vec, -clip, clip)
      else:
          ref = ref_vec

      batch_size = ref.size(0)
      z = log_mean_ef_ref = torch.logsumexp(ref, dim=(0, 1)) - torch.log(torch.tensor(batch_size)).to(self.device)
      dv = vec.mean() - z
      js = self._ent_js_fgan_lower_bound(vec, ref_vec)

      with torch.no_grad():
          dv_js = dv - js

      return js + dv_js

  def entropic_smile_lower_bound(self, f, clip=None):
      t_xy, t_xy_ref, t_x, t_x_ref, t_y, t_y_ref = f

      d_xy = self._ent_smile_lower_bound(t_xy, t_xy_ref, clip=clip)
      d_x = self._ent_smile_lower_bound(t_x, t_x_ref, clip=clip)
      d_y = self._ent_smile_lower_bound(t_y, t_y_ref, clip=clip)

      return d_xy, d_x, d_y

  def chisquare_pred(self, vec, ref_vec, beta=1, with_smile=False, clip=None):
      b = torch.tensor(beta).to(self.device)
      if with_smile:
        if clip is not None:
            ref = torch.clamp(ref_vec, -clip, clip)
        else:
            ref = ref_vec

        batch_size = ref.size(0)
        z = log_mean_ef_ref = torch.logsumexp(ref, dim=(0, 1)) - torch.log(torch.tensor(batch_size)).to(self.device)
      else:
        z = log_mean_ef_ref = torch.mean(torch.exp(ref_vec))/b - torch.log(b) - 1

      dv = vec.mean() - z
      js = self._ent_js_fgan_lower_bound(vec, ref_vec)

      with torch.no_grad():
          dv_js = dv - js

      return js + dv_js

  def chi_square_lower_bound(self, f, beta=1, with_smile=False, clip=None):
      t_xy, t_xy_ref = f

      d_xy = self.chisquare_pred(t_xy, t_xy_ref, beta=beta, with_smile=with_smile, clip=clip)

      return d_xy

  def mi_est_loss(self, net_output, **kwargs):
      """Estimate variational lower bounds on mutual information.

    Args:
      net_output: output(s) of the neural network estimator

    Returns:
      scalar estimate of mutual information
      """
      if self.estimator == 'infonce':
          mi = self.infonce_lower_bound(net_output)
      elif self.estimator == 'nwj':
          mi = self.nwj_lower_bound(net_output)
      elif self.estimator == 'tuba':
          mi = self.tuba_lower_bound(net_output, **kwargs)
      elif self.estimator == 'js':
          mi = self.js_lower_bound(net_output)
      elif self.estimator == 'smile':
          mi = self.smile_lower_bound(net_output, **kwargs)
      elif self.estimator == 'dv':
          mi = self.dv_upper_lower_bound(net_output)
      elif self.estimator == 'ent_smile':
          mi = self.entropic_smile_lower_bound(net_output, **kwargs)
      elif self.estimator == 'chi_square':
          mi = self.chi_square_lower_bound(net_output, **kwargs)
      return mi

def mlp(dim, hidden_dim, output_dim, layers, activation):
    """Create a mlp from the configurations."""
    activation = {
        'relu': nn.ReLU
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)


class SeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(SeparableCritic, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)

    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores


class ConcatCritic(nn.Module):
    """Concat critic, where we concat the inputs and use one MLP to output the value."""

    def __init__(self, dim, hidden_dim, layers, activation, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(dim * 2, hidden_dim, 1, layers, activation)

    def forward(self, x, y):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return torch.reshape(scores, [batch_size, batch_size]).t()

class EntropicCritic(nn.Module):
  # Inner class that defines the neural network architecture
  def __init__(self, dim, hidden_dim, embed_dim, layers, activation):
      super(EntropicCritic, self).__init__()
      self.f = mlp(dim, hidden_dim, embed_dim, layers, activation)

  def forward(self, inp):
      output = self.f(inp)
      return output


class ConcatEntropicCritic():
    """Concat entropic critic, where we concat the inputs and use one MLP to output the value."""
    def __init__(self, dim, hidden_dim, layers, activation,
                 device, ref_batch_factor=1,**extra_kwargs):
        # output is scalar score
        self.ref_batch_factor = ref_batch_factor
        self.fxy = EntropicCritic(dim * 2, hidden_dim, 1, layers, activation)
        self.fx = EntropicCritic(dim, hidden_dim, 1, layers, activation)
        self.fy = EntropicCritic(dim, hidden_dim, 1, layers, activation)
        self.device = device

    def _uniform_sample(self, data, batch_size):
      # Sample the reference uniform distribution
      data_min = data.min(dim=0)[0]
      data_max = data.max(dim=0)[0]
      return (data_max - data_min) * torch.rand((batch_size, data_min.shape[0])).to(self.device) + data_min

    def to(self, device):
      self.fxy.to(device)
      self.fx.to(device)
      self.fy.to(device)

    def forward(self, x, y):
        batch_size = x.size(0)
        XY = torch.cat((x, y), dim=1)
        X_ref = self._uniform_sample(x, batch_size=int(
            self.ref_batch_factor * batch_size))
        Y_ref = self._uniform_sample(y, batch_size=int(
            self.ref_batch_factor * batch_size))
        XY_ref = torch.cat((X_ref, Y_ref), dim=1)

        # Compute t function outputs approximated by NNs
        t_xy = self.fxy(XY)
        t_xy_ref = self.fxy(XY_ref)
        t_x = self.fx(x)
        t_x_ref = self.fx(X_ref)
        t_y = self.fy(y)
        t_y_ref = self.fy(Y_ref)
        return (t_xy, t_xy_ref, t_x, t_x_ref, t_y, t_y_ref)

class chiCritic():
    """Concat entropic critic, where we concat the inputs and use one MLP to output the value."""
    def __init__(self, dim, hidden_dim, layers, activation,
                 device, ref_batch_factor=1,**extra_kwargs):
        # output is scalar score
        self.ref_batch_factor = ref_batch_factor
        self.fxy = EntropicCritic(dim * 2, hidden_dim, 1, layers, activation)


        self.device = device

    def _uniform_sample(self, data, batch_size):
      # Sample the reference uniform distribution
      data_min = data.min(dim=0)[0]
      data_max = data.max(dim=0)[0]
      return (data_max - data_min) * torch.rand((batch_size, data_min.shape[0])).to(self.device) + data_min

    def to(self, device):
      self.fxy.to(device)

    def forward(self, x, y):
        batch_size = x.size(0)
        XY = torch.cat((x, y), dim=1)
        X_ref = self._uniform_sample(x, batch_size=int(
            self.ref_batch_factor * batch_size))
        Y_ref = self._uniform_sample(y, batch_size=int(
            self.ref_batch_factor * batch_size))
        XY_ref = torch.cat((X_ref, Y_ref), dim=1)

        # Compute t function outputs approximated by NNs
        t_xy = self.fxy(XY)
        t_xy_ref = self.fxy(XY_ref)
        return (t_xy, t_xy_ref)

# from numba import jit, njit, prange
# @jit(nopython=True, parallel=True, fastmath = True)
# V1=0.5,V2=0.8,V3=0.9
dirname='/home/baharsalafian/P242'
savedir='/home/baharsalafian/SMILEFold62/'
dir_mi_results = savedir
critic_params = {
    'dim': 256,
    'NNdim': 32,
    'layers': 2,
    'embed_dim': 32,
    'hidden_dim': 256,
    'activation': 'relu',
    'ref_batch_factor': 10,
    'learning_rate': 0.0005,
    'batch_size': 128
}
WindowSize=[32]
estimator = 'smile'
clip = 0.9
init_epoch = 200
rest_epoch = 35
WindowSize2=4

train_indx=5


if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

device = torch.device(dev)

mi_est_loss = MI_Est_Losses('smile', device)

vec_dim = 1
knn_k = 5
start_time = time.time()
labels = [0, 1]
FoldNum=6
kfold = KFold(n_splits=FoldNum, shuffle=False)
train=[]

for trainindx, testindx in kfold.split(range(24)):
  train.append(testindx)



indices,EDFFiles=ReadMatFiles(dirname,train[train_indx][-1])
for q in range(len(WindowSize)):
  for file_idx in range(len(indices)):

    X_train,Y_train,X_4sec,Ylabel_4sec=Label(dirname,EDFFiles[indices[file_idx]],WindowSize[q],WindowSize2)
    X_train=np.transpose(X_train,(0, 2, 1))
    X_4sec=np.transpose(X_4sec,(0, 2, 1))
    print(X_train.shape)
    print(Y_train.shape)
    Y_train = (Y_train > 0.03).astype(int)
    signal_max,signal_min,den=normalize(X_train)
    mean1,stdv=standardize(X_train)
    estimated_MI = np.zeros((X_train.shape[0], X_train.shape[2], X_train.shape[2]))
    # estimated_MI_std = np.zeros((len(labels), X_train.shape[2], X_train.shape[2]))
    # estimated_MI_per4Sec = np.zeros((len(labels), X_train.shape[0],X_train.shape[2], X_train.shape[2]))
    all_mi_neuralnet = []
    all_opt_crit = []
    for j in range(X_train.shape[2]-1):
      col_mi_neuralnet = list()
      col_opt_crit = list()
      for k in range(j+1, X_train.shape[2]):
        mi_neuralnet = ConcatCritic( critic_params['NNdim'], critic_params['hidden_dim'],
                                  critic_params['layers'], critic_params['activation'])
        mi_neuralnet.to(device)
        opt_crit = torch.optim.Adam(mi_neuralnet.parameters(), lr=critic_params['learning_rate'])
        col_mi_neuralnet.append(mi_neuralnet)
        col_opt_crit.append(opt_crit)
      all_mi_neuralnet.append(col_mi_neuralnet)
      all_opt_crit.append(col_opt_crit)
    for i in range(X_train.shape[0]):
      if i==0:
        max_epoch = init_epoch
      else:
        max_epoch = rest_epoch
      for j in tqdm(range(X_train.shape[2]-1)):
        jk = -1
        for k in range(j+1, X_train.shape[2]):
          jk += 1
          X = X_train[i, :, j]/signal_max[j]
          Y = X_train[i, :, k]/signal_max[k]
          # mi_neuralnet = ConcatCritic( critic_params['NNdim'], critic_params['hidden_dim'],
          #                           critic_params['layers'], critic_params['activation'])
          # mi_neuralnet.to(device)
          # opt_crit = torch.optim.Adam(mi_neuralnet.parameters(), lr=critic_params['learning_rate'])
          #all_x = sliding_window_view(X, critic_params['dim'])
          #all_y = sliding_window_view(Y, critic_params['dim'])
          all_x=np.array(np.split(X, critic_params['dim']))
          all_y=np.array(np.split(Y, critic_params['dim']))
          # prfint(all_y.shape)
        # all_x = np.concatenate(all_x)
        # # print(all_x.shape)
        # all_y = np.concatenate(all_y)
        # print(all_y.shape)
          #print(j, k)
          estimates = np.zeros(max_epoch)
          for epc in range(max_epoch):
            total_sampl = all_x.shape[0]
            selected_idx = np.random.choice(total_sampl, critic_params['batch_size'])
            batch_x = all_x[selected_idx]
            batch_y = all_y[selected_idx]
            batch_x = torch.tensor(batch_x, dtype=torch.float).to(device)
            # print(batch_x.shape)
            batch_y = torch.tensor(batch_y, dtype=torch.float).to(device)
            # print(batch_y.shape)
            all_opt_crit[j][jk].zero_grad()
            net_out = all_mi_neuralnet[j][jk](batch_x, batch_y)
            # print(net_out.shape)
            mi_est = mi_est_loss.mi_est_loss(net_out, clip=clip)
            loss = -mi_est
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(mi_neuralnet.parameters(), 2)
            all_opt_crit[j][jk].step()
            mi_est = mi_est.detach().cpu().numpy()
            estimates[epc]= mi_est
          # plt.plot(estimates)
          # plt.show()
          estimated_MI[i,j,k] = np.mean(estimates[-50:])
      Name=EDFFiles[indices[file_idx]].split('.')
      savemat(os.path.join(savedir, Name[0]+'MI_RollWindow'+str(WindowSize[0])+'_NNdim'+str(critic_params['NNdim'])+'_FoldNo{}_MaxEpoch{}_MinEpoch{}.mat'.format(train_indx+1,init_epoch,rest_epoch)), {"estimated_MI": estimated_MI, "Y_label": Y_train ,"Y_label_4sec": Ylabel_4sec,"X_4sec": X_4sec})
      print("-----File_idx {},Name{}, indx {}, label {}".format(file_idx,Name, i, Y_train[i]))
      plt.imshow(estimated_MI[i], vmin=0,  vmax=6)
      plt.show()
      #print(estimated_MI[i])
        # sys.stdout = log_file
        #print("-----File_idx {}, indx {}, Chan1 {},  Chan2 {}, Estimated MI {:.4f}".format(file_idex, i,j, k, estimated_MI[i,j,k]))
          # sys.stdout = old_stdout


# log_file.close()


     # plt.imshow(estimated_MI[l])
      # plt.show()
# torch.save(mi_neuralnet.state_dict(), './Trained_MI_Model')
print("--- %s seconds ---" % (time.time() - start_time))

# def PatientsEDFFile(dirname):

#     os.chdir(dirname)
#     a=[]
#     X=[]
#     Y=[]
#     k=0
#     for file in glob.glob("*.mat"):

#         a.append(file)
#         print(a)

#     return a

# dirname='/content/drive/MyDrive/Colab Notebooks/Bahareh/1DCNN10SecData'
# # a=PatientsEDFFile(dirname)
# # # X,Y=Label(dirname,a[2],32)

# a[0].split('.')

# def Label(dirname,Name,WindowSize1,WindowSize2):

#   DenoisedSig,Fs,Siezure_start,Siezure_end,Sig_start,Sig_end = LoadData(dirname,Name)

#   n_channels= DenoisedSig.shape[0]

#   # X=np.zeros((np.int64(Sig_end)-np.int64(Sig_start)-WindowSize+1,n_channels,WindowSize*Fs))

#   # Ylabel=np.zeros((np.int64(Sig_end)-np.int64(Sig_start)-WindowSize+1,1))

#   n1=1
#   n2=1+WindowSize1
#   s=Siezure_start-Sig_start+1
#   e=Siezure_end-Sig_start+1
#   t1 = 0
#   t2 = WindowSize1*Fs
#   X=[]
#   Ylabel=[]
#   Ylabel_4sec=[]
#   Sup=DenoisedSig.shape[1]
#   k=0

#   while t2 <= Sup:


#     X.append(DenoisedSig[:,t1:min(t2,Sup)])

#     if  (n2<=s or n1>=e):
#       Ylabel.append(0)

#     elif (n1<s and n2<e):

#       Ylabel.append(min(n2-s,e-s,WindowSize1)/WindowSize1)

#     elif (n1>=s and n2<=e):
#       Ylabel.append(1)

#     elif n2>=e:
#       Ylabel.append(min(e-n1,e-s)/WindowSize1)

#     n11=n2-WindowSize2

#     if  (n2<=s or n11>=e):

#       Ylabel_4sec.append(0)

#     elif (n11<s and n2<e):

#       Ylabel_4sec.append(min(n2-s,e-s,WindowSize2)/WindowSize2)

#     elif (n11>=s and n2<=e):

#       Ylabel_4sec.append(1)

#     elif n2>=e:

#       Ylabel_4sec.append(min(e-n11,e-s)/WindowSize2)

#     if t2+Fs > Sup:
#       X=np.array(X)
#       X_4sec=X[:,:,-1024::]

#     t2 = t2 + Fs
#     t1 = t1 + Fs
#     n2=n2+1
#     n1=n1+1
#     k  = k + 1

#   X=np.array(X)
#   Ylabel=np.array(Ylabel)
#   Ylabel_4sec=np.array(Ylabel_4sec)
#   return X,Ylabel,X_4sec,Ylabel_4sec

# dirname='/content/drive/MyDrive/Colab Notebooks/Bahareh/1DCNN10SecData'
# Name='chb02_19_1.mat'
# # x,y,x2,y2=Label(dirname,Name,32,4)
