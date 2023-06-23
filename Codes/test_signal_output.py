import pickle
import collections
import numpy as np
from parse_label import parse_event_labels
import itertools

# please change the following to your own directory
pkl_path = '/home/baharsalafian/tuh_without_preprocess/v1.5.2/raw_seizures/szr_1002_pid_00011333_type_FNSZ.pkl'

seizure_type_data = collections.namedtuple('seizure_type_data',
                                           ['patient_id', 'seizure_type', 'seizure_start', 'seizure_end', 'data', 
                                           'new_sig_start', 'new_sig_end', 'original_sample_frequency','TSE_channels', 
                                           'label_matrix', 'tse_label_matrix','lbl_channels','data_preprocess',
                                           'TSE_channels_preprocess','lable_matrix_preprocess','lbl_channels_preprocess'])
# lbl_path = '/home/baharsalafian/TUH_test/v1.5.2/raw_seizures/szr_0_pid_00000258_type_TCSZ.lbl'


# lbl_path = lbl_filepath

#
#unpickled_contents = pickle.load(open(pkl_path, 'rb'))


#raw_sig=unpickled_contents.data_preprocess
#label_matrix=unpickled_contents.lable_matrix_preprocess
#channels_name=unpickled_contents.lbl_channels_preprocess

def unpickle(pkl_file_path):

    unpickled_contents = pickle.load(open(pkl_file_path, 'rb'))

    return unpickled_contents
pkl=unpickle(pkl_path)

data=pkl.data_preprocess
#label=pkl.tse_label_segment
current_tse_label_vec=pkl.tse_label_matrix
print(data.shape)
#print(label.shape)
# def read_lbl(label_file_path):
#     with open(label_file_path) as f:
#         test_lbl = f.readlines()
#
#     lbl_contents =k[x.strip() for x in test_lbl]
#
#     return lbl_contents
#
# lbl = read_lbl(lbl_path)


#for i in pkl:
    #print("this is i",i)

def Segmentation(DenoisedSig,Fs,Siezure_start,Siezure_end,Sig_start,Sig_end,WindowSize):
    n_channels= DenoisedSig.shape[0]
    print(DenoisedSig.shape)
    # X=np.zeros((np.int64(Sig_end)-np.int64(Sig_start)-WindowSize+1,n_channels,WindowSize*Fs))
    # Ylabel=np.zeros((np.int64(Sig_end)-np.int64(Sig_start)-WindowSize+1,1))
    n1=1
    n2=1+WindowSize
    s=Siezure_start-Sig_start+1
    e=Siezure_end-Sig_start+1
    t1 = 0
    t2 = WindowSize*Fs
    X=[]
    Ylabel=[]
    
    Sup=DenoisedSig.shape[1]
    k=0
    while t2 <= Sup:
        X.append(DenoisedSig[:,t1:min(t2,Sup)])
        if  (n2<=s or n1>=e):
            Ylabel.append(0)
        elif (n1<s and n2<e):
            Ylabel.append(min(n2-s,e-s,WindowSize)/WindowSize)
        elif (n1>=s and n2<=e):
            Ylabel.append(1)
        elif n2>=e:
            Ylabel.append(min(e-n1,e-s)/WindowSize)
    
        
        t2 = t2 + Fs
        t1 = t1 + Fs
        n2=n2+1
        n1=n1+1
        k  = k + 1
    X=np.array(X)
    Ylabel=np.array(Ylabel)
    X=np.transpose(X,(0, 2, 1))
    print("shape of X",X.shape)
    return X,Ylabel
fs=250
WindowSize=4
start=pkl.seizure_start
stop=pkl.seizure_end
desired_intv = 10
signal_new=data
seizure_duration = stop - start
new_sig_start = start - desired_intv * seizure_duration
new_sig_end = stop + desired_intv * seizure_duration
if new_sig_start < 0:
    new_sig_start = 0
if new_sig_end > len(current_tse_label_vec):
    new_sig_end = len(current_tse_label_vec)
original_start_index = int(np.floor(new_sig_start * float(fs)))
original_stop_index = int(np.floor(new_sig_end * float(fs)))

seizure_signal = signal_new[:,original_start_index:original_stop_index]

data_seg,label_seg=Segmentation(seizure_signal,fs,start,stop,new_sig_start,new_sig_end,WindowSize)
print(data_seg.shape)
print('debug')
print('debug')