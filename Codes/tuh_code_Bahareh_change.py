import os
import sys
import time
from scipy import signal
from scipy.signal import butter, lfilter, freqz
import numpy
import itertools
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

import platform
import argparse
import pandas as pd
import numpy as np
import collections
from tabulate import tabulate
import pyedflib
import re
from scipy.signal import resample
import pickle
import progressbar
import time

parameters = pd.read_csv('/home/dplatnick/research/codetest2/parameters.csv', index_col=['parameter'])
seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id', 'seizure_type',
 'seizure_start', 'seizure_end', 'data', 'new_sig_start', 'new_sig_end', 'original_sample_frequency',
 'TSE_channels', 'label_matrix', 'tse_label_matrix','lbl_channels','data_preprocess','TSE_channels_preprocess',
 'lable_matrix_preprocess','lbl_channels_preprocess','data_segment','tse_label_segment','tse_timepoints'])


def generate_data_dict(xlsx_file_name, sheet_name, tuh_eeg_szr_ver):
    seizure_info = collections.namedtuple('seizure_info', ['patient_id', 'filename', 'start_time', 'end_time'])
    data_dict = collections.defaultdict(list)

    excel_file = os.path.join(xlsx_file_name)
    data = pd.read_excel(excel_file, sheet_name=sheet_name)
    if tuh_eeg_szr_ver == 'v1.5.2':
        data = data.iloc[1:]  # remove first row
    elif tuh_eeg_szr_ver == 'v1.4.0':
        data = data.iloc[1:-4]  # remove first and last 4 rows
    else:
        exit('tuh_eeg_szr_ver %s is not supported' % tuh_eeg_szr_ver)

    col_l_file_name = data.columns[11]
    col_m_start = data.columns[12]
    col_n_stop = data.columns[13]
    col_o_szr_type = data.columns[14]
    train_files = data[[col_l_file_name, col_m_start, col_n_stop, col_o_szr_type]]
    train_files = np.array(train_files.dropna())

    for item in train_files:
        a = item[0].split('/')
        if tuh_eeg_szr_ver == 'v1.5.2':
            patient_id = a[4]
        elif tuh_eeg_szr_ver == 'v1.4.0':
            patient_id = a[5]
        else:
            exit('tuh_eeg_szr_ver %s is not supported' % tuh_eeg_szr_ver)

        v = seizure_info(patient_id=patient_id, filename=item[0], start_time=item[1], end_time=item[2])
        k = item[3]  # szr_type
        data_dict[k].append(v)


    return data_dict

def print_type_information(data_dict):
    l = []
    for szr_type, szr_info_list in data_dict.items():
        # how many different patient id for seizure K?
        patient_id_list = [szr_info.patient_id for szr_info in szr_info_list]
        unique_patient_id_list, counts = np.unique(patient_id_list, return_counts=True)

        dur_list = [szr_info.end_time - szr_info.start_time for szr_info in szr_info_list]
        total_dur = sum(dur_list)
        # l.append([szr_type, str(len(szr_info_list)), str(len(unique_patient_id_list)), str(total_dur)])
        l.append([szr_type, (len(szr_info_list)), (len(unique_patient_id_list)), (total_dur)])

        #  numpy.asarray((unique, counts)).T
        '''
        if szr_type=='TNSZ':
            print('TNSZ Patient ID list:')
            print(np.asarray((unique_patient_id_list, counts)).T)
        if szr_type=='SPSZ':
            print('SPSZ Patient ID list:')
            print(np.asarray((unique_patient_id_list, counts)).T)
        '''


    sorted_by_szr_num = sorted(l, key=lambda tup: tup[1], reverse=True)
    print(tabulate(sorted_by_szr_num, headers=['Seizure Type', 'Seizure Num', 'Patient Num', 'Duration(Sec)']))


def merge_train_test(train_data_dict, dev_test_data_dict):
    merged_dict = collections.defaultdict(list)
    for item in train_data_dict:
        merged_dict[item] = train_data_dict[item] + dev_test_data_dict[item]


    return merged_dict


def extract_signal(f, signal_labels, electrode_name, start, stop):
    tuh_label = [s for s in signal_labels if 'EEG ' + electrode_name + '-' in s]

    if len(tuh_label) > 1:
        print(tuh_label)
        exit('Multiple electrodes found with the same string! Abort')

    channel = signal_labels.index(tuh_label[0])
    signal = np.array(f.readSignal(channel))


    original_sample_frequency = f.getSampleFrequency(channel)

    start_raw, stop_raw = float(0), len(signal)/original_sample_frequency
    start, stop = float(start), float(stop)

    # changing seizure time
    seizure_duration = stop - start
    desired_intv = 10
    new_sig_start = start - desired_intv * seizure_duration
    new_sig_end = stop + desired_intv * seizure_duration

    if new_sig_start < 0:
        new_sig_start = 0
    if new_sig_end > len(signal)/original_sample_frequency:
        new_sig_end = len(signal)/original_sample_frequency


    # new_sig_start_index = int(np.floor(new_sig_start * float(original_sample_frequency)))

    # new_sig_stop_index = int(np.floor(new_sig_end * float(original_sample_frequency)))


    #read in eeg signal
    # new_signal = signal[new_sig_start_index:new_sig_stop_index]
    seizure_signal = signal

    # original_sample_frequency = f.getSampleFrequency(channel)
    # original_start_index = int(np.floor(start * float(original_sample_frequency)))
    # original_stop_index = int(np.floor(stop * float(original_sample_frequency)))
    # seizure_signal = signal[original_start_index:original_stop_index]

    new_sample_frequency = int(parameters.loc['sampling_frequency']['value'])
    raw_sig_duration=f.getFileDuration()
    # print("origianl and sample", original_sample_frequency,new_sample_frequency)
    # time.sleep(2)
    new_num_time_points = int(raw_sig_duration * new_sample_frequency)

    # Resampling signal
    seizure_signal_resampled = resample(seizure_signal, new_num_time_points)
    # seizure_signal_resampled = resample(sliced_seizure_signal, new_num_time_points)

    return seizure_signal_resampled, new_sig_start, new_sig_end, original_sample_frequency


def read_edfs_and_extract(edf_path, edf_start, edf_stop):

    f = pyedflib.EdfReader(edf_path)


    dur=f.getFileDuration()
    # print("duration of whole singal", dur)
    # time.sleep(5)

    montage = str(parameters.loc['montage']['value'])
    montage_list = re.split(';', montage)
    signal_labels = f.getSignalLabels()
    x_data = []

    for i in montage_list:
        electrode_list = re.split('-', i)
        electrode_1 = electrode_list[0]
        # print("elec name", electrode_1)
        extracted_signal_from_electrode_1 = extract_signal(f, signal_labels, electrode_name=electrode_1,
                                                           start=edf_start, stop=edf_stop)[0]
        electrode_2 = electrode_list[1]
        extracted_signal_from_electrode_2 = extract_signal(f, signal_labels, electrode_name=electrode_2,
                                                           start=edf_start, stop=edf_stop)[0]
        this_differential_output = extracted_signal_from_electrode_1 - extracted_signal_from_electrode_2
        x_data.append(this_differential_output)



    new_sig_start = extract_signal(f, signal_labels, electrode_name=electrode_1,
                   start=edf_start, stop=edf_stop)[1]
    new_sig_end = extract_signal(f, signal_labels, electrode_name=electrode_1,
                                                           start=edf_start, stop=edf_stop)[2]

    original_sample_frequency = extract_signal(f, signal_labels, electrode_name=electrode_1,
                                                           start=edf_start, stop=edf_stop)[3]

    f._close()
    del f

    x_data = np.array(x_data)

    return x_data, new_sig_start, new_sig_end, original_sample_frequency,montage_list


def load_edf_extract_seizures_v140(base_dir, save_data_dir, data_dict):
    seizure_data_dict = collections.defaultdict(list)

    count = 0
    bar = progressbar.ProgressBar(maxval=sum(len(v) for k, v in data_dict.items()),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for seizure_type, seizures in data_dict.items():
        for seizure in seizures:
            rel_file_location = seizure.filename.replace('.tse', '.edf').replace('./', '')
            patient_id = seizure.patient_id
            abs_file_location = os.path.join(base_dir, rel_file_location)
            temp = seizure_type_data(patient_id=patient_id, seizure_type=seizure_type,
                                     data=read_edfs_and_extract(abs_file_location, seizure.start_time,
                                                                seizure.end_time)[0])
            with open(os.path.join(save_data_dir,
                                   'szr_' + str(count) + '_pid_' + patient_id + '_type_' + seizure_type + '.pkl'),
                      'wb') as fseiz:
                pickle.dump(temp, fseiz)
            count += 1
            bar.update(count)
    bar.finish()

    return seizure_data_dict


def load_edf_extract_seizures_v152(base_dir, save_data_dir, data_dict):
    seizure_data_dict = collections.defaultdict(list)

    count = 0
    bar = progressbar.ProgressBar(maxval=sum(len(v) for k, v in data_dict.items()),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for seizure_type, seizures in data_dict.items():
        for seizure in seizures:
            rel_file_location = seizure.filename.replace('.tse', '.edf').replace('./', 'edf/')
            patient_id = seizure.patient_id

            # copy seizure lbl file into output directory
            rel_lbl_file_location = seizure.filename.replace('.tse', '.lbl').replace('./', 'edf/')
            rel_tse_file_location = seizure.filename.replace('.tse', '.tse').replace('./', 'edf/')
            cp_lbl_files = ('cp ' + os.path.join(base_dir, rel_lbl_file_location) + ' ' + save_data_dir + '/' + 'szr_' + str(count) + '_pid_' + patient_id + '_type_' + seizure_type + '.lbl')
            os.system(cp_lbl_files)
            cp_lbl_files = ('cp ' + os.path.join(base_dir, rel_tse_file_location) + ' ' + save_data_dir + '/' + 'szr_' + str(count) + '_pid_' + patient_id + '_type_' + seizure_type + '.tse')
            os.system(cp_lbl_files)


            lbl_abs_path = os.path.join(base_dir,rel_lbl_file_location)
            tse_abs_path = os.path.join(base_dir,rel_tse_file_location)

            ###CODE PREPROCESSING LBL FILE###
            def preprocess_lbl(lbl_filepath):

                from parse_label import parse_event_labels

                lbl_path = lbl_filepath

                def read_lbl(label_file_path):

                    with open(label_file_path) as f:
                        test_lbl = f.readlines()

                    lbl_contents = [x.strip() for x in test_lbl]

                    channels = []
                    channel_numbers = []

                    for line in lbl_contents:
                        if line.startswith('montage'):
                            chan_num = line.split(',')[0]
                            chan_num = [int(num) for num in chan_num.split() if num.isdigit()]
                            chan_num = chan_num[0]
                            montage_name = line.split(':')[0]
                            montage_name = montage_name.split(',')[1]
                            montage_name = montage_name[1:]

                            channel_numbers.append(chan_num)
                            channels.append(montage_name)
                #return lbl_contents
                    return lbl_contents,channels, channel_numbers

                # channels = electrodename_channelnumbers()[0]
                # channel_numbers = electrodename_channelnumbers()[1]

                # print('channel names: ', channels)
                # print('channel numbers: ', channel_numbers)

                lbl_contents,channels, channel_numbers= read_lbl(lbl_path)

                def create_event_list():

                    event_list = []
                    for line in lbl_contents:
                        if line.startswith('label'):
                            event_list.append(line)
                    return event_list

                event_list = create_event_list()
                # label_list is a matrix holding channel number, start, end time, and class label for each event in an observation
                label_list = parse_event_labels(event_list)

                number_of_events = len(label_list[0])

                print(f'number of events in lbl file: {number_of_events}')

                signal_length = round(label_list[2][-1])

                # print(signal_length)
                # print(rounded_start_times)

                def preprocess_label_list(label_list):
                    # label vector has dimensions NxT, where N is the number of channels and T is the number of time steps (seconds)
                    label_vector = []

                    for event in range(number_of_events - 1):
                        # print(f'channel: {label_list[0][event]}   start time: {label_list[1][event]}    end time: {label_list[2][event]}    class label: {label_list[3][event]}    rounded start times: {label_list[4][event]}    rounded end times: {label_list[5][event]}')
                        # print(f'channel: {label_list[0][event]}   class label: {label_list[3][event]}    rounded start times: {label_list[4][event]}    rounded end times: {label_list[5][event]}')
                        # print(label_list[0][event], label_list[3][event], label_list[4][event], label_list[5][event])

                        current_channel_num = label_list[0][event]
                        next_channel_num = label_list[0][event + 1]
                        current_start_time = label_list[4][event]
                        current_end_time = label_list[5][event]
                        current_label = label_list[3][event]

                        if current_channel_num != next_channel_num:
                            label_vector.append(list())
                            for seconds in range(current_end_time - current_start_time):
                                label_vector[current_channel_num].append(current_label)

                        elif current_channel_num == next_channel_num:
                            try:
                                label_vector[current_channel_num]
                            except:
                                label_vector.append(list())
                            for seconds in range(current_end_time - current_start_time):
                                label_vector[current_channel_num].append(label_list[3][event])

                    # edge case of final event
                    final_channel, final_end_time, final_label, final_start_time = label_list[0][-1], label_list[5][-1], \
                                                                                   label_list[3][-1], label_list[4][-1]
                    for seconds in range(final_end_time - final_start_time):
                        label_vector[-1].append(final_label)

                    return label_vector

                label_vector = preprocess_label_list(label_list)

                symbol_vector = ['NULL', 'SPSW', 'GPED', 'PLED', 'EYBL', 'ARTF', 'BCKG', 'SEIZ', 'FNSZ', 'GNSZ', 'SPSZ',
                                 'CPSZ', 'ABSZ', 'TNSZ', 'CNSZ', 'TCSZ', 'ATSZ', 'MYSZ', 'NESZ', 'INTR', 'SLOW', 'EYEM',
                                 'CHEW', 'SHIV', 'MUSC', 'ELPP', 'ELST', 'CALB']

                def label_vec_str_to_int(lbl_vec):

                    for channel in lbl_vec:
                        for index, label in enumerate(channel):
                            for ind, symbol in enumerate(symbol_vector):
                                if label == symbol:
                                    channel[index] = ind
                                    break
                                else:
                                    pass

                    return lbl_vec

                label_vector = label_vec_str_to_int(label_vector)

                label_vector = np.asarray(label_vector)

                return label_vector,channels,channel_numbers

            current_label_vec ,channels,_= preprocess_lbl(lbl_abs_path)

            def get_tse_label_vector(tse_filepath):

                tse_path = tse_filepath

                def read_tse(tse_file_path):

                    with open(tse_file_path) as f:
                        tse_file = f.readlines()

                    tse_list = [x.strip() for x in tse_file]
                    tse_list = tse_list[2:]

                    return tse_list

                tse_contents = read_tse(tse_path)

                start_times = []
                end_times = []
                class_labels = []
                rounded_start_times = []
                rounded_end_times = []

                for line in tse_contents:
                    line = line.split(' ')
                    start_times.append(float(line[0]))
                    rounded_start_times.append(round(float(line[0])))
                    end_times.append(float(line[1]))
                    rounded_end_times.append(round(float(line[1])))
                    class_labels.append(line[2])

                file_duration = rounded_end_times[-1]
                number_of_events = len(tse_contents)

                label_intervals = rounded_end_times

                label_list = []
                label_list.append(rounded_start_times)
                label_list.append(rounded_end_times)
                label_list.append(class_labels)

                label_vector = []

                for event in range(number_of_events):
                    current_start_time = rounded_start_times[event]
                    current_end_time = rounded_end_times[event]

                    for seconds in range(current_end_time - current_start_time):
                        label_vector.append(class_labels[event])

                symbol_vector = ['NULL', 'SPSW', 'GPED', 'PLED', 'EYBL', 'ARTF', 'BCKG', 'SEIZ', 'FNSZ', 'GNSZ', 'SPSZ',
                                 'CPSZ', 'ABSZ', 'TNSZ', 'CNSZ', 'TCSZ', 'ATSZ', 'MYSZ', 'NESZ', 'INTR', 'SLOW', 'EYEM',
                                 'CHEW', 'SHIV', 'MUSC', 'ELPP', 'ELST', 'CALB']

                for index, label in enumerate(label_vector):
                    label = label.upper()
                    for ind, symbol in enumerate(symbol_vector):
                        if label == symbol:
                            label_vector[index] = ind
                            break
                        else:
                            pass

                tse_label_vector = label_vector

                return tse_label_vector

###############Additional preprocessing steps####################
            def channels_check(ch1_name,ch2_name):

                both = set(ch1_name).intersection(ch2_name)
                indices_ch2 = [ch2_name.index(x) for x in both]

                # print(indices_lbl)

                indices_ch2.sort()
                ch2_name_new=[]
                for i in indices_ch2:

                    ch2_name_new.append(ch2_name[i])

                return ch2_name_new,indices_ch2

            def apply_notchfilt(data):

                dataT = data.T
                f0 = 60.0  # Frequency to be removed from signal (Hz)

                Q = 30.0  # Quality factor

                # Design notch filter
                Fs=250

                b, a = signal.iirnotch(f0, Q, Fs)

                # Apply notch filter to the noisy signal using signal.filtfilt

                outputSignal = signal.filtfilt(b, a, dataT)
                outputSignal = outputSignal.T

                return  outputSignal

            def butter_highpass(cutoff, fs, order=5):

                nyq = 0.5 * fs
                normal_cutoff = cutoff / nyq
                b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
                return b, a

            def butter_highpass_filter(data, cutoff, fs, order=5):
                # print("This is a test")
                dataT = data.T
                b, a = butter_highpass(cutoff, fs, order=order)
                y = signal.filtfilt(b, a, dataT)
                y=y.T
                return y

            def preprocess_output(data,tse_channels,lbl_channels,lbl_matrix):

                common_channels=['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'T3-C3', 'C3-CZ',
                                 'CZ-C4', 'C4-T4', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']

                tse_channels_new,indices_tse_channels=channels_check(common_channels, tse_channels)

                data_new=data[indices_tse_channels]
                data_denoise=apply_notchfilt(data_new)
            
            

                ######
                b, a = butter_highpass(cutoff=0.5, fs=250, order=5)
                data_denoise = butter_highpass_filter(data_denoise, cutoff=0.5, fs=250, order=5)

                ######
                lbl_channels_new,lbl_indices=channels_check(tse_channels_new,lbl_channels)

                lbl_matrix_new=lbl_matrix[lbl_indices]

                return  data_denoise,tse_channels_new,lbl_matrix_new,lbl_channels_new
            
            def Segmentation(DenoisedSig,Fs,Siezure_start,Siezure_end,Sig_start,Sig_end,WindowSize):

                n_channels= DenoisedSig.shape[0]

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
                return X,Ylabel



            current_tse_label_vec = get_tse_label_vector(tse_abs_path)
            current_tse_label_vec = numpy.array(current_tse_label_vec)



            abs_file_location = os.path.join(base_dir, rel_file_location)

            data=read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time)[0]
            tse_channels=read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time)[4]
            lbl_channels=channels
            lbl_matrix = current_label_vec
            data_new,tse_channels_new,lbl_matrix_new,lbl_channels_new=preprocess_output(data, tse_channels, lbl_channels, lbl_matrix)

            WindowSize=4
            fs=250
            tse_label_expand=[]
            for i in range(len(current_tse_label_vec)):

                tse_label_expand.append(list(itertools.repeat(current_tse_label_vec[i], fs)))

            tse_label_expand=np.concatenate(tse_label_expand,axis=0)
           


            ####### spliting the data to 10  times before and after

            start=seizure.start_time
            stop=seizure.end_time
            desired_intv = 10

            signal_new=data_new

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
            tse_split=tse_label_expand[original_start_index:original_stop_index] 

            seizure_index=[7,8,9,10,11,12,13,14,15,16,17]
            tse_split = [1 if ((x==seizure_index).any()) else 0 for x in tse_split]

            data_seg,label_seg=Segmentation(seizure_signal,fs,start,stop,new_sig_start,new_sig_end,WindowSize)


            
            try:
                temp = seizure_type_data(patient_id=patient_id, seizure_type=seizure_type, seizure_start=seizure.start_time, seizure_end=seizure.end_time,
                                         data=read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time)[0],
                                         new_sig_start = read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time)[1],
                                         new_sig_end = read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time)[2],
                                         original_sample_frequency = read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time)[3],
                                         TSE_channels=read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time)[4],
                                         label_matrix=current_label_vec, tse_label_matrix=current_tse_label_vec,lbl_channels=channels,
                                         data_preprocess=data_new,TSE_channels_preprocess=tse_channels_new,lable_matrix_preprocess=lbl_matrix_new,
                                         lbl_channels_preprocess=lbl_channels_new,data_segment=data_seg,
                                         tse_label_segment=label_seg,tse_timepoints=tse_split)
                with open(os.path.join(save_data_dir, 'szr_' + str(count) + '_pid_' + patient_id + '_type_' + seizure_type + '.pkl'), 'wb') as fseiz:
                    pickle.dump(temp, fseiz)
                count += 1
            except Exception as e:
                print(e)
                print(rel_file_location)

            bar.update(count)
    bar.finish()

    return seizure_data_dict


# to convert raw edf data into pkl format raw data
def gen_raw_seizure_pkl(args, tuh_eeg_szr_ver, anno_file):
    base_dir = args.base_dir

    save_data_dir = os.path.join(args.save_data_dir, tuh_eeg_szr_ver, 'raw_seizures')
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    raw_data_base_dir = os.path.join(base_dir, tuh_eeg_szr_ver)
    szr_annotation_file = os.path.join(raw_data_base_dir, '_DOCS', anno_file)

    # For training files
    print('Parsing the seizures of the training set...\n')
    train_data_dict = generate_data_dict(szr_annotation_file, 'train', tuh_eeg_szr_ver)
    print('Number of seizures by type in the training set...\n')
    print_type_information(train_data_dict)
    print('\n\n')

    # For dev files
    if tuh_eeg_szr_ver == 'v1.5.2':
        dev_name = 'dev'
    elif tuh_eeg_szr_ver == 'v1.4.0':
        dev_name = 'dev_test'
    else:
        exit('tuh_eeg_szr_ver %s is not supported' % tuh_eeg_szr_ver)

    print('Parsing the seizures of the validation set...\n')
    dev_test_data_dict = generate_data_dict(szr_annotation_file, dev_name, tuh_eeg_szr_ver)
    print('Number of seizures by type in the validation set...\n')
    print_type_information(dev_test_data_dict)
    print('\n\n')

    # Now we combine both
    print('Combining the training and validation set...\n')
    merged_dict = merge_train_test(dev_test_data_dict, train_data_dict)
    # merged_dict = merge_train_test(train_data_dict,dev_test_data_dict)
    print('Number of seizures by type in the combined set...\n')
    print_type_information(merged_dict)
    print('\n\n')

    # Extract the seizures from the edf files and save them
    if tuh_eeg_szr_ver == 'v1.5.2':
        seizure_data_dict = load_edf_extract_seizures_v152(raw_data_base_dir, save_data_dir, merged_dict)
    elif tuh_eeg_szr_ver == 'v1.4.0':
        seizure_data_dict = load_edf_extract_seizures_v140(raw_data_base_dir, save_data_dir, merged_dict)
    else:
        exit('tuh_eeg_szr_ver %s is not supported' % tuh_eeg_szr_ver)

    print_type_information(seizure_data_dict)
    print('\n\n')


def main():
    parser = argparse.ArgumentParser(description='Build data for TUH EEG data')

    if platform.system() == 'Linux':
        parser.add_argument('--base_dir', default='/media/datadrive/seizuredetection/TUHData',
                            help='path to raw seizure dataset')
        parser.add_argument('--save_data_dir', default='/home/baharsalafian/TUH_Bahareh_experiment',
                            help='path to save processed data')
    elif platform.system() == 'Darwin':
        parser.add_argument('--base_dir', default='/media/datadrive/seizuredetection/TUHData',
                            help='path to raw seizure dataset')
        parser.add_argument('--save_data_dir',
                            default='/home/baharsalafian/TUH_Bahareh_experiment',
                            help='path to save processed data')
    else:
        print('Unknown OS platform %s' % platform.system())
        exit()

    parser.add_argument('-v', '--tuh_eeg_szr_ver',
                        default='v1.5.2',
                        # default='v1.4.0',
                        help='version of TUH seizure dataset')

    args = parser.parse_args()
    tuh_eeg_szr_ver = args.tuh_eeg_szr_ver

    if tuh_eeg_szr_ver == 'v1.4.0':  # for v1.4.0
        anno_file = 'seizures_v31r.xlsx'
        gen_raw_seizure_pkl(args, tuh_eeg_szr_ver, anno_file)
    elif tuh_eeg_szr_ver == 'v1.5.2':  # for v1.5.2
        anno_file = 'seizures_v36r.xlsx'
        gen_raw_seizure_pkl(args, tuh_eeg_szr_ver, anno_file)
    else:
        exit('Not supported version number %s' % tuh_eeg_szr_ver)


if __name__ == '__main__':
    main()
