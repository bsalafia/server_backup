import pickle
import collections

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id', 'seizure_type', 'seizure_start', 'seizure_end', 'data', 'new_sig_start', 'new_sig_end', 'original_sample_frequency'])
filename_pkl = '/home/dplatnick/research/codetest2/TUH_Output_test/v1.5.2/raw_seizures/szr_0_pid_00000258_type_TCSZ.pkl'
lbl_path = '/home/dplatnick/research/codetest2/TUH_Output_test/v1.5.2/raw_seizures/szr_182_pid_00009839_type_FNSZ.lbl'


def unpickle():
    unpickled_contents = pickle.load(open(filename_pkl, 'rb'))

    return unpickled_contents


def read_lbl(path):
    # read .lbl as text (Line-by-line)
    with open(path) as f:
        test_lbl = f.readlines()
    # remove newlines
    lbl_contents = [x.strip() for x in test_lbl]

    return lbl_contents


lbl_contents = read_lbl(lbl_path)

# for line in lbl_contents:
#     print(line)


# extracting channel number/electrode name
def electrodename_channelnumbers():

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

    return channels, channel_numbers


channels = electrodename_channelnumbers()[0]
channel_numbers = electrodename_channelnumbers()[1]

print('channel names: ', channels)
print('channel numbers: ', channel_numbers)


event_list = []
for line in lbl_contents:
    if line.startswith('label'):
        event_list.append(line)
# print(event_list)

label_list = []
channel_list = []
start_time_list = []
end_time_list = []
class_label_list = []


for event in event_list:
    event_channel = int(event.split(',')[4].strip())
    # print('event channel: ', event_channel)
    channel_list.append(event_channel)

    event_start_time = float(event.split(',')[2].strip())
    # print('event start time: ', event_start_time)
    start_time_list.append(event_start_time)

    event_end_time = float(event.split(',')[3].strip())
    # print('event end time: ',  event_end_time)
    end_time_list.append(event_end_time)

    event_label = event.split('[')[1].strip()
    event_label = event_label[:-2]

    if event_label[0] == '1':
        event_label = 'NULL'

    elif event_label[5] == '1':
        event_label = 'SPSW'

    elif event_label[10] == '1':
        event_label = 'GPED'

    elif event_label[15] == '1':
        event_label = 'PLED'

    elif event_label[20] == '1':
        event_label = 'EYEM'

    elif event_label[25] == '1':
        event_label = 'ARTF'

    elif event_label[30] == '1':
        event_label = 'BCKG'

    elif event_label[35] == '1':
        event_label = 'SEIZ'

    elif event_label[40] == '1':
        event_label = 'FNSZ'

    elif event_label[45] == '1':
        event_label = 'GNSZ'

    elif event_label[50] == '1':
        event_label = 'SPSZ'

    elif event_label[55] == '1':
        event_label = 'CPSZ'

    elif event_label[60] == '1':
        event_label = 'ABSZ'

    elif event_label[65] == '1':
        event_label = 'TNSZ'

    elif event_label[70] == '1':
        event_label = 'CNSZ'

    elif event_label[75] == '1':
        event_label = 'TCSZ'

    elif event_label[80] == '1':
        event_label = 'ATSZ'

    elif event_label[85] == '1':
        event_label = 'MYSZ'

    elif event_label[90] == '1':
        event_label = 'NESZ'

    elif event_label[95] == '1':
        event_label = 'INTR'

    elif event_label[100] == '1':
        event_label = 'SLOW'

    elif event_label[105] == '1':
        event_label = 'EYEM'

    elif event_label[110] == '1':
        event_label = 'CHEW'

    elif event_label[115] == '1':
        event_label = 'SHIV'

    elif event_label[120] == '1':
        event_label = 'MUSC'

    elif event_label[125] == '1':
        event_label = 'ELPP'

    elif event_label[130] == '1':
        event_label = 'ELST'

    elif event_label[135] == '1':
        event_label = 'CALB'

    class_label_list.append(event_label)

    # print('event label: ', event_label)

label_list.append(channel_list)
label_list.append(start_time_list)
label_list.append(end_time_list)
label_list.append(class_label_list)

#label list is a matrix holding channel number, start/end time, and class label for each event in an observation
# print(label_list)

print('number of events in lbl file: ', len(label_list[0]))

for i in range(len(label_list[0])):
    print('   channel: ', label_list[0][i], '   start time: ', label_list[1][i], '   end time: ', label_list[2][i], '   class label: ', label_list[3][i])


