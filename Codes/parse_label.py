def parse_event_labels(events):
    channel_list = []
    start_time_list = []
    end_time_list = []
    class_label_list = []
    label_list = []

    for event in events:

        event_channel = int(event.split(',')[4].strip())
        channel_list.append(event_channel)

        event_start_time = float(event.split(',')[2].strip())
        start_time_list.append(event_start_time)

        event_end_time = float(event.split(',')[3].strip())
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
            event_label = 'EYBL'

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

    rounded_start_times = [round(start_time) for start_time in start_time_list]
    rounded_end_times = [round(start_time) for start_time in end_time_list]

    label_list.append(rounded_start_times)
    label_list.append(rounded_end_times)

    return label_list
