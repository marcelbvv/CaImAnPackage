import numpy as np
from numba.typed import List
from numba import njit
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, resample
import pyabf
import statistics
import easygui
import pandas as pd
import time
import random
import scipy.stats as stats
import itertools

import save_session
 
"""
To do:
    - Correct test function
    - Remove unnecessary modules
"""


def get_data():
    path = easygui.diropenbox(title='Select folder containing traces')
    files = easygui.fileopenbox(title='Select files to analyze', multiple=True)
    return path, files


def set_settings_Digidata():
    settings = {
        'sampling frequency':2000,
        'final_sampling_frequency': 30,
        'binary conversion threshold':1.5,
        'cpr':1000,
        'perimeter_cm':25,
        'holding samples':10000,
        'excess samples':20000,
        'time rec':300,
        }
    return settings

def set_settings_Igor():
    settings = {
        'sampling frequency':2000,
        'final_sampling_frequency': 30,
        'binary conversion threshold':1.5,
        'cpr':1000,
        'perimeter_cm':25,
        'holding samples':200,
        'excess samples':200,
        'time rec':300,
        }
    return settings

def import_data_Digidata(settings, path):
    '''
    Imports data from rotary encoder
    ---------------
    Input:
        path to .abf file 
    Output:
        numpy array with each channel
    '''
    abf = pyabf.ABF(path)
    abf.setSweep(sweepNumber=0, channel=0)
    ch_A = abf.sweepY
    abf.setSweep(sweepNumber=0, channel=1)
    ch_B = abf.sweepY
    ch_A = ch_A[(settings['holding samples']):(len(ch_A)-settings['holding samples']-settings['excess samples'])]
    ch_B = ch_B[settings['holding samples']:len(ch_B)-settings['holding samples']-settings['excess samples']]
    return ch_A, ch_B

def import_data_Igor(settings, path):
    '''
    Imports data from rotary encoder
    ---------------
    Input:
        path to .abf file 
    Output:
        numpy array with each channel
    '''
    file = pd.read_csv(path, sep='\t')
    columns = list(file.columns)
    
    if len(columns) >= 3:
        ch_A = file[[columns[0]]].to_numpy()
        ch_B = file[[columns[1]]].to_numpy()
        ch_C = file[[columns[2]]].to_numpy()
        
        ch_A = ch_A[(settings['holding samples']):(len(ch_A)-settings['holding samples']-settings['excess samples'])]
        ch_B = ch_B[settings['holding samples']:len(ch_B)-settings['holding samples']-settings['excess samples']]
        ch_C = ch_C[settings['holding samples']:len(ch_C)-settings['holding samples']-settings['excess samples']]
        return [ch_A, ch_B, ch_C]
        
    elif len(columns) == 2:
        ch_A = file[[columns[0]]].to_numpy()
        ch_B = file[[columns[1]]].to_numpy()
        ch_A = ch_A[(settings['holding samples']):(len(ch_A)-settings['holding samples']-settings['excess samples'])]
        ch_B = ch_B[settings['holding samples']:len(ch_B)-settings['holding samples']-settings['excess samples']]
    
        return [ch_A, ch_B]

@njit
def convert_binary(trace, thr):
    '''
    Converts Analog Files to Binary (0, 1) using a given threshold (thr)
    ---------------
    Input:
        trace - trace to convert
        thr - threshold to use
    
    Output:
        Converted trace
    '''
    binary = List()
    for x in trace:
        if x >= thr:
            binary.append(1)
        else:
            binary.append(0)
    return binary

def position(A, B, settings):
    '''
    Takes traces A and B and converts it to a trace that has the same number of
    points but with positions points.
    ---------------
    Input:
        A, B - traces to convert
    
    Output:
        Positions through time
    '''
    positions = [0]
    a_last = 0
    b_last = 0
    for nA, nB in zip(A, B):
        if nA != a_last and nA == 1 and nB == 0:
            positions.append(positions[-1]+1)
        elif nB != b_last and nB == 1 and nA == 0:
            #test
            # positions.append(positions[-1]-1)  
            positions.append(positions[-1]+1)  
        else:
            positions.append(positions[-1])
        a_last = nA
        b_last = nB
    positions.pop(0)
    for i, v in enumerate(positions):
        positions[i] = v * settings['perimeter_cm']/settings['cpr']
    if positions[-1] <= positions[0]:
        positions = [i * -1 for i in positions]
        # for i in positions:
        #     positions[i] = positions[i]*-1
    return positions

def calc_speed(positions, settings, n_samples):
    '''
    Takes the positions through time and calculate the change between every
    n_samples.
    ---------------
    Input:
        positions - list containing the position at every time point. 
        n_samples - time points to jump when calculating the delta position.
        
    Output:
        speed - speed of the mouse at every time point in cm/s
        t - time distribution in seconds
    '''
    #factor = round(settings['sampling frequency']/settings['final_sampling_frequency']-0.5)
    #target_down = settings['final_sampling_frequency'] * settings['time rec']
    #positions_downsampled = positions[0:len(positions):factor]
    #speed = np.gradient(positions_downsampled)
    #speed = resample(speed, target_down)
    #speed_filt = savgol_filter(speed, 11, 1)
    #t = np.linspace(0,settings['time rec'],len(speed))
    
    if settings['sampling frequency'] % 2 > 0:
        window = settings['sampling frequency']
    else:
        window = settings['sampling frequency']+1
    positions_smooth = savgol_filter(positions, window, 5)
    t_positions = np.linspace(0,settings['time rec'],len(positions))
    speed_full = np.gradient(positions_smooth, t_positions)
    speed_downsampled = resample(speed_full, n_samples)
    speed_smooth = savgol_filter(speed_downsampled, 31, 5)
    speed = np.where(speed_smooth<0, 0, speed_smooth)
    t = np.linspace(0,settings['time rec'],len(speed))
    
    return speed, t

def move_nomove(data_speed, settings):
    '''
    Converts the raw data into an array of moving/no moving 
    determined by a minimum speed.
    ---------------        
    Output:
        binary_movement - array with 0 or 1 corresponding to still or moving.
        bool_binary_movement - boolean array corresponding to still or moving
    '''
    binary_movement = (data_speed > settings['speed threshold']) * 1
    bool_binary_movement = (data_speed > settings['speed threshold'])
    return binary_movement, bool_binary_movement

def extend_movenomove(binary_movement, settings):
    original = np.copy(binary_movement)
    ext_move = np.copy(binary_movement)
    before = original[0]
    n_before = int(settings['time_before'] * settings['final_sampling_frequency'])
    array_before = np.full((1,n_before),1)
    n_after = int(settings['time_after'] * settings['final_sampling_frequency'])
    array_after = np.full((1,n_after),1)
    for i, v in enumerate(original):
        if v == before:
            before = v
        elif v > before and i < n_before:
            ext_move[0:i] = array_before[0,0:i]
            before = v
        elif v > before:
            ext_move[(i-n_before):i] = array_before
            before = v
        elif v < before and i >= (len(ext_move)-n_after):
            ext_move[i:len(ext_move)] = array_after[0,0:(len(ext_move)-i)]
            before = v
        elif v < before:
            ext_move[i:i+n_after] = array_after
            before = v
        else:
            pass
    return ext_move

def create_binary_trace(len_recording=9000, n_events=20, max_duration=100):
   """
   Creates a list of 1 and 0  of length 'len_recording' with 'n_events' of 
   duration between 'min_duration' and 'max_duration'.
   """
   
   # Start of each event
   start_time = random.sample(range(len_recording-max_duration), n_events)
   
   # Duration
   duration = []
   for i in range(n_events):
       duration.append(random.randint(1, max_duration))
       
   # Create trace
   trace = [0] * len_recording
   for t, d in zip(start_time, duration):
       trace[t:t+d] = [1] * d
   
   return trace


def calc_event_duration(trace):    
    """
    Calculates the duration of the periods where trace is 1
    
    Inputs:
        trace - list of 0s and 1s
    Outputs:
        delta - list of event durations
        loc - list of event locations
    """
    # Detect start and end of event
    change = list(np.where(np.diff(trace,prepend=np.nan))[0])
    
    # Calculate lenght of each event
    delta = []
    loc = []
    if trace[change[1]] == 0: # Mouse is running at start
        delta.append(change[1])
        loc.append((0, change[1]))
        
        if (len(change)-2)%2 == 0:
            for i in range(2, len(change), 2):
                delta.append(change[i+1]-change[i])
                loc.append((change[i], change[i+1]))
        else:
            for i in range(2, len(change)-1, 2):
                delta.append(change[i+1]-change[i])
                loc.append((change[i], change[i+1]))
            delta.append(len(trace) - change[-1])  
            loc.append((change[-1], len(trace)))
        
    else: # Mouse is resting at start
        if (len(change)-1)%2 == 0:
            for i in range(1, len(change), 2):
                delta.append(change[i+1]-change[i])
                loc.append((change[i], change[i+1]))
        else:
            for i in range(1, len(change)-1, 2):
                delta.append(change[i+1]-change[i])
                loc.append((change[i], change[i+1]))
            delta.append(len(trace) - change[-1])  
            loc.append((change[-1], len(trace)))
            
    return delta, loc

def aligned_events(dF, event_loc, settings, type_data, save_location):
    
    # Settings
    if type_data == 'locomotion':
        t_before = settings['align locomotion']['onset t_before']
        t_after = settings['align locomotion']['onset t_after']
    elif type_data == 'whisking' or type_data == 'whisking only':
        t_before = settings['align whisking']['onset t_before']
        t_after = settings['align whisking']['onset t_after']

    fs = settings['fs']

    # extract traces
    trace = []
    delta_before = int(t_before * fs)
    delta_after = int(t_after * fs)
    for event in enumerate(event_loc):
        loc = event[1][0]
        if loc-delta_before > 0 and loc+delta_after < dF.shape[1]:
            #extract dF 
            dF_section = dF[:,(loc-delta_before):(loc+delta_after)]
            trace.append(dF_section)
        
    # calculate mean response per cell
    for i in range(0, len(trace)):
        trace[i] = np.expand_dims(trace[i], -1)
    combined_matrix = np.concatenate(trace, axis=2)
    mean_combined = np.mean(combined_matrix, axis=2)    
    
    time_array = np.linspace(-t_before, t_after, delta_after+delta_before)
    
    # Create dictionary
    dic = {'all_traces': combined_matrix,
           'response per cell': mean_combined,
           'time': time_array}
    
    # plot results
    plt.figure()
    for i in range(0, len(mean_combined)):
        plt.plot(time_array, mean_combined[i], color='grey')
    plt.plot(time_array, np.mean(mean_combined, axis=0), color='red')
    plt.savefig('{}/Aligned_{}'.format(save_location, type_data))
    plt.savefig('{}/Aligned_{}.pdf'.format(save_location, type_data))
    plt.close()
    
    return mean_combined, dic


def remove_short_events(binary_trace, delta, loc, fs=30, min_duration=1):
    """
    Removes events from 'trace' whose duration is shorter than 'min_duration'
    
    Inputs:
        binary_trace - list of 0s and 1s
        delta - list of event durations 
        loc - list of event locations
        fs - sampling rate
        min_duration - minimum event duration is seconds
        
    Output:
        new_trace - similar to 'trace' but without events shorter than 'min_duration'
    """
    new_trace = binary_trace.copy()
    delta_s = list(np.divide(delta.copy(), fs))
    pop = []
    for i, (d,loc) in enumerate(zip(delta_s, loc)):
        if d < min_duration:
           new_trace[loc[0]:loc[1]] = [0]*delta[i]
           pop.append(i)
    new_delta = []
    new_loc = []
    for i, (d, l) in enumerate(zip(delta, loc)):
        if i in pop:
            pass
        else:
            new_delta.append(d)
            new_loc.append(l)
            
    return new_trace, new_delta, new_loc

def remove_short_interevent_periods(binary_trace, loc, sampling_rate, max_interevent=1):
    
    # New binary trace
    new_binary = np.copy(binary_trace)
    
    # Maximum number of frames accepted between bouts
    n_frames = int(round(sampling_rate * max_interevent))
    
    # Calculate frames between bouts
    list_intervals = list()
    for i in range(0,len(loc)-1):
        if loc[i+1][0]-loc[i][1] <= n_frames:
            list_intervals.append([loc[i][1],loc[i+1][0]])
            
    print(list_intervals)
    
    if len(list_intervals) > 0:
        for i in list_intervals:
            new_binary[i[0]:i[1]] = 1
    
    delta, loc = calc_event_duration(new_binary)
    
    return new_binary, delta, loc


def plot_positions(positions, speed, binary_movement, t, settings):
    t_positions = np.linspace(0,len(positions)/settings['sampling frequency'],len(positions))
    plt.subplot(3, 1, 1)
    plt.plot(t_positions, positions)
    plt.subplot(3, 1, 2)
    plt.plot(t, speed)
    plt.subplot(3, 1, 3)
    plt.plot(t, binary_movement)
    return

def single_getspeed(file_path, n_samples, speed_threshold, time_before, time_after, fs, remove_short_events, min_event_duration):
    if file_path.endswith('.txt'):
        settings = set_settings_Igor()
        trace_A, trace_B = import_data_Igor(settings, file_path)
    elif file_path.endswith('.abf'):
        settings = set_settings_Digidata()
        trace_A, trace_B = import_data_Digidata(settings, file_path)
    else:
        print('File not recognized')
        
    #directory = os.path.dirname(file_path)
    #file = file_path.split('\\')[-1]
    settings['n_samples'] = n_samples
    settings['speed threshold'] = speed_threshold
    settings['time_before'] = time_before
    settings['time_after'] = time_after
    bi_A = convert_binary(trace_A,settings['binary conversion threshold'])
    bi_B = convert_binary(trace_B, settings['binary conversion threshold'])
    positions = position(bi_A, bi_B, settings)
    speed, t = calc_speed(positions, settings, n_samples)
    binary_movement, bool_binary_movement = move_nomove(speed, settings)
    extended_binary_movement = extend_movenomove(binary_movement, settings)
    n_changes = np.count_nonzero(binary_movement)
    if n_changes == len(binary_movement) or len(binary_movement)-n_changes == 0:
        events = {}
        events['duration'] = []
        events['location'] = []
        events['mean duration'] = []
        events['max duration'] = []
        
    else:
        events = {}
        events['duration'], events['location'] = calc_event_duration(binary_movement)
        events['mean duration'] = statistics.mean(events['duration'])/fs
        events['max duration'] = max(events['duration'])/fs
    
        if remove_short_events == True:
            new_trace = binary_movement.copy()
            delta_s = list(np.divide(events['duration'].copy(), fs))
            loc = events['location'].copy()
            delta = events['duration'].copy()
            pop = []
            for i, (d,l) in enumerate(zip(delta_s, loc)):
                if d < min_event_duration:
                   new_trace[l[0]:l[1]] = [0]*delta[i]
                   pop.append(i)
            new_delta = []
            new_loc = []
            for i, (d, l) in enumerate(zip(delta, loc)):
                if i in pop:
                    pass
                else:
                    new_delta.append(d)
                    new_loc.append(l)
            
            binary_movement = new_trace.copy()
            events['duration'] = new_delta.copy()
            events['location'] = new_loc.copy()
            if not events['duration']:
                events['mean duration'] = 0
                events['max duration'] = 0
            else:
                events['mean duration'] = statistics.mean(events['duration'])/fs
                events['max duration'] = max(events['duration'])/fs
            bool_binary_movement = (binary_movement>0)
    
    total_samples = np.size(binary_movement)
    running_samples = np.count_nonzero(binary_movement == 1)
    percentage_running = running_samples/total_samples
    results = {'File': file_path,
               'Channel_A': np.copy(trace_A),
               'Channel_B': np.copy(trace_B),
               'Binary_A': np.copy(bi_A),
               'Binary_B': np.copy(bi_B),
               'binary_movement':np.array(binary_movement),
               'bool_binary_movement': bool_binary_movement,
               'extended_binary_movement':extended_binary_movement,
               'positions':positions, 
               'settings':settings,
               'speed':speed, 't':t,
               'percentage':percentage_running,
               'events': events}
    return results


def get_speed_position():
    file_path = easygui.fileopenbox()
    settings = set_settings_Igor()
    trace_A, trace_B = import_data_Igor(settings, file_path)
    
    bi_A = convert_binary(trace_A,settings['binary conversion threshold'])
    bi_B = convert_binary(trace_B, settings['binary conversion threshold'])
    
    positions = position(bi_A, bi_B, settings)
    speed, t = calc_speed(positions, settings, 9000)
    
    return positions, speed, t


def process_locomotion(file_path, settings, rec_points=9000):
    if file_path.endswith('.txt'):
        loc_settings = set_settings_Igor()
        data = import_data_Igor(loc_settings, file_path)
        if len(data) >= 3:
            trace_A = data[0]
            trace_B = data[1]
            trace_C = data[2]
        
        elif len(data) == 2:
            trace_A = data[0]
            trace_B = data[1]
            
    elif file_path.endswith('.abf'):
        loc_settings = set_settings_Digidata()
        trace_A, trace_B = import_data_Digidata(loc_settings, file_path)
    else:
        print('File not recognized')
        
    #directory = os.path.dirname(file_path)
    #file = file_path.split('\\')[-1]
    loc_settings['n_samples'] = rec_points
    loc_settings['speed threshold'] = settings['locomotion']['speed threshold']
    loc_settings['time_before'] = settings['locomotion']['time_before']
    loc_settings['time_after'] = settings['locomotion']['time_after']
    
    bi_A = convert_binary(trace_A, loc_settings['binary conversion threshold'])
    bi_B = convert_binary(trace_B, loc_settings['binary conversion threshold'])
    
    if 'trace_C' in locals():
        downsampled_C = resample(trace_C, rec_points)
        bi_C = convert_binary(downsampled_C, loc_settings['binary conversion threshold'])
   
    positions = position(bi_A, bi_B, loc_settings)
    speed, t = calc_speed(positions, loc_settings, rec_points)
    binary_movement, bool_binary_movement = move_nomove(speed, loc_settings)
    extended_binary_movement = extend_movenomove(binary_movement, loc_settings)
    n_changes = np.count_nonzero(binary_movement)
    if n_changes == len(binary_movement) or len(binary_movement)-n_changes == 0 or n_changes == 0:
        events = {}
        events['duration'] = []
        events['location'] = []
        events['mean duration'] = []
        events['max duration'] = []
      
    else:
        events = {}
        events['duration'], events['location'] = calc_event_duration(binary_movement)
        events['mean duration'] = statistics.mean(events['duration'])/settings['fs']
        events['max duration'] = max(events['duration'])/settings['fs']
    
        if settings['locomotion']['remove_short_events'] == True:
            new_trace = binary_movement.copy()
            delta_s = list(np.divide(events['duration'].copy(), settings['fs']))
            loc = events['location'].copy()
            delta = events['duration'].copy()
            pop = []
            for i, (d,l) in enumerate(zip(delta_s, loc)):
                if d < settings['locomotion']['min_event_duration']:
                   new_trace[l[0]:l[1]] = [0]*delta[i]
                   pop.append(i)
            new_delta = []
            new_loc = []
            for i, (d, l) in enumerate(zip(delta, loc)):
                if i in pop:
                    pass
                else:
                    new_delta.append(d)
                    new_loc.append(l)
            
            binary_movement = new_trace.copy()
            events['duration'] = new_delta.copy()
            events['location'] = new_loc.copy()
            if not events['duration']:
                events['mean duration'] = 0
                events['max duration'] = 0
            else:
                events['mean duration'] = statistics.mean(events['duration'])/settings['fs']
                events['max duration'] = max(events['duration'])/settings['fs']
            bool_binary_movement = (binary_movement>0)
    
    total_samples = np.size(binary_movement)
    running_samples = np.count_nonzero(binary_movement == 1)
    percentage_running = running_samples/total_samples
    
    if 'trace_C' in locals():
        results = {'File': file_path,
                   'Channel_A': np.copy(trace_A),
                   'Channel_B': np.copy(trace_B),
                   'Binary_A': np.copy(bi_A),
                   'Binary_B': np.copy(bi_B),
                   'binary_movement':np.array(binary_movement),
                   'bool_binary_movement': bool_binary_movement,
                   'extended_binary_movement':extended_binary_movement,
                   'positions':positions, 
                   'settings':loc_settings,
                   'speed':speed, 't':t,
                   'percentage':percentage_running,
                   'events': events,
                   'puff signal': list(bi_C)}

    else:
        results = {'File': file_path,
                   'Channel_A': np.copy(trace_A),
                   'Channel_B': np.copy(trace_B),
                   'Binary_A': np.copy(bi_A),
                   'Binary_B': np.copy(bi_B),
                   'binary_movement':np.array(binary_movement),
                   'bool_binary_movement': bool_binary_movement,
                   'extended_binary_movement':extended_binary_movement,
                   'positions':positions, 
                   'settings':loc_settings,
                   'speed':speed, 't':t,
                   'percentage':percentage_running,
                   'events': events}
        
    return results


def plot_total_average_response(dic):
    #file_locomotion = easygui.fileopenbox()
    #file_whisking = easygui.fileopenbox()
    
    dic = save_session.load_variable(file_locomotion)
    
    time_array = dic[list(dic.keys())[0]]['time']
    size = np.shape(dic[list(dic.keys())[0]]['traces'][0])[1]
    matrix = np.zeros((1, int(size)))
    for key in dic:
        for i in range(0,len(dic[key]['traces'])):
            dic[key]['traces'][i] = np.expand_dims(dic[key]['traces'][i], -1)
        combined_matrix = np.concatenate(dic[key]['traces'], axis=2)
        mean_combined = np.mean(combined_matrix, axis=2)
        matrix = np.concatenate((matrix, mean_combined), axis=0)
        
    matrix = np.delete(matrix, (0), axis=0)
    
    mean = np.mean(matrix, axis=0)
    plt.figure()
    plt.plot(time_array, mean)
    plt.show()

    # # Z-score 
    # z = np.abs(stats.zscore(matrix, axis=0)) 
    # data_clean = matrix[(z<3).all(axis=1)]
    # mean_clean = np.mean(data_clean, axis=0)
    # plt.figure()
    # plt.plot(time_array, mean_clean)
    # plt.show()

def plot_combined_events(combined_events, time_array, name, save_location):
    
    """
    combined_events: dictionary containing aligned events for each file
    name: File name
    save_location: location to save in
    
    """

    # Determine number of traces
    keys = list(combined_events.keys())
    n_traces = 0
    labels = []
    
    combined_events_table = np.zeros((1,np.shape(combined_events[keys[0]])[1]))
    for key in keys:
        n_traces += len(combined_events[key])
        labels.append([key]*len(combined_events[key]))
        combined_events_table = np.concatenate((combined_events_table, combined_events[key]), axis=0)
    
    combined_events_table = np.delete(combined_events_table, (0), axis=0)
    labels = list(itertools.chain(*labels))
    labels.insert(0, 'Time')
    labels.insert(1, 'Mean')
    labels.insert(2, 'Std')
    labels.insert(3, 'Sem')

    time = time_array                        
    mean = np.mean(combined_events_table, axis=0)
    std = np.std(combined_events_table, axis=0)
    sem = stats.sem(combined_events_table, axis=0)
    
    table_data = pd.DataFrame(np.concatenate((np.expand_dims(time, axis=0), np.expand_dims(mean, axis=0), np.expand_dims(std, axis=0), np.expand_dims(sem, axis=0), combined_events_table), axis=0), labels)
    table_data.to_excel('{}/Combined_aligned_events_{}.xlsx'.format(save_location, name))
    
    plt.figure()
    for i in range(len(combined_events_table)):
        plt.plot(time, combined_events_table[i], color='grey', alpha=0.5)
    plt.plot(time, mean, color='r', label='Mean')
    plt.plot(time, mean+std, '--', color='blue', alpha=0.8, label='std')
    plt.plot(time, mean-std, '--', color='blue', alpha=0.8)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('dF/F')
    plt.savefig('{}/Combined_aligned_{}_events'.format(save_location, name))
    plt.savefig('{}/Combined_aligned_{}_events.pdf'.format(save_location, name))
    plt.close()
     
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx 
    
def get_event_MI(settings, time_array, array_events):

    baseline = settings['baseline']
    response = settings['response']
    
    b1 = find_nearest(time_array, baseline[0])
    b2 = find_nearest(time_array, baseline[1])
    r1 = find_nearest(time_array, response[0])
    r2 = find_nearest(time_array, response[1])
    
    baseline_array = array_events[:, b1:b2]
    response_array = array_events[:, r1:r2]
    
    mean_baseline = np.mean(baseline_array, axis=1)
    mean_response = np.mean(response_array, axis=1)

    modulation_index = (mean_response-mean_baseline)/(mean_response+mean_baseline)
    
    return modulation_index

def test_data():
    file_path = easygui.fileopenbox()
    if file_path.endswith('.txt'):
        settings = set_settings_Igor()
        trace_A, trace_B = import_data_Igor(settings, file_path)
    elif file_path.endswith('.abf'):
        settings = set_settings_Digidata()
        trace_A, trace_B = import_data_Digidata(settings, file_path)
    else:
        print('File not recognized')
    
    A = convert_binary(trace_A,settings['binary conversion threshold'])
    B = convert_binary(trace_B, settings['binary conversion threshold'])
    
    positions = position(A, B, settings)
    
    plt.figure()
    plt.plot(A)
    plt.plot(B)
    plt.show()
    
    print(file_path)
    
    plt.figure()
    plt.plot(positions)
    plt.show() 