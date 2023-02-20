# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:23:02 2021

@author: m.debritovanvelze


"""
import numpy as np


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

def remove_short_events(binary_trace, sampling_rate, min_duration):
    """
    Removes events from 'trace' whose duration is shorter than 'min_duration'
    
    Inputs:
        binary_trace - list of 0s and 1s
        fs - sampling rate
        min_duration - minimum event duration is seconds
        
    Output:
        new_trace - similar to 'trace' but without events shorter than 'min_duration'
    """
    # New binary trace
    new_binary = np.copy(binary_trace)
    
    # Minimum number of frames in each bout
    n_frames = int(round(sampling_rate * min_duration))
                   
    # Calculate duration and location of bouts
    delta, loc = calc_event_duration(new_binary)
    
    # Remove bouts shorter than min_duration
    for i, (d, l) in enumerate(zip(delta, loc)):
        if d <= n_frames:
            new_binary[l[0]:l[1]] = 0
                    
    # pop = []
    # for i, (d,loc) in enumerate(zip(delta_s, loc)):
    #     if d < min_duration:
    #        new_binary[loc[0]:loc[1]] = [0]*delta[i]
    #        pop.append(i)
    # new_delta = []
    # new_loc = []
    # for i, (d, l) in enumerate(zip(delta, loc)):
    #     if i in pop:
    #         pass
    #     else:
    #         new_delta.append(d)
    #         new_loc.append(l)
            
    return new_binary

def separate_by_duration(binary_trace, sampling_rate, bins = [1, 5], plot=False):
    """
    Separate events by bins of speed
    
    Inputs:
        binary_trace - list of 0s and 1s
        fs - sampling rate
        bins - list of upper limit of each bin
        
    Output:
        new_trace - similar to 'trace' but without events shorter than 'min_duration'
        
    """
    # New binary trace
    selected_binary = {}
                   
    # Calculate duration and location of bouts
    delta, loc = calc_event_duration(binary_trace)
    
    # Define bin ranges
    bin_names = []
    bin_range = []
    
    for i, bin_value in enumerate(bins):
        if i == 0:
            bin_names.append('less than %s seconds'%bin_value)
            bin_range.append([0, int(round(sampling_rate * bin_value))])
        else:
            bin_names.append('%s second to %s seconds'%(bins[i-1], bins[i]))
            bin_range.append([int(round(sampling_rate * bins[i-1])), int(round(sampling_rate * bin_value))])
    bin_names.append('more than %s seconds'%bins[-1])
    bin_range.append([int(round(sampling_rate * bins[-1])), 'max'])
    
    # Separate by bins
    for i, (n, r) in enumerate(zip(bin_names, bin_range)):
        new_binary = np.copy(binary_trace)
        if r[1] == 'max':
            for i, (d, l) in enumerate(zip(delta, loc)):
                if d < r[0]:
                    new_binary[l[0]:l[1]] = 0
        else:
            for i, (d, l) in enumerate(zip(delta, loc)):
                if (d < r[0]) or (d >= r[1]):
                    new_binary[l[0]:l[1]] = 0
        if np.count_nonzero(new_binary) == 0:
            pass
        else:
            selected_binary[n] = {}
            selected_binary[n]['range'] = r
            selected_binary[n]['binary'] = new_binary
            _, selected_binary[n]['loc'] = calc_event_duration(new_binary)
    
    # Include unbinned trace
    selected_binary['all'] = {'range':['min', 'max'],
                              'binary': binary_trace,
                              'loc': loc}
    
    if plot:
        import matplotlib.pyplot as plt
        keys = list(selected_binary.keys())
        offset = 1.25
        t = np.linspace(0,300,len(selected_binary[keys[0]]['binary']))
        plt.figure()
        for i, key in enumerate(keys):
            trace = selected_binary[key]['binary'] + ((i+1)*offset)
            plt.plot(t, trace, label = key)
        plt.legend(loc=5, prop={'size': 6})
        plt.xlabel('Time (s)')
        plt.show()        
    
    return selected_binary

def remove_short_interevent_periods(binary_trace, sampling_rate, max_interevent):
    
    # New binary trace
    new_binary = np.copy(binary_trace)
    
    # Maximum number of frames accepted between bouts
    n_frames = int(round(sampling_rate * max_interevent))
    
    # Calculate duration and location of bouts
    delta, loc = calc_event_duration(new_binary)
    
    # Calculate frames between bouts
    list_intervals = list()
    for i in range(0,len(loc)-1):
        if loc[i+1][0]-loc[i][1] <= n_frames:
            list_intervals.append([loc[i][1],loc[i+1][0]])
    
    if len(list_intervals) > 0:
        for i in list_intervals:
            new_binary[i[0]:i[1]] = 1
    
    return new_binary


def get_inactive(binary_locomotion, binary_whisking):
    
    inactive = np.zeros(np.shape(binary_locomotion))
    for n, (l, w) in enumerate(zip(binary_locomotion, binary_whisking)):
        if l == 1 or w == 1:
            inactive[n] = 1
        else:
            inactive[n] = 0
    return inactive

# def exclusion_window(loc, inactive, delta, fs):
    
#     n_frames = int(round(fs * delta))
#     new_loc = []
#     for i in loc:
#         if i[0] == 0:
#             pass
#         if i[0] < n_frames:
#             if np.count_nonzero(inactive[0:i[0]]) > 0:
#                 pass
#             else:
#                 new_loc.append(i)
#         else:
#             if np.count_nonzero(inactive[(i[0]-n_frames):i[0]]) > 0:
#                 pass
#             else:
#                 new_loc.append(i)
#     return new_loc

def exclusion_window(loc, inactive, delta, fs):
    # delta is the interval of exclusion window
    # it is a list with 2 values [-5, 3]
    # first value is time before onset and second is after
    
    if delta[1] == 0:
        n_frames = int(round(fs * abs(delta[0])))
        new_loc = []
        for i in loc:
            if i[0] == 0:
                pass
            elif i[0] < n_frames:
                if np.count_nonzero(inactive[0:i[0]]) > 0:
                    pass
                else:
                    new_loc.append(i)
            else:
                if np.count_nonzero(inactive[(i[0]-n_frames):i[0]]) > 0:
                    pass
                else:
                    new_loc.append(i)
                    
    elif delta[0] == 0:
        n_frames = int(round(fs * delta[1]))
        new_loc = []
        for i in loc:
            if i[0]+n_frames > len(inactive):
                if np.count_nonzero(inactive[i[0]:]) > 0:
                    pass
                else:
                    new_loc.append(i)
            else:
                if np.count_nonzero(inactive[i[0]:(i[0]+n_frames)]) > 0:
                    pass
                else:
                    new_loc.append(i)
    
    else:
        frames_before = int(round(fs * abs(delta[0]))) 
        frames_after = int(round(fs * delta[1]))
        new_loc = []
        for i in loc:
            if i[0] == 0:
                pass
            elif i[0] < frames_before:
                if np.count_nonzero(inactive[0:i[0]]) > 0:
                    pass
                else:
                    if i[0]+frames_after > len(inactive):
                        if np.count_nonzero(inactive[i[0]:]) > 0:
                            pass
                        else:
                            new_loc.append(i)
                    else:
                        if np.count_nonzero(inactive[i[0]:(i[0]+frames_after)]) > 0:
                            pass
                        else:
                            new_loc.append(i)
                
            else:
                if np.count_nonzero(inactive[(i[0]-frames_before):i[0]]) > 0:
                    pass
                else:
                    if i[0]+frames_after > len(inactive):
                        if np.count_nonzero(inactive[i[0]:]) > 0:
                            pass
                        else:
                            new_loc.append(i)
                    else:
                        if np.count_nonzero(inactive[i[0]:(i[0]+frames_after)]) > 0:
                            pass
                        else:
                            new_loc.append(i)
            
    return new_loc    


def aligned_events(dF, loc, fs, time_before, time_after):

    # extract traces
    trace = []
    delta_before = int(time_before * fs)
    delta_after = int(time_after * fs)
    for event in enumerate(loc):
        location = event[1][0]
        if len(dF.shape) == 2:
            if location-delta_before > 0 and location+delta_after < dF.shape[1]:
                #extract dF 
                dF_section = dF[:,(location-delta_before):(location+delta_after)]
                trace.append(dF_section)
        elif len(dF.shape) == 1:
            if location-delta_before > 0 and location+delta_after < dF.shape[0]:
                #extract dF 
                dF_section = dF[(location-delta_before):(location+delta_after)]
                trace.append(dF_section)
        
    # calculate mean response per cell
    if len(trace) == 0:
        mean_combined = np.empty(shape=(1,1),dtype='object')
        time_array = np.linspace(-time_before, time_after, delta_after+delta_before)
        dic = {}
    else:
        if len(dF.shape) == 2:
            for i in range(0, len(trace)):
                trace[i] = np.expand_dims(trace[i], -1)
            combined_matrix = np.concatenate(trace, axis=2)
            mean_combined = np.mean(combined_matrix, axis=2)  
        elif len(dF.shape) == 1:
            for i in range(0, len(trace)):
                trace[i] = np.expand_dims(trace[i], 0)
            combined_matrix = np.concatenate(trace, axis=0)
            mean_combined = np.mean(combined_matrix, axis=0)  
            
        # mtx = []
        # mean = np.zeros((len(dF), delta_after+delta_before))
        # for i in range(0,len(dF)):
        #     matrix = np.zeros((len(trace), delta_after+delta_before))
        #     for t in enumerate(trace):
        #         matrix[t[0],:] = t[1][i, :]
        #     mean[i, :] = np.mean(matrix, axis=0)
        #     mtx.append(matrix)
        
        time_array = np.linspace(-time_before, time_after, delta_after+delta_before)
        
        # Create dictionary
        dic = {'all_traces': combined_matrix,
               'response per cell': mean_combined,
               'time': time_array}
    
    return mean_combined, time_array, dic
    
            
