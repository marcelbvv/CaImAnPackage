# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:11:32 2022

@author: m.debritovanvelze
"""


import easygui
import time as t
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import itertools

import save_session, binary_calculations, compute_stats

#import tqdm

"""
TO DO:
    - Correct list files
    - Get events 
    - Plot

"""

def get_info(trace):
    dic = {}
    dic['trace'] = trace
    dic['delta'], dic['loc'] = binary_calculations.calc_event_duration(trace)
    return

def combined_events(combined_events, save_path):
    
    combined = {}
    combined_dic = {}
    
    keys = list(combined_events.keys())
    n_traces = 0
    labels = []
    
    combined_events_table = np.zeros((1,np.shape(combined_events[keys[0]]['events'])[1]))
    combined_stim_table = np.zeros((1,len(combined_events[keys[0]]['stim'])))
    combined_speed_table = np.zeros((1,len(combined_events[keys[0]]['locomotion events'])))
    time_array = combined_events[keys[0]]['time']
    for key in keys:
        n_traces += len(combined_events[key]['events'])
        labels.append([key]*len(combined_events[key]['events']))
        combined_events_table = np.concatenate((combined_events_table, combined_events[key]['events']), axis=0)
        combined_stim_table = np.concatenate((combined_stim_table, np.expand_dims(combined_events[key]['stim'], 0)), axis=0)
        combined_speed_table = np.concatenate((combined_speed_table, np.expand_dims(combined_events[key]['locomotion events'], 0)), axis=0)
    combined_events_table = np.delete(combined_events_table, (0), axis=0)
    combined_stim_table = np.delete(combined_stim_table, (0), axis=0)
    combined_speed_table = np.delete(combined_speed_table, (0), axis=0)
    labels = list(itertools.chain(*labels))
    labels.insert(0, 'Time')
    labels.insert(1, 'Mean Event')
    labels.insert(2, 'Sem Event')
    labels.insert(3, 'Mean Stim')
    labels.insert(4, 'Sem Stim')
    labels.insert(5, 'Mean Speed')
    labels.insert(6, 'Sem Speed')
    
    mean_event = np.mean(combined_events_table, axis=0) 
    sem_event = np.std(combined_events_table, axis=0) / np.sqrt(len(combined_events_table))
    n_event = len(combined_events_table)
    mean_stim = np.mean(combined_stim_table, axis=0) 
    sem_stim = np.std(combined_stim_table, axis=0) / np.sqrt(len(combined_stim_table))
    n_stim = len(combined_stim_table)
    mean_speed = np.mean(combined_speed_table, axis=0) 
    sem_speed = np.std(combined_speed_table, axis=0) / np.sqrt(len(combined_speed_table))
    n_speed = len(combined_speed_table)
    
    table_data = pd.DataFrame(np.concatenate((np.expand_dims(time_array, axis=0), 
                                              np.expand_dims(mean_event, axis=0), 
                                              np.expand_dims(sem_event, axis=0), 
                                              np.expand_dims(mean_stim, axis=0), 
                                              np.expand_dims(sem_stim, axis=0),
                                              np.expand_dims(mean_speed, axis=0), 
                                              np.expand_dims(sem_speed, axis=0),
                                              combined_events_table), axis=0), labels)
    combined = table_data
    combined_dic = {'time_array': time_array,
                    'mean_event': mean_event,
                    'sem_event': sem_event, 
                    'N_event': n_event,
                    'mean_stim': mean_stim,
                    'sem_stim': sem_stim, 
                    'N_stim': n_stim,
                    'mean_speed': mean_speed,
                    'sem_speed': sem_speed, 
                    'N_speed': n_speed,
                    'events': combined_events_table,
                    'labels': labels}
    
    table_data.to_excel('{}/Combined_aligned_events_{}.xlsx'.format(save_path, 'WhiskerStimulation'))
    save_session.save_variable('{}/Data_aligned_events_WhiskerStimulation'.format(save_path), combined_dic)
    
    return combined, combined_dic

def heatmap_events(dic, save_location):
    
    f, axs = plt.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [0.2, 0.2, 0.2, 2]})
    
    
    
    decimated = signal.decimate(dic['events'], 2, axis = 1)
    # Normalize data
    normalized = np.copy(decimated)
    for i in range(len(decimated)):
        normalized[i] = (decimated[i] - decimated[i].min()) / (decimated[i].max() - decimated[i].min())
    normalized, _ = compute_stats.sort_data(normalized, method='StandardScaler',
                                            sorted_first_pc=True, 
                                            binned_01=False)
    
    #axs[0].set_title('Whisker Stimulation')
    axs[0].plot(dic['time_array'], dic['mean_stim'], color='red', label='Stimulation', linewidth=0.5)
    axs[0].legend(loc='upper right', frameon=False)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["left"].set_visible(False)
    axs[0].spines["bottom"].set_visible(False)
    
    #axs[1].set_title('Mean Calcium Response')
    axs[1].plot(dic['time_array'], dic['mean_event'], color='black', label='Response', linewidth=0.5)
    axs[1].legend(loc='upper right', frameon=False)
    axs[1].fill_between(dic['time_array'],  dic['mean_event']-dic['sem_event'], dic['mean_event']+dic['sem_event'], alpha=.5, color= 'grey')
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["left"].set_visible(False)
    axs[1].spines["bottom"].set_visible(False)
    
    #axs[2].set_title('Mean Locomotion Response')
    axs[2].plot(dic['time_array'], dic['mean_speed'], color='black', label='Locomotion', linewidth=0.5)
    axs[2].legend(loc='upper right', frameon=False)
    axs[2].fill_between(dic['time_array'],  dic['mean_speed']-dic['sem_speed'], dic['mean_speed']+dic['sem_speed'], alpha=.5, color= 'grey')
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)
    axs[2].spines["left"].set_visible(False)
    axs[2].spines["bottom"].set_visible(False)
    
    im = axs[3].imshow(normalized, cmap='inferno',interpolation="spline16", aspect='auto', extent = [dic['time_array'][0], dic['time_array'][-1], 0-0.5, np.size(normalized, 0)-0.5])
    cbaxes = f.add_axes([0.91, 0.11, 0.03, 0.35]) #Add position (left, bottom, width, height)
    f.colorbar(im, cax=cbaxes)
    axs[3].spines["top"].set_visible(False)
    axs[3].spines["right"].set_visible(False)
    axs[3].spines["left"].set_visible(False)
    axs[3].spines["bottom"].set_visible(False)
    axs[3].set_ylabel('Cell Number')
    axs[3].set_xlabel('Time (s)')

    plt.savefig(os.path.join(save_location, 'Event_analysis_heatmap Whisker Stimulation'+'.pdf'))
    plt.close()

    return

    
    
def run_analysis(list_files=None, save_path=None):
    
    # Import file list
    if (list_files is None):
        list_files = save_session.load_variable(easygui.fileopenbox())
    # Create combined dictionary
    combined_puff = {}
    
    t0 = t.time()
    # Loop through files
    for file in list_files:
        # Import analysis results
        data = save_session.load_variable(file+'.pickle')
        dF = data['dF']
        fs = data['Settings']['fs']
        name_file = data['table_data']['File'][0]
        negative_F0 = data['negative_F0_cells']
        stim_signal = data['Locomotion_data']['puff signal']
        if 'pearson_shuffle' in data:
            locomotion_binary = data['Locomotion_data']['binary_movement']
            speed = data['Locomotion_data']['speed']
            pearson = data['pearson_shuffle']['pearson']
        
        # Get stimulaton location
        delta, loc = binary_calculations.calc_event_duration(stim_signal)        
        dF_events, time, _ = binary_calculations.aligned_events(dF, loc, fs, time_before= 10, time_after= 10)
        stim_events, _, _ = binary_calculations.aligned_events(np.array(stim_signal), loc, fs, time_before= 10, time_after= 10)
        locomotion_events, _, _ = binary_calculations.aligned_events(speed, loc, fs, time_before= 10, time_after= 10)
        
        # Save results
        if dF_events.size == 1:
            pass
        else:
            if len(dF_events) == 1:
                if negative_F0[0] == False:
                    combined_puff[name_file] = {'time': time,
                                                'events': dF_events,
                                                'stim': stim_events,
                                                'pearson': pearson,
                                                'locomotion events': locomotion_events}
            else:
                selected_events = np.delete(dF_events, [i for i, x in enumerate(list(negative_F0)) if x[0]==True], 0)
                selected_pearson = np.delete(pearson, [i for i, x in enumerate(list(negative_F0)) if x[0]==True], 0)
                selected_locomotion = np.delete(locomotion_events, [i for i, x in enumerate(list(negative_F0)) if x[0]==True], 0)
                combined_puff[name_file] = {'time': time,
                                            'events': selected_events,
                                            'stim': stim_events,
                                            'pearson': selected_pearson,
                                            'locomotion events': selected_locomotion}
    
    # Save results
    
    if (save_path is not None):
        save_location = os.path.join(save_path, 'Event-based Analysis')
        if not os.path.exists(os.path.join(save_path, 'Event-based Analysis')):
            os.mkdir(save_location)
    else:
        save_location = os.path.join(easygui.diropenbox(), 'Event-based Analysis')
        os.mkdir(save_location)    
        
    if combined_puff[list(combined_puff.keys())[0]]:
        data_stim, dic_stim = combined_events(combined_puff, save_location)
        heatmap_events(dic_stim, save_location)  
        
    print('time %4.2f sec'%(t.time()-(t0)))
    
    
    
    return
    
    
    # # Plot combined results
    # decimated = signal.decimate(whisker_stim_events, 2, axis = 1)
    # # Normalize data
    # normalized = np.copy(decimated)
    # for i in range(len(decimated)):
    #     normalized[i] = (decimated[i] - decimated[i].min()) / (decimated[i].max() - decimated[i].min())
    # normalized, _ = compute_stats.sort_data(normalized, method='StandardScaler',
    #                                     sorted_first_pc=True, 
    #                                     binned_01=False)        
    
    # im = plt.imshow(normalized, cmap='inferno', aspect='auto', extent = [time[0], time[-1], 0-0.5, np.size(normalized, 0)-0.5])
    # plt.show()
