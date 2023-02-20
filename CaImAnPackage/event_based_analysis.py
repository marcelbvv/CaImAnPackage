# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 14:14:13 2021

@author: m.debritovanvelze
"""
import save_session, binary_calculations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import easygui
import time as t
import itertools
from matplotlib.gridspec import SubplotSpec
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
import compute_stats

def get_event_location(binary, fs, speed = None, exclusion_window= False, exclusion_time= [-0.5, 0], inactive = [], remove_short= False, min_duration= 0.5, join_bouts= False, max_duration= 1, separate_bins=False, bins=[2, 4, 6]):
    
    dic = {'exclusion_window': exclusion_window, 
           'exclusion_time': exclusion_time,
           'remove_short': remove_short,
           'min_duration': min_duration,
           'join_bouts': join_bouts,
           'max_duration': max_duration, 
           'separate_bins': separate_bins,
           'bins': bins, 
           'bin_location': {}}
    
    # Remove short events
    if remove_short:
        new_binary = binary_calculations.remove_short_events(binary, fs, min_duration)
    else:
        new_binary = np.copy(binary)
    
    # Join close events
    if join_bouts:
        new_binary = binary_calculations.remove_short_interevent_periods(new_binary, fs, max_duration)
    
    # Separate events by speed bins
    if separate_bins:
        binned_binary = binary_calculations.separate_by_duration(new_binary, fs, bins = bins)
    else:
        binned_binary = {'all': {'range':['min', 'max'],
                                 'binary': new_binary}}
        _, binned_binary['all']['loc'] =  binary_calculations.calc_event_duration(new_binary)
        
    # Select only events with resting periods before
    if exclusion_window:
        for key in list(binned_binary.keys()):
            dic['bin_location'][key] = binary_calculations.exclusion_window(binned_binary[key]['loc'], inactive, exclusion_time, fs) 
    else:
        for key in list(binned_binary.keys()):
            dic['bin_location'][key] = binned_binary[key]['loc']
            
    return dic


def calc_mean_event(dF, loc, fs, time_before= 8, time_after= 5):
    _, _, dic = binary_calculations.aligned_events(dF, loc, time_before, time_after, fs)
    events = dic['response per cell']
    mean_event = np.mean(dic['response per cell'], axis=0) 
    sem_event = np.std(dic['response per cell'], axis=0) / np.sqrt(len(dic['response per cell']))
    time = dic['time']
    return mean_event, sem_event, events, time
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx 

def get_event_MI(time_array, array_events):

    baseline = [-8, -1]
    response = [0, 4]
    
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

def combined_events(combined_events, type_data, save_path):
    
    combined = {}
    combined_dic = {}
    
    for bin_name in list(combined_events.keys()):
        for data_type in list(combined_events[bin_name].keys()):
            if combined_events[bin_name][data_type]:
                keys = list(combined_events[bin_name][data_type].keys())
                n_traces = 0
                labels = []
                
                combined_events_table = np.zeros((1,np.shape(combined_events[bin_name][data_type][keys[0]]['events'])[1]))
                combined_speed_table = np.zeros((1,len(combined_events[bin_name][data_type][keys[0]]['speed'])))
                combined_whisking_table = np.zeros((1,len(combined_events[bin_name][data_type][keys[0]]['speed'])))
                time_array = combined_events[bin_name][data_type][keys[0]]['time']
                for key in keys:
                    n_traces += len(combined_events[bin_name][data_type][key]['events'])
                    labels.append([key]*len(combined_events[bin_name][data_type][key]['events']))
                    combined_events_table = np.concatenate((combined_events_table, combined_events[bin_name][data_type][key]['events']), axis=0)
                    combined_speed_table = np.concatenate((combined_speed_table, np.expand_dims(combined_events[bin_name][data_type][key]['speed'], 0)), axis=0)
                    if 'whisking' in combined_events[bin_name][data_type][key]:
                        combined_whisking_table = np.concatenate((combined_whisking_table, np.expand_dims(combined_events[bin_name][data_type][key]['whisking'], 0)), axis=0)
                combined_events_table = np.delete(combined_events_table, (0), axis=0)
                combined_speed_table = np.delete(combined_speed_table, (0), axis=0)
                combined_whisking_table = np.delete(combined_whisking_table, (0), axis=0)
                labels = list(itertools.chain(*labels))
                labels.insert(0, 'Time')
                labels.insert(1, 'Mean Event')
                labels.insert(2, 'Sem Event')
                labels.insert(3, 'Mean Speed')
                labels.insert(4, 'Sem Speed')
                if combined_whisking_table.any():
                    labels.insert(5, 'Mean Whisking')
                    labels.insert(6, 'Sem Whisking')
                
                mean_event = np.mean(combined_events_table, axis=0) 
                sem_event = np.std(combined_events_table, axis=0) / np.sqrt(len(combined_events_table))
                n_event = len(combined_events_table)
                mean_speed = np.mean(combined_speed_table, axis=0) 
                sem_speed = np.std(combined_speed_table, axis=0) / np.sqrt(len(combined_speed_table))
                n_speed = len(combined_speed_table)
                if combined_whisking_table.any():
                    mean_whisking = np.mean(combined_whisking_table, axis=0) 
                    sem_whisking = np.std(combined_whisking_table, axis=0) / np.sqrt(len(combined_whisking_table))
                    n_whisking = len(combined_whisking_table)
            
                    table_data = pd.DataFrame(np.concatenate((np.expand_dims(time_array, axis=0), 
                                                              np.expand_dims(mean_event, axis=0), 
                                                              np.expand_dims(sem_event, axis=0), 
                                                              np.expand_dims(mean_speed, axis=0), 
                                                              np.expand_dims(sem_speed, axis=0),
                                                              np.expand_dims(mean_whisking, axis=0), 
                                                              np.expand_dims(sem_whisking, axis=0),
                                                              combined_events_table), axis=0), labels)
                else:
                    table_data = pd.DataFrame(np.concatenate((np.expand_dims(time_array, axis=0), 
                                                              np.expand_dims(mean_event, axis=0), 
                                                              np.expand_dims(sem_event, axis=0), 
                                                              np.expand_dims(mean_speed, axis=0), 
                                                              np.expand_dims(sem_speed, axis=0),
                                                              combined_events_table), axis=0), labels)
                if bin_name in combined:
                    pass
                else:
                    combined[bin_name] = {}
                combined[bin_name][data_type] = table_data
                if bin_name in combined_dic:
                    pass
                else:
                    combined_dic[bin_name] = {}
                combined_dic[bin_name][data_type] = {'time_array': time_array,
                                           'mean_event': mean_event,
                                           'sem_event': sem_event, 
                                           'N_event': n_event,
                                           'mean_speed': mean_speed,
                                           'sem_speed': sem_speed, 
                                           'N_speed': n_speed,
                                           'events': combined_events_table,
                                           'labels': labels}
                if combined_whisking_table.any():
                    combined_dic[bin_name][data_type]['mean_whisking'] = mean_whisking
                    combined_dic[bin_name][data_type]['sem_whisking'] = sem_whisking
                    combined_dic[bin_name][data_type]['N_whisking'] = n_whisking
                
                table_data.to_excel('{}/{}-Combined_aligned_events_{}_{}.xlsx'.format(save_path, bin_name, type_data, data_type))
    
    return combined, combined_dic

def plot_scalebar(axis, xlength=None, x_unit='s', ylength=None, y_unit='\u0394F/F0'):
    
    xrange = axis.get_xlim()
    yrange = axis.get_ylim()
    
    perc_x = 0.1
    perc_y = 0.2
    
    # Origin
    offset_origin = [0.01, 0.2]
    origin = [xrange[0]+ abs(offset_origin[0]*xrange[1]), ((1-offset_origin[1])*yrange[1])]
    #origin = [0, yrange[1]] # origin in top left corner
    
    if xlength:
        xmin = origin[0]
        xmax = xmin+xlength
    else:
        xmin = origin[0]
        xmax = xmin + perc_x*xrange[1]
        xlength = xmax - xmin
    
    if ylength:
        ymin = origin[1]-ylength
        ymax = origin[1]
    else:
        ymin = origin[1]-perc_y*(yrange[1]-yrange[0])
        ymax = origin[1]
        ylength = ymax - ymin
    
    # x axis  
    axis.hlines(y=origin[1], xmin=origin[0], xmax=xmax)
    #axis.text(xmin+(xlength/3), ymax+(ylength/3), xlength)
    axis.text(xmin+(xlength/6), ymax, str(xlength)+' '+x_unit)
    # y axis
    axis.vlines(x=origin[0], ymin=ymin, ymax=ymax)
    #axis.text(xmin+(xmax/6), ymax-(ylength/1.5), ylength)
    axis.text(xmin, ymax-(ylength/1.5), str(ylength)+' '+y_unit)

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold')
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

def plot_figure(dic, data_type, save_location):
    
    for bin_name in list(dic.keys()):
        keys = list(dic[bin_name].keys())
        n_figures = len(keys)
        
        f, axs = plt.subplots(n_figures, 2, sharex=True, figsize=(n_figures*4,10))
        grid = plt.GridSpec(n_figures, 2)
        f.suptitle(data_type, fontsize=16)
        
        for n, key in enumerate(keys):
            table = dic[bin_name][key]
            create_subtitle(f, grid[n, ::], key)
            if 'Mean Whisking' in table.index.values:
                if 'Mean Speed' in table.index.values:
                    axs[n, 0].set_title('Speed + Whisking')
                    axs[n, 0].plot(table.loc['Time'], table.loc['Mean Speed'], color='red', label='Locomotion')
                    axs[n, 0].fill_between(table.loc['Time'],  table.loc['Mean Speed']-table.loc['Sem Speed'], table.loc['Mean Speed']+table.loc['Sem Speed'], alpha=.5, color= 'grey')
                    axs[n, 0].plot(table.loc['Time'], table.loc['Mean Whisking'], color='blue', label='Whisking')
                    axs[n, 0].fill_between(table.loc['Time'],  table.loc['Mean Whisking']-table.loc['Sem Whisking'], table.loc['Mean Whisking']+table.loc['Sem Whisking'], alpha=.5, color= 'grey')
                    axs[n, 0].set_xlabel('Time from onset (s)')
                else:
                    axs[n, 0].set_title('Whisking')
                    axs[n, 0].plot(table.loc['Time'], table.loc['Mean Whisking'], color='blue', label='Whisking')
                    axs[n, 0].fill_between(table.loc['Time'],  table.loc['Mean Whisking']-table.loc['Sem Whisking'], table.loc['Mean Whisking']+table.loc['Sem Whisking'], alpha=.5, color= 'grey')
                    axs[n, 0].set_xlabel('Time from onset (s)')
            else:
                axs[n, 0].set_title('Speed')
                axs[n, 0].plot(table.loc['Time'], table.loc['Mean Speed'], color='red', label='Locomotion')
                axs[n, 0].fill_between(table.loc['Time'],  table.loc['Mean Speed']-table.loc['Sem Speed'], table.loc['Mean Speed']+table.loc['Sem Speed'], alpha=.5, color= 'grey')
                axs[n, 0].set_xlabel('Time from onset (s)')
            
            axs[n, 0].legend(loc='upper left')
                
            axs[n, 1].set_title('Aligned Events')
            axs[n, 1].plot(table.loc['Time'], table.loc['Mean Event'], color='red')
            axs[n, 1].fill_between(table.loc['Time'],  table.loc['Mean Event']-table.loc['Sem Event'], table.loc['Mean Event']+table.loc['Sem Event'], alpha=.5, color= 'grey')
            axs[n, 1].set_xlabel('Time from onset (s)')
            
        f.tight_layout()   
        plt.savefig(os.path.join(save_location, bin_name + '-Event_analysis '+data_type+'.pdf'))
        plt.close()
    
def make_subplot(ax, dic, data_type, variable, error):
    ax.axvline(x=0, linestyle='--', color='gray')
    ax.axhline(y=0, linestyle='--', color='gray')
    ax.plot(dic[data_type]['time_array'], dic[data_type][variable], 'black')
    ax.fill_between(dic[data_type]['time_array'], dic[data_type][variable]-dic[data_type][error], dic[data_type][variable]+dic[data_type][error], alpha=.2, color= 'red')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    #ax.spines["left"].set_visible(False)
    return ax
    
def figure_traces(dic, title= None, save_path=None):
    
    """
    To improve:
        - Units of face motion are wrong. Change units of Face motion to %. Multiply by 100
        
    """
    
    if dic[0]:
        data_loc = dic[0]
    else:
        data_loc = None
    if dic[2]:
        data_wo = dic[2]
    else:
        data_wo = None
    
    data_type = 'Total'
    
    if data_wo:
        if data_loc:
            fig = plt.figure(figsize=(8, 5))
            if title:
                fig.suptitle(title, fontsize=16)
            fig.subplots_adjust(hspace=0.2, wspace=0.3)
            gs = GridSpec(3, 2, figure=fig)
            ax1 = fig.add_subplot(gs[0, 0])
            if 'N_event' in data_loc[data_type]:
                ax1.text(0.1, 0.9, 'N = '+str(data_loc[data_type]['N_event']), size=10, color='black', transform=ax1.transAxes)
            ax1 = make_subplot(ax1, data_loc, data_type, 'mean_event', 'sem_event')
            ax1.set_title('Locomotion Onset', fontsize = 10)
            ax1.set_ylabel('dF/F0 (%)')
            ax1.tick_params('x', labelbottom=False)
            
            if 'mean_speed' in data_loc[data_type]:
                ax2 = fig.add_subplot(gs[1, 0], sharex = ax1)
                ax2 = make_subplot(ax2, data_loc, data_type, 'mean_speed', 'sem_speed')
                if 'N_speed' in data_loc[data_type]:
                    ax2.text(0.1, 0.9, 'N = '+str(data_loc[data_type]['N_speed']), size=10, color='black', transform=ax2.transAxes)
                ax2.set_ylabel('Speed (cm/s)')
                ax2.tick_params('x', labelbottom=False)
            
            if 'mean_whisking' in data_loc[data_type]:
                ax3 = fig.add_subplot(gs[2, 0], sharex = ax1)
                ax3 = make_subplot(ax3, data_loc, data_type, 'mean_whisking', 'sem_whisking')
                if 'N_whisking' in data_loc[data_type]:
                    ax3.text(0.1, 0.9, 'N = '+str(data_loc[data_type]['N_whisking']), size=10, color='black', transform=ax3.transAxes)
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Face motion (AU)')
                
            ax4 = fig.add_subplot(gs[0, 1])
            ax4 = make_subplot(ax4, data_wo, data_type, 'mean_event', 'sem_event')
            if 'N_event' in data_wo[data_type]:
                ax4.text(0.1, 0.9, 'N = '+str(data_wo[data_type]['N_event']), size=10, color='black', transform=ax4.transAxes)
            ax4.set_title('Whisking onset', fontsize = 10)
            ax4.set_ylabel('dF/F0 (%)')
            ax4.tick_params('x', labelbottom=False)
           
            if 'mean_speed' in data_wo[data_type]:
                ax5 = fig.add_subplot(gs[1, 1], sharex = ax4)
                ax5 = make_subplot(ax5, data_wo, data_type, 'mean_speed', 'sem_speed')
                if 'N_speed' in data_wo[data_type]:
                    ax5.text(0.1, 0.9, 'N = '+str(data_wo[data_type]['N_speed']), size=10, color='black', transform=ax5.transAxes)
                ax5.set_ylabel('Speed (cm/s)')
                ax5.tick_params('x', labelbottom=False)
            
            if 'mean_whisking' in data_wo[data_type]:
                ax6 = fig.add_subplot(gs[2, 1], sharex = ax1)
                ax6 = make_subplot(ax6, data_wo, data_type, 'mean_whisking', 'sem_whisking')
                if 'N_whisking' in data_wo[data_type]:
                    ax6.text(0.1, 0.9, 'N = '+str(data_wo[data_type]['N_whisking']), size=10, color='black', transform=ax6.transAxes)
                ax6.set_xlabel('Time (s)')
                ax6.set_ylabel('Face motion (AU)')
    
        else:
            fig = plt.figure(figsize=(5, 5))
            if title:
                fig.suptitle(title, fontsize=16)
            fig.subplots_adjust(hspace=2)
            gs = GridSpec(3, 1, figure=fig)
            ax4 = fig.add_subplot(gs[0, 0])
            ax4 = make_subplot(ax4, data_wo, data_type, 'mean_event', 'sem_event')
            if 'N_event' in data_wo[data_type]:
                ax4.text(0.1, 0.9, 'N = '+str(data_wo[data_type]['N_event']), size=10, color='black', transform=ax4.transAxes)
            ax4.set_title('Whisking onset', fontsize = 10)
            ax4.set_ylabel('dF/F0 (%)')
            ax4.tick_params('x', labelbottom=False)
            
            if 'mean_speed' in data_wo[data_type]:
                ax5 = fig.add_subplot(gs[1, 0], sharex = ax1)
                ax5 = fig.add_subplot(gs[1, 1], sharex = ax4)
                ax5 = make_subplot(ax5, data_wo, data_type, 'mean_speed', 'sem_speed')
                if 'N_speed' in data_wo[data_type]:
                    ax5.text(0.1, 0.9, 'N = '+str(data_wo[data_type]['N_speed']), size=10, color='black', transform=ax5.transAxes)
                ax5.set_ylabel('Speed (cm/s)')
                ax5.tick_params('x', labelbottom=False)
            
            if 'mean_whisking' in data_wo[data_type]:
                ax6 = fig.add_subplot(gs[2, 1], sharex = ax1)
                ax6 = make_subplot(ax6, data_wo, data_type, 'mean_whisking', 'sem_whisking')
                if 'N_whisking' in data_wo[data_type]:
                    ax6.text(0.1, 0.9, 'N = '+str(data_wo[data_type]['N_whisking']), size=10, color='black', transform=ax6.transAxes)
                ax6.set_xlabel('Time (s)')
                ax6.set_ylabel('Face motion (AU)')
                
    elif data_loc:
        fig = plt.figure(figsize=(5, 5))
        if title:
            fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(hspace=0.3)
        gs = GridSpec(3, 1, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1 = make_subplot(ax1, data_loc, data_type, 'mean_event', 'sem_event')
        if 'N_event' in data_loc[data_type]:
            ax1.text(0.1, 0.9, 'N = '+str(data_loc[data_type]['N_event']), size=10, color='black', transform=ax1.transAxes)
        ax1.set_title('Locomotion Onset', fontsize = 10)
        ax1.set_ylabel('dF/F0 (%)')
        
        if 'mean_speed' in data_loc[data_type]:
            ax2 = fig.add_subplot(gs[1, 0], sharex = ax1)
            ax2 = make_subplot(ax2, data_loc, data_type, 'mean_speed', 'sem_speed')
            if 'N_speed' in data_loc[data_type]:
                ax2.text(0.1, 0.9, 'N = '+str(data_loc[data_type]['N_speed']), size=10, color='black', transform=ax2.transAxes)
            ax2.set_ylabel('Speed (cm/s)')
        
        if 'mean_whisking' in data_loc[data_type]:
            ax3 = fig.add_subplot(gs[2, 0], sharex = ax1)
            ax3 = make_subplot(ax3, data_loc, data_type, 'mean_whisking', 'sem_whisking')
            if 'N_whisking' in data_loc[data_type]:
                ax3.text(0.1, 0.9, 'N = '+str(data_loc[data_type]['N_whisking']), size=10, color='black', transform=ax3.transAxes)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Face motion (AU)')
    else:
        pass
    
    plt.show()
        

    if (save_path is not None):
        plt.savefig(os.path.join(save_path, 'Event_analysis_'+data_type+'.pdf'))
    else:
        save_location = os.path.join(easygui.fileopenbox(), 'Event-based Analysis')
        plt.savefig(os.path.join(save_location, 'Event_analysis_'+data_type+'.pdf'))
    
    return

    
def heatmap_events(dic, data_type, save_location):
    
    for bin_name in list(dic.keys()):
        keys = list(dic[bin_name].keys())
        n_figures = 2*len(keys)
        
        f, axs = plt.subplots(n_figures, 1, figsize=(4,2*n_figures), gridspec_kw={'height_ratios': len(keys)*[0.5, 1.5]})
        grid = plt.GridSpec(n_figures, 1)
        f.suptitle(data_type, fontsize=16)
        
        for n, key in enumerate(keys):
            data = dic[bin_name][key]
            decimated = signal.decimate(data['events'], 2, axis = 1)
            # Normalize data
            normalized = np.copy(decimated)
            for i in range(len(decimated)):
                normalized[i] = (decimated[i] - decimated[i].min()) / (decimated[i].max() - decimated[i].min())
            normalized, _ = compute_stats.sort_data(normalized, method='StandardScaler',
                                                sorted_first_pc=True, 
                                                binned_01=False)
            
            if 'mean_whisking' in data:
                if 'mean_speed' in data:
                    axs[2*n].set_title('%s - Speed + Whisking'% key, fontsize=10)
                    axs[2*n].plot(data['time_array'], data['mean_speed'], color='red', label='Locomotion', linewidth=0.5)
                    axs[2*n].fill_between(data['time_array'],  data['mean_speed']-data['sem_speed'], data['mean_speed']+data['sem_speed'], alpha=.5, color= 'grey')
                    axs[2*n].plot(data['time_array'], data['mean_whisking'], color='blue', label='Whisking', linewidth=0.5)
                    axs[2*n].fill_between(data['time_array'],  data['mean_whisking']-data['sem_whisking'], data['mean_whisking']+data['sem_whisking'], alpha=.5, color= 'grey')
                    axs[2*n].plot(data['time_array'], data['mean_event'], color='black', label='Response', linewidth=0.5)
                    axs[2*n].fill_between(data['time_array'],  data['mean_event']-data['sem_event'], data['mean_event']+data['sem_event'], alpha=.5, color= 'grey')
                    axs[2*n].set_xlabel('Time from onset (s)')
                else:
                    axs[2*n].set_title('%s - Whisking'% key)
                    axs[2*n].plot(data['time_array'], data['mean_whisking'], color='blue', label='Whisking', linewidth=0.5)
                    axs[2*n].fill_between(data['time_array'],  data['mean_whisking']-data['sem_whisking'], data['mean_whisking']+data['sem_whisking'], alpha=.5, color= 'grey')
                    axs[2*n].plot(data['time_array'], data['mean_event'], color='black', label='Response', linewidth=0.5)
                    axs[2*n].fill_between(data['time_array'],  data['mean_event']-data['sem_event'], data['mean_event']+data['sem_event'], alpha=.5, color= 'grey')
                    axs[2*n].set_xlabel('Time from onset (s)')
            else:
                axs[2*n].set_title('%s - Speed'% key)
                axs[2*n].plot(data['time_array'], data['mean_speed'], color='red', label='Locomotion', linewidth=0.5)
                axs[2*n].fill_between(data['time_array'],  data['mean_speed']-data['sem_speed'], data['mean_speed']+data['sem_speed'], alpha=.5, color= 'grey')
                axs[2*n].plot(data['time_array'], data['mean_event'], color='black', label='Response', linewidth=0.5)
                axs[2*n].fill_between(data['time_array'],  data['mean_event']-data['sem_event'], data['mean_event']+data['sem_event'], alpha=.5, color= 'grey')
                axs[2*n].set_xlabel('Time from onset (s)')
            axs[2*n].legend(loc='best', fontsize=6)
            #plot_scalebar(axs[2*n], xlength=5, x_unit='s', ylength=0.5, y_unit='\u0394F/F0 cm/s')
            axs[2*n].vlines(x=0, ymin=axs[2*n].get_ylim()[0], ymax=axs[2*n].get_ylim()[1], linestyle='dotted', color='k', linewidth=0.5)
            axs[2*n].spines['top'].set_visible(False)
            axs[2*n].spines['right'].set_visible(False)
            axs[2*n].spines['bottom'].set_visible(False)
            axs[2*n].spines['left'].set_visible(False)
            
            
            im = axs[2*n+1].imshow(normalized, cmap='inferno',interpolation="spline16", aspect='auto', extent = [data['time_array'][0], data['time_array'][-1], 0-0.5, np.size(normalized, 0)-0.5])
            f.colorbar(im, ax=axs[2*n+1])
            axs[2*n+1].spines["top"].set_visible(False)
            axs[2*n+1].spines["right"].set_visible(False)
            axs[2*n+1].spines["left"].set_visible(False)
            axs[2*n+1].spines["bottom"].set_visible(False)
            axs[2*n+1].set_ylabel('Cell Number')
            axs[2*n+1].set_xlabel('Time (s)')
            #divider2 = make_axes_locatable(axs[2*n+1])
            #cax2 = divider2.append_axes("right", size="3%", pad=0.5)
            #f.colorbar(im, cax=cax2, orientation='vertical')
            
        f.tight_layout()   
        plt.savefig(os.path.join(save_location, bin_name+'-Event_analysis_heatmap '+data_type+'.pdf'))
        plt.close()


def check_data(list_files=None, save_path=None):
    
    if (list_files is None):
        list_files = save_session.load_variable(easygui.fileopenbox())
        
    t0 = t.time()
    files = [list_files[key] for key in list(list_files.keys())]
    
    combined_loc = {}
    combined_whisk = {}
    combined_whisk_only = {}
    
    for file in tqdm(files):
        data = save_session.load_variable(file+'.pickle')
        
        # Extract base data
        dF = data['dF']
        fs = data['Settings']['fs']
        name_file = data['table_data']['File'][0]
        negative_F0 = data['negative_F0_cells']
        
        
        if 'pearson_shuffle' in data:
            locomotion_binary = data['Locomotion_data']['binary_movement']
            speed = data['Locomotion_data']['speed']
            pearson = data['pearson_shuffle']['pearson']
            if 'Whisking_data' in data:
                whisking = data['Whisking_data']['normalized_trace']
                
            
            
            # Locomotion
            locomotion_dic = get_event_location(locomotion_binary, fs, speed = speed,
                                                exclusion_window= True, exclusion_time= [-5, 0], inactive = locomotion_binary, 
                                                remove_short= False, min_duration= 5, 
                                                join_bouts= True, max_duration= 0.5,
                                                separate_bins=True, bins=[2, 4, 6])
            locomotion_loc = locomotion_dic['bin_location']
            
            if locomotion_loc:
                for key in list(locomotion_loc.keys()):
                    if locomotion_loc[key]:
                        if key in combined_loc:
                            pass
                        else:
                            combined_loc[key] = {'Total': {},
                                                 'Positive': {},
                                                 'Negative': {}
                                                 }
                        save_loc = combined_loc[key]

                        locomotion_events, time, _ = binary_calculations.aligned_events(dF, locomotion_loc[key], fs, time_before= 10, time_after= 10)
                        speed_traces, _, _ = binary_calculations.aligned_events(speed, locomotion_loc[key], fs, time_before= 10, time_after= 10)
                        if 'Whisking_data' in data:
                            whisking_traces, _, _ = binary_calculations.aligned_events(whisking, locomotion_loc[key], fs, time_before= 10, time_after= 10)
                        
                        if locomotion_events.size == 1:
                            pass
                        else:
                            if len(locomotion_events) == 1:
                                if negative_F0[0] == False:
                                    # combined_loc['Total'][name_file] = {'time': time,
                                    #                                     'events': locomotion_events,
                                    #                                     'speed': speed_traces,
                                    #                                     'pearson': pearson}
                                    save_loc['Total'][name_file] = {'time': time,
                                                                        'events': locomotion_events,
                                                                        'speed': speed_traces,
                                                                        'pearson': pearson}
                                    if 'Whisking_data' in data:
                                        #combined_loc['Total'][name_file]['whisking'] = whisking_traces
                                        save_loc['Total'][name_file]['whisking'] = whisking_traces
                                        
                                    if pearson[0][0] > 0:
                                        # combined_loc['Positive'][name_file] = {'time': time,
                                        #                                        'events': locomotion_events,
                                        #                                        'speed': speed_traces,
                                        #                                        'pearson': pearson}
                                        save_loc['Positive'][name_file] = {'time': time,
                                                                                'events': locomotion_events,
                                                                                'speed': speed_traces,
                                                                                'pearson': pearson}
                                        
                                        if 'Whisking_data' in data:
                                            #combined_loc['Positive'][name_file]['whisking'] = whisking_traces
                                            save_loc['Positive'][name_file]['whisking'] = whisking_traces
                                            
                                    else:
                                        # combined_loc['Negative'][name_file] = {'time': time,
                                        #                                        'events': locomotion_events,
                                        #                                        'speed': speed_traces,
                                        #                                        'pearson': pearson}
                                        save_loc['Negative'][name_file] = {'time': time,
                                                                               'events': locomotion_events,
                                                                               'speed': speed_traces,
                                                                               'pearson': pearson}
                                        if 'Whisking_data' in data:
                                            # combined_loc['Negative'][name_file]['whisking'] = whisking_traces
                                            save_loc['Negative'][name_file]['whisking'] = whisking_traces
                                                               
                            else:
                                locomotion_total = np.delete(locomotion_events, [i for i, x in enumerate(list(negative_F0)) if x[0]==True], 0)
                                pearson_total = np.delete(pearson, [i for i, x in enumerate(list(negative_F0)) if x[0]==True], 0)
                                # combined_loc['Total'][name_file] = {'time': time,
                                #                                     'events': locomotion_total,
                                #                                     'speed': speed_traces,
                                #                                     'pearson': pearson_total}
                                save_loc['Total'][name_file] = {'time': time,
                                                                    'events': locomotion_total,
                                                                    'speed': speed_traces,
                                                                    'pearson': pearson_total}
                                if 'Whisking_data' in data:
                                    # combined_loc['Total'][name_file]['whisking'] = whisking_traces
                                    save_loc['Total'][name_file]['whisking'] = whisking_traces
                                
                                locomotion_positive = np.delete(locomotion_events, [i for i, (x, p) in enumerate(zip(list(negative_F0), list(pearson))) if (x[0]==True or p[0]<0)], 0)
                                pearson_positive = np.delete(pearson, [i for i, (x, p) in enumerate(zip(list(negative_F0), list(pearson))) if (x[0]==True or p[0]<0)], 0)
                                if not list(locomotion_positive):
                                    pass
                                else:
                                    # combined_loc['Positive'][name_file] = {'time': time,
                                    #                                        'events': locomotion_positive,
                                    #                                        'speed': speed_traces,
                                    #                                        'pearson': pearson_positive}
                                    save_loc['Positive'][name_file] = {'time': time,
                                                                           'events': locomotion_positive,
                                                                           'speed': speed_traces,
                                                                           'pearson': pearson_positive}
                                    if 'Whisking_data' in data:
                                        # combined_loc['Positive'][name_file]['whisking'] = whisking_traces
                                        save_loc['Positive'][name_file]['whisking'] = whisking_traces
                                    
                                locomotion_negative = np.delete(locomotion_events, [i for i, (x, p) in enumerate(zip(list(negative_F0), list(pearson))) if (x[0]==True or p[0]>0)], 0)
                                pearson_negative = np.delete(pearson, [i for i, (x, p) in enumerate(zip(list(negative_F0), list(pearson))) if (x[0]==True or p[0]>0)], 0)
                                if not list(locomotion_negative):
                                    pass
                                else:
                                    # combined_loc['Negative'][name_file] = {'time': time,
                                    #                                        'events': locomotion_negative,
                                    #                                        'speed': speed_traces,
                                    #                                        'pearson': pearson_negative}
                                    save_loc['Negative'][name_file] = {'time': time,
                                                                           'events': locomotion_negative,
                                                                           'speed': speed_traces,
                                                                           'pearson': pearson_negative}
                                    if 'Whisking_data' in data:
                                        # combined_loc['Negative'][name_file]['whisking'] = whisking_traces
                                        save_loc['Negative'][name_file]['whisking'] = whisking_traces
                        
                
            if 'Whisking_data' in data:
                whisking_binary = data['Whisking_data']['original_binary_whisking']
                inactive_binary = binary_calculations.get_inactive(locomotion_binary, whisking_binary)
                whisking = data['Whisking_data']['normalized_trace']
                
                # Whisking
                whisking_dic = get_event_location(whisking_binary, fs, 
                                                  remove_short= True, min_duration= 0.5, 
                                                  join_bouts= True, max_duration= 1)
                whisking_loc = whisking_dic['bin_location']
                
                if whisking_loc:
                    for key in list(whisking_loc.keys()):
                        if whisking_loc[key]:
                            if key in combined_whisk:
                                pass
                            else:
                                combined_whisk[key] = {'Total': {},
                                                     'Positive': {},
                                                     'Negative': {}
                                                     }
                            save_whisk = combined_whisk[key]
                            
                            whisking_events, time, _ = binary_calculations.aligned_events(dF, whisking_loc[key], fs, time_before= 10, time_after= 10)
                            speed_traces, _, _ = binary_calculations.aligned_events(speed, whisking_loc[key], fs, time_before= 10, time_after= 10)
                            whisking_traces, _, _ = binary_calculations.aligned_events(whisking, whisking_loc[key], fs, time_before= 10, time_after= 10)
                            
                            if whisking_events.size == 1:
                                pass
                            else:
                                if len(whisking_events) == 1:
                                    if negative_F0[0] == False:
                                        save_whisk['Total'][name_file] = {'time': time,
                                                                     'events': whisking_events,
                                                                     'speed': speed_traces,
                                                                     'whisking': whisking_traces,
                                                                     'pearson': pearson}
                                        if pearson[0][0] > 0:
                                            save_whisk['Positive'][name_file] = {'time': time,
                                                                                     'events': whisking_events,
                                                                                     'speed': speed_traces,
                                                                                     'whisking': whisking_traces,
                                                                                     'pearson': pearson}
                                        else:
                                            save_whisk['Negative'][name_file] = {'time': time,
                                                                                     'events': whisking_events,
                                                                                     'speed': speed_traces,
                                                                                     'whisking': whisking_traces,
                                                                                     'pearson': pearson}
                                else:
                                    whisking_total = np.delete(whisking_events, [i for i, x in enumerate(list(negative_F0)) if x[0]==True], 0)
                                    pearson_total = np.delete(pearson, [i for i, x in enumerate(list(negative_F0)) if x[0]==True], 0)
                                    save_whisk['Total'][name_file] = {'time': time,
                                                                 'events': whisking_total,
                                                                 'speed': speed_traces,
                                                                 'whisking': whisking_traces,
                                                                 'pearson': pearson_total}
                                    
                                    whisking_positive = np.delete(whisking_events, [i for i, (x, p) in enumerate(zip(list(negative_F0), list(pearson))) if (x[0]==True or p[0]<0)], 0)
                                    pearson_positive = np.delete(pearson, [i for i, (x, p) in enumerate(zip(list(negative_F0), list(pearson))) if (x[0]==True or p[0]<0)], 0)
                                    if not list(whisking_positive):
                                        pass
                                    else:
                                        save_whisk['Positive'][name_file] = {'time': time,
                                                                     'events': whisking_positive,
                                                                     'speed': speed_traces,
                                                                     'whisking': whisking_traces,
                                                                     'pearson': pearson_positive}
                                    
                                    whisking_negative = np.delete(whisking_events, [i for i, (x, p) in enumerate(zip(list(negative_F0), list(pearson))) if (x[0]==True or p[0]>0)], 0)
                                    pearson_negative = np.delete(pearson, [i for i, (x, p) in enumerate(zip(list(negative_F0), list(pearson))) if (x[0]==True or p[0]>0)], 0)
                                    if not list(whisking_negative):
                                        pass
                                    else:
                                        save_whisk['Negative'][name_file] = {'time': time,
                                                                     'events': whisking_negative,
                                                                     'speed': speed_traces,
                                                                     'whisking': whisking_traces,
                                                                     'pearson': pearson_negative}
                                
                                if data['Whisking_data']['whisking only']: 
                                    whisking_only_binary = data['Whisking_data']['whisking only']['binary']
                        
                                    # Whisking only
                                    # wo_loc = get_event_location(whisking_only_binary, fs, 
                                    #                             exclusion_window= True, exclusion_time= [-5,5], inactive = inactive_binary) 
                                    wo_dic = get_event_location(whisking_only_binary, fs, 
                                                                exclusion_window= True, exclusion_time= [-5,5], inactive = locomotion_binary)
                                    wo_loc = wo_dic['bin_location']
                                    
                                    if wo_loc:
                                        for key in list(wo_loc.keys()):
                                            if wo_loc[key]:
                                                if key in combined_whisk_only:
                                                    pass
                                                else:
                                                    combined_whisk_only[key] = {'Total': {},
                                                                         'Positive': {},
                                                                         'Negative': {}
                                                                         }
                                                save_whisk_only = combined_whisk_only[key]
                                        
                                                wo_events, time, _ = binary_calculations.aligned_events(dF, wo_loc[key], fs, time_before= 10, time_after= 10)
                                                speed_traces, _, _ = binary_calculations.aligned_events(speed, wo_loc[key], fs, time_before= 10, time_after= 10)
                                                whisking_traces, _, _ = binary_calculations.aligned_events(whisking, wo_loc[key], fs, time_before= 10, time_after= 10)
                                                
                                                
                                                if wo_events.size == 1:
                                                    pass
                                                else:
                                                    if len(wo_events) == 1:
                                                        if negative_F0[0] == False:
                                                            save_whisk_only['Total'][name_file] = {'time': time,
                                                                                                       'events': wo_events,
                                                                                                       'speed': speed_traces, 
                                                                                                       'whisking': whisking_traces,
                                                                                                       'pearson': pearson}
                                                            if pearson[0][0] > 0:
                                                                save_whisk_only['Positive'][name_file] = {'time': time,
                                                                                                              'events': wo_events,
                                                                                                              'speed': speed_traces, 
                                                                                                              'whisking': whisking_traces,
                                                                                                              'pearson': pearson}
                                                            else:
                                                                save_whisk_only['Negative'][name_file] = {'time': time,
                                                                                                              'events': wo_events,
                                                                                                              'speed': speed_traces, 
                                                                                                              'whisking': whisking_traces,
                                                                                                              'pearson': pearson}
                                                                
                                                    else:
                                                        wo_total = np.delete(wo_events, [i for i, x in enumerate(list(negative_F0)) if x[0]==True], 0)
                                                        pearson_total = np.delete(pearson, [i for i, x in enumerate(list(negative_F0)) if x[0]==True], 0)
                                                        save_whisk_only['Total'][name_file] = {'time': time,
                                                                                                   'events': wo_total,
                                                                                                   'speed': speed_traces,
                                                                                                   'whisking': whisking_traces,
                                                                                                   'pearson': pearson_total}
                                                        
                                                        wo_positive = np.delete(wo_events, [i for i, (x, p) in enumerate(zip(list(negative_F0), list(pearson))) if (x[0]==True or p[0]<0)], 0)
                                                        pearson_positive = np.delete(pearson, [i for i, (x, p) in enumerate(zip(list(negative_F0), list(pearson))) if (x[0]==True or p[0]<0)], 0)
                                                        if not list(wo_positive):
                                                            pass
                                                        else:
                                                            save_whisk_only['Positive'][name_file] = {'time': time,
                                                                                                          'events': wo_positive,
                                                                                                          'speed': speed_traces,
                                                                                                          'whisking': whisking_traces,
                                                                                                          'pearson': pearson_positive}
                                                        
                                                        wo_negative = np.delete(wo_events, [i for i, (x, p) in enumerate(zip(list(negative_F0), list(pearson))) if (x[0]==True or p[0]>0)], 0)
                                                        pearson_negative = np.delete(pearson, [i for i, (x, p) in enumerate(zip(list(negative_F0), list(pearson))) if (x[0]==True or p[0]>0)], 0)
                                                        if not list(wo_negative):
                                                            pass
                                                        else:
                                                            save_whisk_only['Negative'][name_file] = {'time': time,
                                                                                                        'events': wo_negative,
                                                                                                        'speed': speed_traces,
                                                                                                        'whisking': whisking_traces,
                                                                                                        'pearson': pearson_negative}
        
        else:
            pass
    
    # Create save location
    if (save_path is not None):
        save_location = os.path.join(save_path, 'Event-based Analysis')
        os.mkdir(save_location)
    else:
        save_location = os.path.join(easygui.diropenbox(), 'Event-based Analysis')
        os.mkdir(save_location)
    
    # Generate combined event
    if combined_loc[list(combined_loc.keys())[0]]:
        data_loc, dic_loc = combined_events(combined_loc, 'Locomotion', save_location)
        plot_figure(data_loc, 'Locomotion', save_location)
        heatmap_events(dic_loc, 'Locomotion', save_location)
        
    if combined_whisk[list(combined_whisk.keys())[0]]:
        data_whisk, dic_whisk = combined_events(combined_whisk, 'Whisking', save_location)
        plot_figure(data_whisk, 'Whisking', save_location)
        heatmap_events(dic_whisk, 'Whisking', save_location)
    
    if combined_whisk_only[list(combined_whisk_only.keys())[0]]:
        data_wo, dic_wo = combined_events(combined_whisk_only, 'Whisking Only', save_location)
        plot_figure(data_wo, 'Whisking Only', save_location)
        heatmap_events(dic_wo, 'Whisking Only', save_location)
    
    print('time %4.2f sec'%(t.time()-(t0)))
    
    if combined_loc[list(combined_loc.keys())[0]]:
        if combined_whisk[list(combined_whisk.keys())[0]]:
            if combined_whisk_only[list(combined_whisk_only.keys())[0]]:
                return [dic_loc, dic_whisk, dic_wo]
            else:
                return [dic_loc, dic_whisk, None]
        else:
            return [dic_loc, None, None]
    else:
        return [None, None, None]
    

def create_bins(bins = [1, 2, 4]):
    
    bin_range_name = []
    bin_range_val = []
    
    for n, i in enumerate(bins):
        if n == 0:
            bin_range_name.append('Less than {}'.format(str(i)))
            bin_range_val.append([0, float(i)])
        else:
            bin_range_name.append('From {} to {}'.format(str(bins[n-1]), str(i)))
            bin_range_val.append([float(bins[n-1]), float(i)])
    bin_range_name.append('More than {}'.format(str(bins[-1])))
    bin_range_val.append([float(bins[-1]), 'max'])
    
    return bin_range_name, bin_range_val


def bin_by_speed_and_duration(variable= 'locomotion', variable_bin= [1, 2, 4], duration_bin= [2, 4, 6], list_files=None, save_path=None):
    
    """
    Necessary variables:
        Speed
        Whisking
        Bins (speed or whisking)
    """
    
    # Define list of files
    if (list_files is None):
        list_files = save_session.load_variable(easygui.fileopenbox())
        
    t0 = t.time()
    
    # Create data tree
    combined_data = {}
    variable_bin_name, variable_bin_val = create_bins(bins = variable_bin)
    duration_bin_name, duration_bin_val = create_bins(bins = duration_bin)
    
    for v_name, v_val in zip(variable_bin_name, variable_bin_val):
        combined_data[v_name] = {}
        combined_data[v_name]['range'] = v_val
        for d_name, d_val in zip(duration_bin_name, duration_bin_val):
            combined_data[v_name][d_name] = {}
            combined_data[v_name]['range'] = d_val
            for type_data in ['Total', 'Positive', 'Negative', 'Nonsig']:
                combined_data[v_name][d_name][type_data] = {}
    
    # Find data
    files = [list_files[key] for key in list(list_files.keys())]
    for file in tqdm(files):
        data = save_session.load_variable(file+'.pickle')
        
        # Extract base data
        dF = data['dF']
        fs = data['Settings']['fs']
        name_file = data['table_data']['File'][0]
        negative_F0 = data['negative_F0_cells']
        
        if 'pearson_shuffle' in data:
            locomotion_binary = data['Locomotion_data']['binary_movement']
            speed = data['Locomotion_data']['speed']
            pearson = data['pearson_shuffle']['pearson']
            if 'Whisking_data' in data:
                whisking = data['Whisking_data']['normalized_trace']
            
            
        else:
            pass
        
    print('time %4.2f sec'%(t.time()-(t0)))
    
    # Create save location
    if (save_path is not None):
        save_location = os.path.join(save_path, 'Event-based Analysis')
        os.mkdir(save_location)
    else:
        save_location = os.path.join(easygui.diropenbox(), 'Event-based Analysis')
        os.mkdir(save_location)
        
    return
# def check_single_file():
    
#     data = save_session.load_variable(easygui.fileopenbox())
    
#     # Extract base data
#     dF = data['dF']
#     locomotion_binary = data['Locomotion_data']['binary_movement']
#     whisking_binary = data['Whisking_data']['original_binary_whisking']
#     whisking_only_binary = data['Whisking_data']['whisking only']['binary']
#     inactive_binary = binary_calculations.get_inactive(locomotion_binary, whisking_binary)
#     #time = data['Locomotion_data']['t']
#     fs = data['Settings']['fs']
    
#     f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    
#     # Locomotion raw
#     loc = get_event_location(locomotion_binary, fs)
#     mean, sem, events, time = calc_mean_event(dF, loc, fs, time_before= 8, time_after= 5)
#     modulation_index = get_event_MI(time, events)
#     ax1.set_title('Locomotion: MI = {}'.format(np.mean(modulation_index)))
#     ax1.plot(time, mean, color='red')
#     ax1.fill_between(time, mean-sem, mean+sem, alpha=.5, color= 'grey')
    
#     # Locomotion no short + join
#     loc = get_event_location(locomotion_binary, fs, remove_short= True, min_duration= 0.5, join_bouts= True, max_duration= 1)
#     mean, sem, events, time = calc_mean_event(dF, loc, fs, time_before= 8, time_after= 5)
#     modulation_index = get_event_MI(time, events)
#     ax2.set_title('Locomotion: MI = {}'.format(np.mean(modulation_index)))
#     ax2.plot(time, mean, color='red')
#     ax2.fill_between(time, mean-sem, mean+sem, alpha=.5, color= 'grey')
    
#     # Whisking raw
#     loc = get_event_location(whisking_only_binary, fs)
#     mean, sem, events, time = calc_mean_event(dF, loc, fs, time_before= 8, time_after= 5)
#     modulation_index = get_event_MI(time, events)
#     ax3.set_title('Whisking Only: MI = {}'.format(np.mean(modulation_index)))
#     ax3.plot(time, mean, color='red')
#     ax3.fill_between(time, mean-sem, mean+sem, alpha=.5, color= 'grey')
    
#     # Whisking no short + join
#     loc = get_event_location(whisking_only_binary, fs, exclusion_window= True, exclusion_time= [-0.5,0], inactive = inactive_binary)
#     mean, sem, events, time = calc_mean_event(dF, loc, fs, time_before= 8, time_after= 5)
#     modulation_index = get_event_MI(time, events)
#     ax4.set_title('Whisking Only: MI = {}'.format(np.mean(modulation_index)))
#     ax4.plot(time, mean, color='red')
#     ax4.fill_between(time, mean-sem, mean+sem, alpha=.5, color= 'grey')
    
#     plt.show()
    


# if __name__ == "__main__":
#     data = check_data(save_path=os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop'))
#     figure_traces(data, title='Something', save_path=os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop'))

#figure_traces(data, title='VIP BC', save_path='D:/Updated Results/SST BC GCaMP6s/SST_BC_Percentile_New1/Event-based Analysis/')   
