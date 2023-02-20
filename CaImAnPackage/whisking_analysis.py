# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:33:54 2021

@author: m.debritovanvelze
"""
import numpy as np
from scipy import signal
import easygui
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import glob
import pandas as pd
import statistics

import binary_calculations, get_dataRE

# def process_whisking(file_path, settings, rec_points=9000):
#     """
#     Marcel van Velze (m.debritovanvelze@icm-institute.org)
#     2021.02.23
    
#     Function that loads the output of facemap, resamples the data to the same 
#     length as the suite2p data and calculates a threshold.
    
#     Threshold for now is 3rd quartile. To be changed!

#     Parameters
#     ----------
#     file_path : .npy file
#         File containing the analyzed data from facemap
#     rec_points : int
#         The number of frames in the suite2p data.
#         Used to downsample the data if needed.
#         The default is 9000.

#     Returns
#     -------
#     new_whisker_motion : numpy array
#         Downsampled whisking trace.
#     whisker_motion : numpy array
#         Original whisking trace from facemap
#     threshold : float
#         Value of the threshold calculated using the 3rd quartile of the distribution.

#     """
#     whisking_data = np.load(file_path, allow_pickle=True).item()
#     if 'motion' in whisking_data:
#         whisker_motion = np.copy(whisking_data['motion'][1])
#         len_rec = len(whisker_motion)
        
#         # Down sample data to fit 2P data
#         if len_rec != rec_points:
#             if len_rec > rec_points:
#                 new_whisker_motion = signal.resample(whisker_motion, rec_points)
#                 print('-- Whisker data resampled to {n_points}'.format(n_points = rec_points))    
#             else:
#                 new_whisker_motion = signal.resample(whisker_motion, rec_points)
#                 print('-- Whisker data resampled to {n_points}'.format(n_points = rec_points)) 
#         else:
#             new_whisker_motion = np.copy(whisker_motion)
#             print('No resampling needed')
        
#         # Plot data
#         fig, (ax1, ax2, ax3) =  plt.subplots(3,1)
#         # Plot trace
#         t_original = np.linspace(0,int(rec_points/30),len_rec, )
#         t_new = np.linspace(0,int(rec_points/30),rec_points)
#         ax1.plot(t_original, whisker_motion, 'k', label='Original')
#         ax1.plot(t_new, new_whisker_motion, 'r', alpha=0.5, label='Resampled')
#         ax1.set_ylabel('AU')
#         ax1.set_xlabel('Time (s)')
#         ax1.legend()
#         # Plot distribution
#         ax2.hist(new_whisker_motion, bins=100, weights=(np.zeros_like(new_whisker_motion) + 1. / new_whisker_motion.size)*100)
#         ax2.set_ylabel('Probability')
#         # Plot threshold
#         threshold = np.percentile(new_whisker_motion, 75)
#         ax3.plot(t_new, new_whisker_motion, 'k', label='Whisker Motion')
#         ax3.axhline(y = threshold, color = 'r', linestyle = 'dashed', label='Threshold (3rd quartile)')
#         ax3.legend()
#         if settings['do_show']:
#             plt.show()
#         fig.savefig('{}/whisking_data.pdf'.format(settings['save_path']))
#         return new_whisker_motion, whisker_motion, threshold
#     else:
#         print('-- Whisking analysis results not available')
 
# plot = True
# rec_points = 10000
# frame_rate = 30

# # Get whisker analysis results
# dic = np.load(easygui.fileopenbox(), allow_pickle=True).item()    
# file_path = easygui.fileopenbox()
# new_whisker_motion, _, _ = process_whisking(easygui.fileopenbox())

def new_test(file_path, sigma=1, percentile=20, min_duration=0.25, fs=30):
    
    # Get whisking data
    rec_points = 9000
    whisking_data = np.load(file_path, allow_pickle=True).item()
    if 'motion' in whisking_data:
        whisker_motion = np.copy(whisking_data['motion'][1])
        len_rec = len(whisker_motion)
        
        # Down sample data to fit 2P data
        if len_rec != rec_points:
            if len_rec > rec_points:
                new_whisker_motion = signal.resample(whisker_motion, rec_points)
                print('-- Whisker data resampled to {n_points}'.format(n_points = rec_points))    
            else:
                new_whisker_motion = signal.resample(whisker_motion, rec_points)
                print('-- Whisker data resampled to {n_points}'.format(n_points = rec_points)) 
        else:
            new_whisker_motion = np.copy(whisker_motion)
            print('No resampling needed')
    
    # Filter data using gaussian filter
    filtered_whisking = gaussian_filter1d(new_whisker_motion, sigma)
    # Normalize data using minmax method
    normalized = (filtered_whisking-min(filtered_whisking))/(max(filtered_whisking)-min(filtered_whisking))
    
    # Create whisk vs no whisk
    binary_whisking = (normalized > (percentile/100)) * 1
    bool_binary_whisking = (normalized > (percentile/100))
    # Min duration of 250 msec
    delta, loc = get_dataRE.calc_event_duration(binary_whisking)
    new_binary_whisking, new_delta, newloc = get_dataRE.remove_short_events(binary_whisking, delta, loc, fs, min_duration)
    
    # Plot results
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
    ax1.plot(new_whisker_motion)
    ax1.title.set_text('Original')

    ax2.plot(normalized)
    ax2.title.set_text('Filtered (sigma='+str(sigma)+') and Normalized + Threshold ('+str(percentile)+'th percentile of max)')
    ax2.axhline(y = (percentile/100), color = 'r', linestyle = 'dashed', label='Threshold')
    ax2.legend()
    
    ax3.plot(binary_whisking)
    ax3.title.set_text('Binary whisking')
    
    ax4.plot(new_binary_whisking)
    ax4.title.set_text('Binary whisking (min duration = '+str(min_duration)+'s)')
    
    fig.tight_layout()
    plt.show()
    
    
    
def new_test2(dir_path):
    
    settings = {
                'N_samples': 9000,
                
                # Whisking data
                'sigma': 5,
                'percentile': 20,
                'min_duration': 1,
                'fs': 30,
                
                # Locomotion data
                'speed threshold':0.1,
                'time_before':0.5,
                'time_after':2.5,
                'remove_short_events': False, 
                'min_event_duration': 2
                }
    
    # Get file paths
    whisking_file = glob.glob(dir_path+'/*.npy')
    locomotion_file = glob.glob(dir_path+'/*.abf')
    if locomotion_file == []:
        locomotion_file = glob.glob(dir_path+'/*.txt')
        
    if locomotion_file == [] or whisking_file == []:
        print('Files not found!')
    else:
        
        ####################
        # Get whisking data#
        ####################
        whisking_data = np.load(whisking_file[0], allow_pickle=True).item()
        if 'motion' in whisking_data:
            whisker_motion = np.copy(whisking_data['motion'][1])
            len_rec = len(whisker_motion)
            
            # Down sample data to fit 2P data
            if len_rec != settings['N_samples']:
                if len_rec > settings['N_samples']:
                    new_whisker_motion = signal.resample(whisker_motion, settings['N_samples'])
                    print('-- Whisker data resampled to {n_points}'.format(n_points = settings['N_samples']))    
                else:
                    new_whisker_motion = signal.resample(whisker_motion, settings['N_samples'])
                    print('-- Whisker data resampled to {n_points}'.format(n_points = settings['N_samples'])) 
            else:
                new_whisker_motion = np.copy(whisker_motion)
                print('No resampling needed')
        
        # Filter data using gaussian filter
        filtered_whisking = gaussian_filter1d(new_whisker_motion, settings['sigma'])
        # Normalize data using minmax method
        #normalized = (filtered_whisking-min(filtered_whisking))/(max(filtered_whisking)-min(filtered_whisking))
        normalized = (filtered_whisking-min(filtered_whisking))/((np.percentile(filtered_whisking, 90))-min(filtered_whisking))
        
        # Create whisk vs no whisk
        binary_whisking = (normalized > (settings['percentile']/100)) * 1
        bool_binary_whisking = (normalized > (settings['percentile']/100))
        # Min duration of 250 msec
        delta, loc = get_dataRE.calc_event_duration(binary_whisking)
        new_binary_whisking, new_delta, newloc = get_dataRE.remove_short_events(binary_whisking, delta, loc, settings['fs'], settings['min_duration'])
        
        ######################
        # Get locomotion data#
        ######################
        locomotion = get_dataRE.single_getspeed(locomotion_file[0], settings['N_samples'],
                                                            settings['speed threshold'], 
                                                            settings['time_before'],
                                                            settings['time_after'],
                                                            settings['fs'],
                                                            settings['remove_short_events'], 
                                                            settings['min_event_duration'])
        speed = locomotion['speed']
        filtered_locomotion = gaussian_filter1d(speed, settings['sigma'])
        binary_locomotion = (filtered_locomotion > settings['speed threshold']) * 1
        
        # Plot results
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1)
        ax1.plot(new_whisker_motion)
        ax1.title.set_text('Whisking')
        
        ax2.plot(normalized)
        ax2.title.set_text('Filtered (sigma='+str(settings['sigma'])+') and Normalized + Threshold ('+str(settings['percentile'])+'th percentile of max)')
        ax2.axhline(y = (settings['percentile']/100), color = 'r', linestyle = 'dashed', label='Threshold')
        
        ax3.plot(locomotion['speed'])
        ax3.title.set_text('Locomotion')
        
        ax4.plot(filtered_locomotion)
        ax4.title.set_text('Filtered (sigma='+str(settings['sigma'])+') + Threshold ('+str(settings['speed threshold'])+'cm/s)')
        ax4.axhline(y = (settings['percentile']/100), color = 'r', linestyle = 'dashed', label='Threshold')
    
        ax5.plot(binary_whisking, color='#bd0026', label='Whisking')
        ax5.plot(locomotion['binary_movement'], color='#74a9cf', label='Locomotion', alpha=0.6)
        ax5.title.set_text('Binary Locomotion and whisking')
        ax5.legend()
        
        df = pd.DataFrame({'locomotion': binary_locomotion, 'whisking': binary_whisking})
        loc_only = (df.index[(df['locomotion'] == 1) & (df['whisking'] == 0)].tolist())
        whisk_only = (df.index[(df['locomotion'] == 0) & (df['whisking'] == 1)].tolist())
        both = (df.index[(df['locomotion'] == 1) & (df['whisking'] == 1)].tolist())
        nothing = (df.index[(df['locomotion'] == 0) & (df['whisking'] == 0)].tolist())
        
        for i in enumerate(both):
            ax6.axvline(x=i[1], ymin=0, ymax=1, color='#fed976')
        for i in enumerate(loc_only):
            ax6.axvline(x=i[1], ymin=0, ymax=1, color='#74a9cf')
        for i in enumerate(whisk_only):
            ax6.axvline(x=i[1], ymin=0, ymax=1, color='#bd0026')
        ax6.title.set_text('Binary whisking and locomotion')
        fig.tight_layout()
        plt.show()
        
        fig1, ax1 = plt.subplots()
        ax1.pie([len(nothing)/settings['N_samples'], len(both)/settings['N_samples'], len(loc_only)/settings['N_samples'], len(whisk_only)/settings['N_samples']], labels=['Nothing', 'Both', 'Only locomotion', 'Only whisking'], colors= ['#f7f7f7', '#fed976', '#74a9cf', '#bd0026'], autopct='%1.1f%%', startangle=180, shadow=True)
        plt.show()
        
def process_whisking(file_path, settings, rec_points=9000):
    """
    Marcel van Velze (m.debritovanvelze@icm-institute.org)
    2021.10.28
    
    Function that loads the output of facemap, resamples the data to the same 
    length as the suite2p data and outputs a dictionary with whisking data

    Parameters
    ----------
    file_path : .npy file
        File containing the analyzed data from facemap
    rec_points : int
        The number of frames in the suite2p data.
        Used to downsample the data if needed.
        The default is 9000.

    Returns
    -------
    'path_analysis_file': path to facemap file
    'original_trace': output of facemap
    'resampled_trace': whisking trace resampled to 2P data
    'filtered_trace': filtered whisking trace
    'normalized_trace': normalized traces, min max
    'original_binary_whisking': binary whisking of original data
    'binary_whisking': binary whisking with short events removed 
    'location_bouts': location of bouts
    'duration bouts': duration of bouts
    """
    
    
    whisking_data = np.load(file_path, allow_pickle=True).item()
    
    if 'motion' in whisking_data:
        whisker_motion = np.copy(whisking_data['motion'][1])
        len_rec = len(whisker_motion)
        
        # Down sample data to fit 2P data
        if len_rec != rec_points:
            if len_rec > rec_points:
                new_whisker_motion = signal.resample(whisker_motion, rec_points)
                #print('-- Whisker data resampled to {n_points}'.format(n_points = settings['N_samples']))    
            else:
                new_whisker_motion = signal.resample(whisker_motion, rec_points)
                #print('-- Whisker data resampled to {n_points}'.format(n_points = settings['N_samples'])) 
        else:
            new_whisker_motion = np.copy(whisker_motion)
            #print('No resampling needed')
        
        # Filter data using gaussian filter
        filtered_whisking = gaussian_filter1d(new_whisker_motion, settings['whisking']['sigma'])
        
        # Normalize data using minmax method
        normalized = (filtered_whisking-min(filtered_whisking))/(max(filtered_whisking)-min(filtered_whisking))
        #normalized = (filtered_whisking-min(filtered_whisking))/((np.percentile(filtered_whisking, 90))-min(filtered_whisking))
        
        # Create binary whisking
        binary_whisking = (normalized > (settings['whisking']['percentile']/100)) * 1
        bool_binary_whisking = (normalized > (settings['whisking']['percentile']/100))
        
        # Remove short events
        if settings['whisking']['remove short bouts'] == True:
            new_binary_whisking = binary_calculations.remove_short_events(binary_whisking, settings['fs'], settings['whisking']['whisk_min_duration'])
        
            # Join events
            if settings['whisking']['join bouts'] == True:
                new_binary_whisking = binary_calculations.remove_short_interevent_periods(new_binary_whisking, settings['fs'], settings['whisking']['whisk_max_inter_bout'])
        
        else: 
            new_binary_whisking = np.copy(binary_whisking)
        
        dic = { 'path_analysis_file': file_path,
                'original_trace': whisker_motion,
                'resampled_trace': new_whisker_motion,
                'filtered_trace': filtered_whisking,
                'normalized_trace': normalized,
                'original_binary_whisking': binary_whisking,
                'binary_whisking': new_binary_whisking,
                } 
            
        change = list(np.where(np.diff(new_binary_whisking,prepend=np.nan))[0])    
        if len(change) == 1:
            dic['duration bouts'] = []
            dic['location_bouts'] = []
            dic['percentage_whisking'] = np.count_nonzero(new_binary_whisking)/len(new_binary_whisking)
            dic['mean event duration'] = []
            dic['max event duration'] = []
                  
        else:
            delta, loc = binary_calculations.calc_event_duration(new_binary_whisking)
            dic['duration bouts'] = delta
            dic['location_bouts'] = loc
            dic['percentage_whisking'] = np.count_nonzero(new_binary_whisking)/len(new_binary_whisking)
            dic['mean event duration'] = statistics.mean(delta)/settings['fs']
            dic['max event duration'] = max(delta)/settings['fs']
        
        return dic
    return {}

def whisking_only(binary_whisking, binary_locomotion, settings):
    binary_whisking_only = []
    for w, r in zip(binary_whisking, binary_locomotion):
        if w == 1 and r == 0:
            binary_whisking_only.append(1)
        else:
            binary_whisking_only.append(0)
    binary = binary_calculations.remove_short_events(binary_whisking_only, settings['fs'], settings['whisking']['whisk_min_duration'])
    if np.count_nonzero(binary) < 1:
        return []
    else:
        delta, loc = binary_calculations.calc_event_duration(binary)
        dic = {'binary': binary,
               'bout_duration': delta,
               'bout_location': loc}
        return dic
       
       
       
       
       
       
       
       