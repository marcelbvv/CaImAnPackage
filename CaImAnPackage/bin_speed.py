# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:50:05 2022

@author: m.debritovanvelze
"""
import numpy as np
import easygui
#import statistics
#import math
from scipy import stats
import matplotlib.pyplot as plt
import pickle
#from scipy import stats
import save_session
import pandas as pd


# list_files = np.load(easygui.fileopenbox(), allow_pickle=True)
# keys = list(list_files.keys())

# path = 'C:/Users/m.debritovanvelze/Desktop/Data and Figures/VIP BC/VIP_percentile/2022.04.19/M_145/TSeries-04192022-1035-003/Analysis_Data.pickle'
# data = np.load(path, allow_pickle=True)



def mean_per_cell(data, settings):
    # Import data
    dF = data['dF']
    if 'pearson_shuffle' in data:
        speed = data['Locomotion_data']['speed']
        pearson = data['pearson_shuffle']['pearson']
        sig = data['pearson_shuffle']['sig']
        
        # Create bins
        bins = np.arange(settings['min_val'], settings['max_val']+settings['bin_size'], settings['bin_size'])
        # Define bin centers
        centers = np.delete(bins, -1)+(settings['bin_size']/2)
        # Get index of points for each bin
        digitized_speed = np.digitize(speed, bins)
        # Create array
        mean_array = np.zeros((len(dF), len(centers)))
        sem_array = np.zeros((len(dF), len(centers)))
        #
        for cell in range(len(dF)):
            cell_bin_mean = [dF[cell][digitized_speed == i].mean() for i in range(1, len(bins))]
            cell_bin_sem = [stats.sem(dF[cell][digitized_speed == i]) for i in range(1, len(bins))]
            mean_array[cell] = cell_bin_mean
            sem_array[cell] = cell_bin_sem
        return (mean_array, sem_array, centers, pearson, sig)
    else:
        return ()

def plot_curve(curves, save_path=None):
    fig, ax = plt.subplots()
    for i in range(len(curves[0])):
        ax.plot(curves[2], curves[0][i], linewidth=0.8, label= 'R = %s - %s'%(curves[3][i][0], curves[4][i]))
        ax.fill_between(curves[2], curves[0][i]-curves[1][i], curves[0][i]+curves[1][i], color='#bdbdbd')
        ax.legend(prop={'size': 6})
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel('Speed bins (cm/s)')
        ax.set_ylabel(u'Δ F/F0')
    
    if save_path == None:
        path = easygui.diropenbox()
        pickle.dump(fig, open('{}/Binned_speed.fig.pickle'.format(path), 'wb')) 
        plt.savefig('{}/Binned_speed.pdf'.format(path))
    else:        
        pickle.dump(fig, open('{}/Binned_speed.fig.pickle'.format(save_path), 'wb')) 
        plt.savefig('{}/Binned_speed.pdf'.format(save_path))
    plt.close()
    
def open_figure():
    figx = pickle.load(open(easygui.fileopenbox(), 'rb'))
    figx.show()
    
def create_arrays(centers, data):
    dic_data = {'cell': data['cell'],
                'pearson': data['pearson']}
    mean_array = np.zeros((len(data['cell']), len(centers)))
    sem_array = np.zeros((len(data['cell']), len(centers)))
    for i, (m, s) in enumerate(zip(data['mean'], data['sem'])):
        mean_array[i] = m
        sem_array[i] = s
    
    dic_data['mean'] = mean_array
    dic_data['sem'] = sem_array
    dic_data['centers'] = centers
    
    return dic_data
    
def plot_combined_binned_speed(combined_data, save_path=None):
    
    if save_path == None:
        save_path = easygui.diropenbox()
    
    name_file = save_path+'/Grouped_binned_speed_data'+'.xlsx'
    writer = pd.ExcelWriter(name_file, engine='xlsxwriter')
    
    total = {'cell':[],
             'mean':[],
             'sem':[],
             'pearson':[],
             'sig':[]}
    positive = {'cell':[],
                'mean':[],
                'sem':[],
                'pearson':[],
                'sig':[]}
    negative = {'cell':[],
                'mean':[],
                'sem':[],
                'pearson':[],
                'sig':[]}
    nonsig = {'cell':[],
              'mean':[],
              'sem':[],
              'pearson':[],
              'sig':[]}
    
    centers = []
    
    for fov in combined_data:
        for cell, (mean, sem, pearson, sig) in enumerate(zip(combined_data[fov][0], combined_data[fov][1], combined_data[fov][3], combined_data[fov][4])):
            total['cell'].append('%s_%s'%(fov, cell))
            total['mean'].append(mean)
            total['sem'].append(sem)
            total['pearson'].append(pearson[0])
            total['sig'].append(sig)
            if sig == 'Non sig':
                nonsig['cell'].append('%s_%s'%(fov, cell))
                nonsig['mean'].append(mean)
                nonsig['sem'].append(sem)
                nonsig['pearson'].append(pearson[0])
                nonsig['sig'].append(sig)
            else:
                if pearson[0] > 0:
                    positive['cell'].append('%s_%s'%(fov, cell))
                    positive['mean'].append(mean)
                    positive['sem'].append(sem)
                    positive['pearson'].append(pearson[0])
                    positive['sig'].append(sig)
                elif pearson[0] < 0:
                    negative['cell'].append('%s_%s'%(fov, cell))
                    negative['mean'].append(mean)
                    negative['sem'].append(sem)
                    negative['pearson'].append(pearson[0])
                    negative['sig'].append(sig)
            centers = combined_data[fov][2]
    
    # Create arrays with data
    total = create_arrays(centers, total)
    positive = create_arrays(centers, positive)
    negative = create_arrays(centers, negative)
    nonsig = create_arrays(centers, nonsig)
    
    # Calculate average response
    
    labels = ['Bins', 'Mean', 'SEM', 'SD', 'N']
    
    total_mean = np.nanmean(total['mean'], axis=0)
    total_sem = stats.sem(total['mean'], axis=0, nan_policy='omit')
    total_sd = np.nanstd(total['mean'], axis=0)
    total_N = np.count_nonzero((~np.isnan(total['mean'])), axis=0)
    total_table = pd.DataFrame(np.concatenate((np.expand_dims(centers, axis=0), 
                                              np.expand_dims(total_mean, axis=0),
                                              np.expand_dims(total_sem, axis=0),
                                              np.expand_dims(total_sd, axis=0), 
                                              np.expand_dims(total_N, axis=0)), axis=0), labels)
    total_table.to_excel(writer, sheet_name='Total')
                               
    positive_mean = np.nanmean(positive['mean'], axis=0)
    positive_sem = stats.sem(positive['mean'], axis=0, nan_policy='omit')
    positive_sd = np.nanstd(positive['mean'], axis=0)
    positive_N = np.count_nonzero((~np.isnan(positive['mean'])), axis=0)
    positive_table = pd.DataFrame(np.concatenate((np.expand_dims(centers, axis=0), 
                                                  np.expand_dims(positive_mean, axis=0), 
                                                  np.expand_dims(positive_sem, axis=0), 
                                                  np.expand_dims(positive_sd, axis=0), 
                                                  np.expand_dims(positive_N, axis=0)), axis=0), labels)
    positive_table.to_excel(writer, sheet_name='Positive')
    
    negative_mean = np.nanmean(negative['mean'], axis=0)
    negative_sem = stats.sem(negative['mean'], axis=0, nan_policy='omit')
    negative_sd = np.nanstd(negative['mean'], axis=0)
    negative_N = np.count_nonzero((~np.isnan(negative['mean'])), axis=0)
    negative_table = pd.DataFrame(np.concatenate((np.expand_dims(centers, axis=0), 
                                                  np.expand_dims(negative_mean, axis=0), 
                                                  np.expand_dims(negative_sem, axis=0), 
                                                  np.expand_dims(negative_sd, axis=0), 
                                                  np.expand_dims(negative_N, axis=0)), axis=0), labels)
    negative_table.to_excel(writer, sheet_name='Negative')
    
    nonsig_mean = np.nanmean(nonsig['mean'], axis=0)
    nonsig_sem = stats.sem(nonsig['mean'], axis=0, nan_policy='omit')
    nonsig_sd = np.nanstd(nonsig['mean'], axis=0)
    nonsig_N = np.count_nonzero((~np.isnan(nonsig['mean'])), axis=0)
    nonsig_table = pd.DataFrame(np.concatenate((np.expand_dims(centers, axis=0), 
                                                  np.expand_dims(nonsig_mean, axis=0), 
                                                  np.expand_dims(nonsig_sem, axis=0), 
                                                  np.expand_dims(nonsig_sd, axis=0), 
                                                  np.expand_dims(nonsig_N, axis=0)), axis=0), labels)
    nonsig_table.to_excel(writer, sheet_name='Nonsig')
    
    writer.save()
    
    # Plot data
    fig, ax = plt.subplots()
    ax.plot(total['centers'], total_mean, color='#4d4d4d', label='Total (N=%s)'%(len(total['cell'])))
    ax.fill_between(total['centers'], total_mean-total_sem, total_mean+total_sem, color='#e0e0e0')
    ax.plot(positive['centers'], positive_mean, color='#2166ac', label='Positive (N=%s)'%(len(positive['cell'])))
    ax.fill_between(positive['centers'], positive_mean-positive_sem, positive_mean+positive_sem, color='#e0e0e0')
    ax.plot(negative['centers'], negative_mean, color='#b2182b', label='Negative (N=%s)'%(len(negative['cell'])))
    ax.fill_between(negative['centers'], negative_mean-negative_sem, negative_mean+negative_sem, color='#e0e0e0')
    ax.plot(nonsig['centers'], nonsig_mean, color='#fddbc7', label='Nonsig (N=%s)'%(len(nonsig['cell'])))
    ax.fill_between(nonsig['centers'], nonsig_mean-nonsig_sem, nonsig_mean+nonsig_sem, color='#e0e0e0')
    ax.legend(prop={'size': 6}, loc='upper left')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel('Speed bins (cm/s)')
    ax.set_ylabel(u'Δ F/F0')
    
    pickle.dump(fig, open('{}/Grouped_Binned_speed.fig.pickle'.format(save_path), 'wb')) 
    plt.savefig('{}/Grouped_Binned_speed.pdf'.format(save_path))
    plt.close()
    
    return{'Total': total, 'Positive': positive, 'Negative': negative, 'Nonsig': nonsig}


def calc_binned_speed(list_files=None, settings=None, plot_data=True, save_path=None):
    
    combined_data = {}
    
    if list_files == None:
        list_files = np.load(easygui.fileopenbox(), allow_pickle=True)
    
    if save_path == None:
        save_path = easygui.diropenbox()
    
    if settings == None:
        settings = {'bin_size': 0.1,
                    'min_val': 0,
                    'max_val': 20}
        
    for i in list_files.keys():
        data = np.load(list_files[i]+'.pickle', allow_pickle=True)
        curves = mean_per_cell(data, settings)
        if not curves:
            pass
        else:
            if plot_data:
                plot_curve(curves, save_path=data['Analysis_save_location'])
            combined_data[i] = curves
    
    grouped_data = plot_combined_binned_speed(combined_data, save_path)
    
    if save_path:
        save_session.save_variable((save_path+'\Binned_speed'), combined_data)
        save_session.save_variable((save_path+'\Grouped_binned_speed'), grouped_data)
    return
    
# # Test
# list_files = np.load(easygui.fileopenbox(), allow_pickle=True)
# keys = list(list_files.keys())
# path = list_files[keys[85]]+'.pickle'
# data = np.load(path, allow_pickle=True)

# settings = {'bin_size': 1,
#             'min_val': 0,
#             'max_val': 20}
            
# curves = mean_per_cell(data, settings)
# plot_curve(curves, save_path=None)


# # Plot saved figure 
# import matplotlib.pyplot as plt
# import pickle as pl
# import numpy as np
# import easygui

# fig = pl.load(open(easygui.fileopenbox(),'rb'))
# fig.show()