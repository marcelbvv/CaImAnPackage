#import save_session
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import math
from scipy.interpolate import interp1d
import pandas as pd
import itertools
import save_session

def normalize(data, method= 'none'):
    """
    Normalizes data:
        Method:
            -minmax
            -std
            -none
        """
    
    if np.ndim(data) == 1:
        data = np.reshape(data, (1, len(data)))
    norm_data = np.zeros((np.shape(data)))
    
    for i in range(0, len(data)):
        if method == 'minmax':
            norm_data[i] = (data[i]-min(data[i]))/(max(data[i])-min(data[i]))
        
        elif method == 'std':
            norm_data[i] = data[i]/data[i].std()
        
        elif method == 'none':
            norm_data[i] = data[i]
            
    return norm_data

def crosscorrel(Signal1, Signal2, tmax, dt):
    """
    Yann Zerlaut
    argument : Signal1 (np.array()), Signal2 (np.array())
    returns : np.array()
    take two Signals, and returns their crosscorrelation function 
    CONVENTION:
    --------------------------------------------------------------
    when the peak is in the past (negative t_shift)
    it means that Signal2 is delayed with respect to Signal 1
    --------------------------------------------------------------
    """
    if len(Signal1)!=len(Signal2):
        print('Need two arrays of the same size !!')
        
    steps = int(tmax/dt) # number of steps to sum on
    time_shift = dt*np.concatenate([-np.arange(1, steps)[::-1], np.arange(steps)])
    CCF = np.zeros(len(time_shift))
    for i in np.arange(steps):
        ccf = np.corrcoef(Signal1[:len(Signal1)-i], Signal2[i:])
        CCF[steps-1+i] = ccf[0,1]
    for i in np.arange(steps):
        ccf = np.corrcoef(Signal2[:len(Signal1)-i], Signal1[i:])
        CCF[steps-1-i] = ccf[0,1]
    return CCF, time_shift


def get_lags(corr, shift, fs):
    """
    Returns index and time array for defined shift
    """
    n = shift * fs
    mid = (np.size(corr, 1)-1)/2
    if len(corr)%2 == 1:
        mid = int(mid)
    start = int(mid -n)
    stop = int(mid + n)
    time_trace = np.linspace(-shift, shift, shift*fs*2)
    #time_trace = np.reshape(time_trace, (1,len(time_trace)))
    return start, stop, time_trace

def calc_correlation(data1, data2):
    """
    Calculate correlation between array and trace
    data1 = array of fluorescence data
    data2 = one dimentional array 
    """
    mid = int((len(data2[0])-1)/2)
    if len(data2[0])%2 == 1:
        mid = int(mid)
    corr = np.zeros((np.shape(data1)))
    for i in range(0,len(data1)):
        print(i)
        corr[i] = signal.correlate(data1[i], data2[0], mode='same')/(math.sqrt(signal.correlate(data1[i],data1[i], mode='same')[mid]*signal.correlate(data2[0], data2[0], mode='same')[mid]))
    return corr, mid

def corrcoef(norm_dF, norm_speed):
    coefficient = []
    for i in range(0, len(norm_dF)):
        coefficient.append(np.corrcoef(norm_dF[i], norm_speed)[0,1])
    return coefficient

def plot_CC(CC, zero_vals, path, show= True, fs = 30, shift=60):
    time_trace = np.linspace(-shift, shift, shift*fs*2)
    plt.figure()
    grid = plt.GridSpec(1,2, wspace=0.4)
    ax1 = plt.subplot(grid[0, 0])
    ax1.set_title('Cross-Correlation with speed')
    ax1.set_ylabel('Cross-Correlation Coefficient')
    ax1.set_xlabel('Time (s)')
    ax1.axvline(linewidth=2, color='#bdbdbd', ls='--')
    ax2 = plt.subplot(grid[0,1])
    ax2.xaxis.set_ticks(np.arange(-1, 1.1, 0.5))
    ax2.set_title('Relative frequency histogram')
    ax2.set_xlabel('Cross-Correlation (zero-time)')
    ax2.set_ylabel('% Neurons')
    ax2.axvline(linewidth=2, color='#bdbdbd', ls='--')
    for i in range(0,len(CC)):
        ax1.plot(time_trace, CC[i], color ='#bdbdbd', linewidth=0.4)
    ax1.plot(time_trace, np.mean(CC, axis=0), color = '#636363')
    ax2.hist(zero_vals, weights=(np.zeros_like(zero_vals) + 1. / zero_vals.size)*100, bins=20, range = (-1,1), color = '#9ecae1',ec = '#ffffff', lw=1)
    plt.savefig('{}/cross_correlation'.format(path))
    if show == False:
        plt.close()
    return

# # Import Data
#     VIPnr1ketamine = save_session.load_variable(easygui.fileopenbox())
#     VIPnr1ketamine_slope = np.array(VIPnr1ketamine['R value Slope'].dropna().tolist(), dtype=np.float64)
#     #Combined_Data3 = save_session.load_variable(easygui.fileopenbox())
#     #data2 = np.array(Combined_Data3['Speed Slope'].dropna().tolist(), dtype=np.float64)
    
def plot_2_hist_CC(data1, data2):
    plt.figure()
    grid = plt.GridSpec(1,1, wspace=0.4)
    ax1 = plt.subplot(grid[0,0])
    #ax1.xaxis.set_ticks(np.arange(-1, 1.1, 0.5))
    ax1.set_title('Relative frequency histogram')
    ax1.set_xlabel('Cross-Correlation (zero-time)')
    ax1.set_ylabel('% Neurons')
    ax1.axvline(linewidth=2, color='#bdbdbd', ls='--')
    ax1.hist(data1, weights=(np.zeros_like(data1) + 1. / data1.size)*100, bins=20, range = (-1,1), color = '#9ecae1',ec = '#ffffff', lw=1)
    ax1.hist(data2, weights=(np.zeros_like(data2) + 1. / data2.size)*100, bins=20, range = (-1,1), color = '#fdbb84',ec = '#ffffff', lw=1, alpha = 0.7)
    return
             
def calculate(dF, speed, fs = 30, normalization = 'none', shift= 60):
    """
    Calculates cross-correlation and plots distribution
    Input:
        dF: 2D array of fluorescent data
        speed: 1D array of speed
        normalization: how data is normalized before cross correlation
            -minmax
            -std
            -none
        fs: sampling rate
        shift: time shift for cross-correlation (s)
    Output:
        zero_vals: cross-correlation coefficient at zero-time
    To do:
        -Save image
        
    Info on normalization:
        https://fr.mathworks.com/help/matlab/ref/xcorr.html 
    """
    # Normalize Data
    norm_dF = normalize(dF, normalization)
    norm_speed = normalize(speed, normalization)
    # Calc correlation
    corr, mid = calc_correlation(norm_dF, norm_speed)
    # Extract values
    zero_vals = corrcoef(norm_dF, norm_speed)
    start, stop, _ = get_lags(corr, shift, fs)
    
    return zero_vals, corr[:, start:stop]


def plot_combined_CC(CC_traces, name, save_location):
    
    """
    CC_traces: array of CC arrays
    name: File name
    save_location: location to save in
    
    """

    # Determine number of traces
    keys = list(CC_traces.keys())
    n_traces = 0
    labels = []
    for key in enumerate(keys):
        if 'traces' in CC_traces[key[1]]:
            n_traces += len(CC_traces[key[1]]['traces'])
            labels.append([key[1]]*len(CC_traces[key[1]]['traces']))
    labels = list(itertools.chain(*labels))
    labels.insert(0, 'Time')
    labels.insert(1, 'Mean')
    labels.insert(2, 'Std')
    labels.insert(3, 'Sem')

    # Create arrays
    time = CC_traces[keys[0]]['time']
    cross_correlation_array = np.zeros((n_traces, len(time)))
    n = 0
    for key in enumerate(keys): 
        if 'traces' in CC_traces[key[1]]:
            n_traces = len(CC_traces[key[1]]['traces'])
            cross_correlation_array[n:n+n_traces] = CC_traces[key[1]]['traces']
            n += n_traces                            

    mean = np.mean(cross_correlation_array, axis=0)
    std = np.std(cross_correlation_array, axis=0)
    sem = stats.sem(cross_correlation_array, axis=0)
    
    table_data = pd.DataFrame(np.concatenate((np.expand_dims(time, axis=0), np.expand_dims(mean, axis=0), np.expand_dims(std, axis=0), np.expand_dims(sem, axis=0), cross_correlation_array), axis=0), labels)
    table_data.to_excel('{}/Combined_CC_{}.xlsx'.format(save_location, name))
    
    plt.figure()
    for i in range(len(cross_correlation_array)):
        plt.plot(time, cross_correlation_array[i], color='grey', alpha=0.5)
    plt.plot(time, mean, color='r', label='Mean')
    plt.plot(time, mean+std, '--', color='blue', alpha=0.8, label='std')
    plt.plot(time, mean-std, '--', color='blue', alpha=0.8)
    plt.legend()
    plt.xlabel('Offset (s)')
    plt.ylabel('Correlation Coefficient')
    plt.savefig('{}/Combined_CC_{}'.format(save_location, name))
    plt.savefig('{}/Combined_CC_{}.pdf'.format(save_location, name))
    plt.close()
    
def plot_combined_CC_pearson(CC_traces, name, save_location):
    keys = list(CC_traces.keys())
    n_traces_total = 0
    labels_total = []
    n_traces_positive = 0
    labels_positive = []
    n_traces_negative = 0
    labels_negative = []
    
    for key in enumerate(keys):
        if 'traces' in CC_traces[key[1]]:
            # Total
            n_traces_total += len(CC_traces[key[1]]['traces'])
            labels_total.append([key[1]]*len(CC_traces[key[1]]['traces']))
            if 'pearson' in CC_traces[key[1]]:
                # Positive
                n_traces_positive += np.sum(CC_traces[key[1]]['pearson'] > 0)
                labels_positive.append([key[1]]*np.sum(CC_traces[key[1]]['pearson'] > 0))
                # Negative
                n_traces_negative += np.sum(CC_traces[key[1]]['pearson'] < 0)
                labels_negative.append([key[1]]*np.sum(CC_traces[key[1]]['pearson'] < 0))
    # Total            
    labels_total = list(itertools.chain(*labels_total))
    labels_total.insert(0, 'Time')
    labels_total.insert(1, 'Mean')
    labels_total.insert(2, 'Std')
    labels_total.insert(3, 'Sem')    
    # Positive
    labels_positive = list(itertools.chain(*labels_positive))
    labels_positive.insert(0, 'Time')
    labels_positive.insert(1, 'Mean')
    labels_positive.insert(2, 'Std')
    labels_positive.insert(3, 'Sem') 
    # Negative
    labels_negative = list(itertools.chain(*labels_negative))
    labels_negative.insert(0, 'Time')
    labels_negative.insert(1, 'Mean')
    labels_negative.insert(2, 'Std')
    labels_negative.insert(3, 'Sem') 
    
    time = CC_traces[keys[0]]['time']
    cross_correlation_total = np.zeros((n_traces_total, len(time)))
    cross_correlation_positive = np.zeros((n_traces_positive, len(time)))
    cross_correlation_negative = np.zeros((n_traces_negative, len(time)))
    
    n_total = 0
    n_positive = 0
    n_negative = 0
    for key in enumerate(keys): 
        if 'traces' in CC_traces[key[1]]:
            # Total
            n_traces = len(CC_traces[key[1]]['traces'])
            cross_correlation_total[n_total:n_total+n_traces] = CC_traces[key[1]]['traces']
            n_total += n_traces     
            if 'pearson' in CC_traces[key[1]]:
                # Positive
                n_traces = np.sum(CC_traces[key[1]]['pearson'] > 0)
                mask_positive = [CC_traces[key[1]]['pearson'] > 0][0][:,0]
                cross_correlation_positive[n_positive:n_positive+n_traces] = CC_traces[key[1]]['traces'][mask_positive,:]
                n_positive += n_traces 
                # Negative
                n_traces = np.sum(CC_traces[key[1]]['pearson'] < 0)
                mask_negative = [CC_traces[key[1]]['pearson'] < 0][0][:,0]
                cross_correlation_negative[n_negative:n_negative+n_traces] = CC_traces[key[1]]['traces'][mask_negative,:]
                n_negative += n_traces 
            
    mean_total = np.mean(cross_correlation_total, axis=0)
    std_total = np.std(cross_correlation_total, axis=0)
    sem_total = stats.sem(cross_correlation_total, axis=0)
    
    mean_positive = np.mean(cross_correlation_positive, axis=0)
    std_positive = np.std(cross_correlation_positive, axis=0)
    sem_positive = stats.sem(cross_correlation_positive, axis=0)
    
    mean_negative = np.mean(cross_correlation_negative, axis=0)
    std_negative = np.std(cross_correlation_negative, axis=0)
    sem_negative = stats.sem(cross_correlation_negative, axis=0)
    
    table_total = pd.DataFrame(np.concatenate((np.expand_dims(time, axis=0), np.expand_dims(mean_total, axis=0), np.expand_dims(std_total, axis=0), np.expand_dims(sem_total, axis=0), cross_correlation_total), axis=0), labels_total)
    table_positive = pd.DataFrame(np.concatenate((np.expand_dims(time, axis=0), np.expand_dims(mean_positive, axis=0), np.expand_dims(std_positive, axis=0), np.expand_dims(sem_positive, axis=0), cross_correlation_positive), axis=0), labels_positive)
    table_negative = pd.DataFrame(np.concatenate((np.expand_dims(time, axis=0), np.expand_dims(mean_negative, axis=0), np.expand_dims(std_negative, axis=0), np.expand_dims(sem_negative, axis=0), cross_correlation_negative), axis=0), labels_negative)
    writer = pd.ExcelWriter('{}/Combined_CC_{}.xlsx'.format(save_location, name), engine='xlsxwriter')
    table_total.to_excel(writer, sheet_name='Total')
    table_positive.to_excel(writer, sheet_name='Positive')
    table_negative.to_excel(writer, sheet_name='Negative')
    writer.save()
    dic = {'Total': table_total,
           'Positive': table_positive,
           'Negative': table_negative}
    
    # Plot
    fig, (ax1, ax2, ax3) =  plt.subplots(1,3, figsize=(15, 6))
    
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(time, mean_total, color='r', label='Mean')
    ax1.plot(time, mean_total+sem_total, '--', color='blue', alpha=0.8, label='sem')
    ax1.plot(time, mean_total-sem_total, '--', color='blue', alpha=0.8)
    ax1.title.set_text('Total (N=%s)'%n_total)
    ax1.set_xlabel('Offset (s)')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    ax2 = plt.subplot(1, 3, 2, sharey=ax1)
    ax2.plot(time, mean_positive, color='r', label='Mean')
    ax2.plot(time, mean_positive+sem_positive, '--', color='blue', alpha=0.8, label='sem')
    ax2.plot(time, mean_positive-sem_positive, '--', color='blue', alpha=0.8)
    ax2.title.set_text('Positive Pearson (N=%s)'%n_positive)
    ax2.set_xlabel('Offset (s)')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    ax3 = plt.subplot(1, 3, 3, sharey=ax1)
    ax3.plot(time, mean_negative, color='r', label='Mean')
    ax3.plot(time, mean_negative+sem_negative, '--', color='blue', alpha=0.8, label='sem')
    ax3.plot(time, mean_negative-sem_negative, '--', color='blue', alpha=0.8)
    ax3.title.set_text('Negative Pearson (N=%s)'%n_negative)
    ax3.set_xlabel('Offset (s)')
    ax3.set_ylabel('Correlation Coefficient')
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    
    plt.savefig('{}/Combined_CC_{}'.format(save_location, name))
    plt.savefig('{}/Combined_CC_{}.pdf'.format(save_location, name))
    plt.close()

    return dic


def CC_calculate(dF, speed, settings, data_type):
    
    """
    Calculates Cross correlation using the function 'crosscorrel' from Yann Zerlaut.
    
    Input:
        dF: array of shape (x,y)
        speed: array of shape (1,y).
        Settings: dic containing the following keys:    
            tmax: maximum time offset (in seconds)
            dt: sampling rate
        data_type: 'locomotion' or 'whisking'
    
    Output:
        CC: array of shape (x,z) containing the cross correlations of dF with speed
        shift: array of shape (1, z)
        lag: array of 
        
        
    """
    tmax = settings['CC shift (s)']
    dt = float(1/settings['fs'])
    save_location = settings['save_path']
    
    CC_array = np.zeros((dF.shape[0], int(tmax/dt)*2-1))
    lag = np.zeros((dF.shape[0], 1))
    for i in range(dF.shape[0]):
        CC_array[i], CC_shift = crosscorrel(dF[i], speed, tmax, dt)
        lag[i] = CC_shift[np.argmax(CC_array[i])]
    
    mean = np.mean(CC_array, axis=0)
    std = np.std(CC_array, axis=0)
    
    plt.figure()
    for i in range(dF.shape[0]):
        plt.plot(CC_shift, CC_array[i], color='grey', alpha=0.5)
    plt.plot(CC_shift, mean+std, '--', color='blue', alpha=0.8, label='std')
    plt.plot(CC_shift, mean-std, '--', color='blue', alpha=0.8)
    plt.plot(CC_shift, mean, color='black', label='Mean')
    plt.legend()
    plt.xlabel('Offset (s)')
    plt.ylabel('Correlation Coefficient')
    plt.savefig('{}/Cross_Correlation_{}'.format(save_location, data_type))
    plt.savefig('{}/Cross_Correlation_{}.pdf'.format(save_location, data_type))
    plt.close()
    
    return CC_array, CC_shift, lag

 
    
