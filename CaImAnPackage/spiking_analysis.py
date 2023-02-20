# Functions for analysis of deconvolved data

import numpy as np
from tqdm import tqdm
from numba.typed import List

import compute_stats


def compute_baseline(trace, settings, duration_w = 10):

    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ............................................................................

    This function compute the baseline of a trace (fluorescence or spike) using
    a sliding window. The baseline is the segment of the trace with the lowest
    variance.

    ............................................................................

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    trace             trace (F, dF/F or spikes), a list with only one ROI
    settings          settings (use of the final "sampling frequency)
    duration_w        duration of the sliding window (10s by default)

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - -

    baseline          segment of the trace with the lowest variance
    index_baseline    index of the first element of the baseline in the trace

    ............................................................................
    """

    sampling_rate = settings['fs']

    sliding_window = duration_w * sampling_rate
    min_std = np.zeros(len(trace) - sliding_window)

    indice = 0
    while (sliding_window + indice) <= len(trace) - 1:
        trace_window = trace[indice:indice + sliding_window]
        min_std[indice] = np.std(trace_window)
        indice += 1

    index_baseline = np.argmin(min_std)
    baseline = trace[index_baseline:index_baseline + sliding_window]

    return baseline, index_baseline


### compute the threshold applied on the firing trace . . . . . . . . . . . . .

    #compute thresholds even for up-state cells
def sumbre_threshold(dF, S, state):
    
    """

    ............................................................................

    This function compute the threshold based on the method developped by 
    Sumbre. We fit the Gaussian only on negative value of the dF/F to compute 
    the threshold. If up-state cell : noise on the positive values of dF/F
    
        1. 3*std computed on  the negative values of dF/F
        2. percentage of dF/F represented by this 3*std
        3. apply this percentage on firing trace

    ............................................................................

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    dF                dF trace 
    S                 firing trace (spike) of only one ROI
    state               state of the ROI

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - -
    th                threshold computed with the method developped by Sumbre

    ............................................................................
    """
    f_inf=np.array(np.shape(dF))
    if state==-1:
        f_bin = (dF > 0)*1 
    else:
        f_bin = (dF < 0)*1 
    f_inf=dF[f_bin==1]
    f_tot=np.concatenate((f_inf,f_inf*(-1)))
    th = 3*np.std(f_tot) * (max(S)-min(S)) / (max(dF)-min(dF))
    return(th)
    
    
    
    
def threshold(S, dF, state, settings, method = 'constant', constant = 0, 
              duration_w_min = 5, duration_w_max = 10, per_error = 0.2):

    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................

    This function compute the threshold we can applied on the deconcolved trace
    (after suite2P OASIS deconvolution) to get the most meaningfull events.

    ...........................................................................

    - - - - - - - - - - - - - - - METHODS - - - - - - - - - - - - - - - - - - -

    'constant'        constant threshold (by default with constant = 0)
    'std'             threshold computed on the standard deviation of the firing
                      trace. The threshold is 3 times the standard deviation.
                      https://github.com/cortex-lab/Suite2P/issues/157
    '10_lowest'       threshold computed on the baseline of the firing trace
                      (10 lowest %)
    'baseline'        threshold computed on the baseline of the firing trace
                      (sliding window). This method calls the function
                      compute_baseline()
    'sumbre'          threshold computed using the method developped by Sumbre 
                      on the dF/F and then applied on the firing trace. 
                      1. 3*std computed on  the negative values of dF/F
                      2. percentage of dF/F represented by this 3*std
                      3. apply this percentage on firing trace.

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    S                 firing trace (spike) of only one ROI
    dF                dF trace 
    method            (constant at 0 by default)
    constant          (only usefull if method = 'constant')
    duration_w        duration of the sliding window (method = 'baseline')

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    th                the computed threshold

    ...........................................................................
    """

    # constant threshold
    if method == 'constant':
        th = 0

    # threshold computed on the standard deviation of the firing trace
    if method == 'std':
        th = 3 * np.std(S)

    # threshold computed on the baseline of the firing trace (10 lowest %)
    if method == '10_lowest':
        percentage = 0.1
        only_spikes = S[np.nonzero(S)]
        only_spikes.sort()
        th = np.mean(S[:int(len(only_spikes)*percentage)]) + 3*np.std(S[:int(len(only_spikes)*percentage)])

    # threshold computed on the baseline of the firing trace (sliding window)
    if method == 'baseline':
        baseline, _ = compute_baseline(S, settings, duration_w_min)
        th_min = 2 * max(baseline)
        baseline, _ = compute_baseline(S, settings, duration_w_max)
        th_max = 2 * max(baseline)
        
        if int(np.maximum(th_min, th_max)) == 0:
            th = 0
        elif abs((th_max - th_min)/np.maximum(th_min, th_max)) < per_error:
            th = th_max
        else:
            th = th_min
    
    if method == 'sumbre':
        th = sumbre_threshold(dF, S, state)

    return th


def randomizer(S):
    random_S = np.copy(S)
    for i in range(0,len(random_S)):
        np.random.shuffle(random_S[i])
    return random_S


def get_average_rand_synchronicity(S, Settings, N_iterations, thr_spikes, method='STTC', w_size=10):

    S_shuffled = randomizer(S)
    matrix_shuffled = np.expand_dims(compute_stats.synchrony(S_shuffled, Settings, thr_spikes, method, w_size), 2)
    
    for n in tqdm(range(0, N_iterations-1)):
        matrix = np.expand_dims(compute_stats.synchrony(randomizer(S), Settings, thr_spikes, method, w_size), 2)
        matrix_shuffled = np.dstack((matrix_shuffled, matrix))
        
    avg_rand_synchro = np.mean(matrix_shuffled, axis=2)
    
    return avg_rand_synchro


def combined(spikes, dF, settings, state_cells):
    amplitude = np.sum(spikes, axis = 1)*settings['fs']/spikes.shape[1]
    Nb_of_events = np.zeros(spikes.shape[0])
    spikes_binary = np.zeros(spikes.shape)
    thr_spikes = List()
    for roi in range(0, len(spikes)):
        thr = threshold(spikes[roi], dF[roi], state_cells[roi], settings,
                        method=settings['threshold_method'],
                        duration_w_min=5, duration_w_max=10)
        thr_spikes.append(thr)
        Nb_of_events[roi] = np.count_nonzero(spikes[roi]>thr)*settings['fs']/spikes.shape[1]
        spikes_binary[roi] = (spikes[roi]>thr)*1
    
    return amplitude, Nb_of_events, spikes_binary, thr_spikes

# TODO
#def synchro_n_times(S, N, Settings, thr, method='STTC', w_size=10 ):
#    number = 0
#    S_rand = randomizer(S)                                                    
#    matrix_synchro_rand = compute_stats.synchrony(S_rand, Settings, thr, 
#                                                  method, w_size)
#    while number < N:
#        S_rand = randomizer(S)
#        m_synchro_rand = compute_stats.synchrony(S_rand, Settings, thr, 
#                                                      method, w_size)                 
#        matrix_synchro_rand = np.average()
#        number += 1
#        
         