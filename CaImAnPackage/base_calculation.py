import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy.signal import savgol_filter


def load_red_cells(file_path, settings):
    
    # Import data
    F_all = np.load(os.path.join(file_path, 'F.npy'))
    Fneu_all = np.load(os.path.join(file_path, 'Fneu.npy'))
    iscell = np.load(os.path.join(file_path, 'iscell.npy'))
    spikes_all = np.load(os.path.join(file_path, 'spks.npy'))
    try:
        red_cell = np.load(os.path.join(file_path, 'redcell.npy'))
        print('-- Red cell channel found')
    except:
        print('-- No red channel present')
    
    # Select data
    if 'red_cell' in locals():
        if settings['red_cells']['cell_type'] == 'Green+Red':
            index = [i for i, (cell, red) in enumerate(zip(iscell, red_cell)) if (cell[0]==1 and red[0]==1)]
            F = np.zeros((len(index), np.size(F_all, 1)))
            Fneu = np.zeros((len(index), np.size(F_all, 1)))
            spikes = np.zeros((len(index), np.size(F_all, 1)))
            print('-- Selecting Green and red cells')
        elif settings['red_cells']['cell_type'] == 'Green':
            index = [i for i, (cell, red) in enumerate(zip(iscell, red_cell)) if (cell[0]==1 and red[0]==0)]
            F = np.zeros((len(index), np.size(F_all, 1)))
            Fneu = np.zeros((len(index), np.size(F_all, 1)))
            spikes = np.zeros((len(index), np.size(F_all, 1)))
            print('-- Selecting Green only cells')
    else:
        index = [i for i, x in enumerate(iscell) if x[0] == 1]
        F = np.zeros((len(index), np.size(F_all, 1)))
        Fneu = np.zeros((len(index), np.size(F_all, 1)))
        spikes = np.zeros((len(index), np.size(F_all, 1)))
        print('-- Selecting all cells')
    
    for i, x in enumerate(index):
        F[i] = F_all[x]
        Fneu[i] = Fneu_all[x]
        spikes[i] = spikes_all[x]
    
    return F, Fneu, spikes, index

def load_data(file_path):
    '''Loads output of Suite2P and selects cells
    
    Parameters:
        file_path(string): string of the path to the folder containing
            example: '.../suite2p/plane0'  
    Returns:
        F: numpy array with fluorescence of selected cells
        Fneu: numpy array with fluorescence of neuropil
        spikes: numpy array with deconvolved spikes
        cells: numpy array with the number of the selected cells
        
    '''
    F_all = np.load('F.npy')
    Fneu_all = np.load('Fneu.npy')
    iscell = np.load('iscell.npy')
    spikes_all = np.load('spks.npy')
    N_cells = np.count_nonzero(iscell[:,0] == 1)
    cells = [i for i, x in enumerate(iscell[:,0]) if x]
    F = np.zeros((N_cells, np.size(F_all, 1)))
    Fneu = np.zeros((N_cells, np.size(F_all, 1)))
    spikes = np.zeros((N_cells, np.size(F_all, 1)))
    y = 0
    for i in range(0, np.size(iscell, 0)):
        if iscell[i,0] == 1:
            F[y] = F_all[i]
            Fneu[y] = Fneu_all[i]
            spikes[y] = spikes_all[i]
            y += 1
    return F, Fneu, spikes, cells

def load_specific_data(file_path, chosen_cells):
    '''Loads output of Suite2P and selects cells
    
    Parameters:
        file_path(string): string of the path to the folder containing
            example: '.../suite2p/plane0'  
    Returns:
        F: numpy array with fluorescence of selected cells
        Fneu: numpy array with fluorescence of neuropil
        spikes: numpy array with deconvolved spikes
        cells: numpy array with the number of the selected cells
        
    '''
    F_all = np.load('F.npy')
    Fneu_all = np.load('Fneu.npy')
    #iscell = np.load('iscell.npy')
    spikes_all = np.load('spks.npy')
    N_cells = len(chosen_cells)
    cells = chosen_cells
    F = np.zeros((N_cells, np.size(F_all, 1)))
    Fneu = np.zeros((N_cells, np.size(F_all, 1)))
    spikes = np.zeros((N_cells, np.size(F_all, 1)))
    y = 0
    for i in cells:
        F[y] = F_all[i]
        Fneu[y] = Fneu_all[i]
        spikes[y] = spikes_all[i]
        y += 1
    return F, Fneu, spikes, cells

def cell_select(list_files, df, cell_group='Green'):
    df1 = df[df['Mouse'].notna()]
    mouse = []
    date = []
    experiment = []
    for file in list_files:
        split_file = file.split('\\')
        mouse.append(split_file[-2])
        date.append(split_file[-3])
        experiment.append(int(split_file[-1].split('-')[-1]))
    # Find matching experiments
    red_cells = []
    green_cells = []
    list_files_g = []
    list_files_r = []
    for m, d, e, file in zip(mouse, date, experiment, list_files):
        i = df1[(df1['Mouse'] == m) & (df['Day'] == d) & (df['Trial'] == e)].index.tolist()
        if not i:
            list_files_g.append(str(file))
            red_cells.append([])
            list_files_r.append(str(file))
            green_cells.append([])
        elif len(i) > 1:
            print('More than one file found')
        else:
            if not pd.isna(df.loc[i[0],'N red cells']):
                red_cells.append(list(map(int, str(df1.loc[i[0],'N red cells']).split('.'))))
                list_files_r.append(str(file))
                
            if not pd.isna(df.loc[i[0],'N other cells']):
                green_cells.append(list(map(int, str(df1.loc[i[0],'N other cells']).split('.'))))
                list_files_g.append(str(file))
    # Output
    if cell_group == 'Green':    
        return list_files_g, green_cells
    elif cell_group == 'Green + Red':
        return list_files_r, red_cells
    else:
        print('Wrong color chosen')


# Isolate cells -- Not used
def cell_select1(F_all, Fneu_all, iscell):
    N_cells = np.count_nonzero(iscell[:,0] == 1)
    N_samples = np.size(F_all, 1)
    F = np.zeros((N_cells, N_samples))
    Fneu = np.zeros((N_cells, N_samples))
    y = 0
    for i in range(0, np.size(iscell, 0)):
        if iscell[i,0] == 1:
            F[y] = F_all[i]
            Fneu[y] = Fneu_all[i]
            y += 1
    return(F, Fneu)
    
def hamming_filter(F_roi, fs, window_length):
    '''Performs a hamming filter on a single trace
    
    Parameters:
        F_roi: one-dimentional array of fluorescence
        fs: sampling frequency
        window_length: length of filter window in seconds
    Returns:
        F_roi_smoothed: smoothed array
        
    '''
    hamming_window = np.hamming(int(window_length*fs))
    F_roi_smoothed = np.convolve(F_roi, hamming_window/hamming_window.sum(), mode='same')
    return F_roi_smoothed    

def percentile_extract(data, percentile):
    '''Calculates the average fluorescence below a certain percentile
    
    Parameters:
        data: 2-dimentional array of fluorescence
        percentile: value of percentage    
    Returns:
        roi_percentile_values: array with values of fluorescence below percentile
        
    '''
    N_roi = np.size(data, 0)
    N_samples = np.size(data, 1)
    roi_percentile = np.percentile(data, percentile, 1)
    roi_percentile_values = np.zeros((N_roi,(1+int(percentile*0.01*N_samples))))
    for i in range(0, N_roi):
        roi_percentile_values[i] = np.extract(data[i] <= roi_percentile[i], data[i])    
    return roi_percentile_values

def sliding_window(F, fs, sig= 10, win = 60):
    '''Sliding window using gaussian filter
    
    Parameters:
        F: array with fluorescence values
        fs: sampling rate
        sig: smoothing constant for gaussian filter
        win: size of window for smoothing
    Returns:
        F: baseline subtracted fluorescence traces
        
    '''
    Flow = filters.gaussian_filter(F, [0., sig])
    Flow = filters.minimum_filter1d(Flow,    win * fs, mode='wrap')
    Flow = filters.maximum_filter1d(Flow,    win * fs, mode='wrap')
    #F = F - Flow
    return F, Flow

# Calculate F0 selected area
def F0_selection(F, time_sec, fs):
    '''Define a section of the trace 
    
    Parameters:
        F: array with fluorescence values
        time_sec: total length of recording in seconds
        fs: sampling rate
    Returns:
        positions: start and end position of selected area in seconds
        positions_absolute: start and end position of selected area
        
    '''
    t = np.linspace(0,time_sec,np.size(F, 1))
    positions = np.zeros((np.size(F,0),2))
    for i in range(len(F)):
        plt.figure()
        plt.subplot(1,1,1)
        plt.plot(t, F[i], 'k-', label='ROI {}'.format(i))
        plt.title(label='Select period to calculate F0 (2 clicks)')
        plt.legend()
        points = plt.ginput(n=2, show_clicks=True, timeout=0)
        plt.close()
        positions[i] = [x[0] for x in points]
    positions_absolute = positions * fs
    positions_absolute = positions_absolute.astype(int)
    np.save('positions.npy', positions)
    np.save('positions_absolute.npy', positions_absolute)
    return positions, positions_absolute
    
def Fsmooth_calculation(F, fs, window_length):
    hamming_window = np.hamming(int(window_length*fs))
    F_smooth = np.copy(F)
    for i in range(0, np.size(F, 0)):
        F_smooth[i]=np.convolve(F[i], hamming_window/hamming_window.sum(), mode='same')
    return F_smooth
    
def F0_calculation(F, fs, time_sec, window_length, percentile, positions_absolute):
    F_smooth = Fsmooth_calculation(F, fs, window_length)
    F0 = np.zeros((np.size(F,0), 1))
    for i in range(len(F_smooth)):
        x1 = positions_absolute[i,0]
        x2 = positions_absolute[i,1]
        F0_selected = F_smooth[i,x1:x2]
        roi_percentile = np.percentile(F0_selected, percentile, 0)
        roi_percentile_values = np.extract(F0_selected <= roi_percentile, F0_selected)
        f0 = np.mean(roi_percentile_values)
        F0[i] = f0
    return F0, F_smooth

def full_trace(F, Settings):
    d1 = np.zeros(shape=(F.shape[0], 1))
    d2 = np.zeros(shape=(F.shape[0], 1))
    d1.fill(0)
    d2.fill(F.shape[1])   
    positions_absolute = np.concatenate((d1,d2), axis=1)
    positions_absolute = positions_absolute.astype(int)
    positions = positions_absolute / Settings['fs']
    return positions, positions_absolute

# Calculate dF/F0
def deltaF_calculate(F, F0):
    normalized_F = np.copy(F)
    for i in range(0, np.size(F, 0)):
        normalized_F[i] = (F[i]-F0[i])/F0[i]
    return normalized_F


# Calculate Average Fluorescence
def average_fluo(dF, cells):
    average_dF = np.mean(dF, axis=1)
    cells = np.array(cells)
    cells = cells.reshape((len(cells),1))
    average_dF = average_dF.reshape((len(average_dF),1))
    #average_dF_new = np.concatenate([cells, average_dF], axis=1)
    #average_dF_new = pd.DataFrame(average_dF_new)
    #np.savetxt('Average_dF.csv', average_dF_new, delimiter=',')
    #average_dF_new.to_excel('Average_dF.xlsx', index=False)
    return average_dF


def sort_data(dF, move, nomove):
    active = np.zeros((len(dF),1))
    inactive = np.zeros((len(dF),1))
    for i, v in enumerate(move):
        if v == 1:
            active = np.append(active, np.reshape(dF[:, i],(len(dF),1)), axis=1)
    for i, v in enumerate(nomove):
        if v == 0:
            inactive = np.append(inactive, np.reshape(dF[:, i],(len(dF),1)), axis=1)
    active = np.delete(active, 0, 1)
    inactive = np.delete(inactive, 0, 1)
    active_mean = np.mean(active,1)
    inactive_mean = np.mean(inactive,1)
    return active_mean, inactive_mean


def get_run_rest(data, movement_active, movement_inactive):
    
    # reshape the data if there is only one ROI
    one_ROI = len(data.shape)
    if one_ROI == 1 :
        data = data.reshape((1, len(data)))

    NbOfROI, _ = data.shape

    neuronal_run = np.zeros((NbOfROI,np.count_nonzero(movement_active)))
    neuronal_rest = np.zeros((NbOfROI,np.count_nonzero(movement_inactive<1)))

    for roi in range(NbOfROI):
        neuronal_run[roi,:] = [data[roi,i] for i in np.nonzero(movement_active)[0]]
        neuronal_rest[roi,:] = [data[roi,i] for i in np.nonzero(movement_inactive<1)[0]]
        
    return neuronal_run, neuronal_rest


def dF_std(dF, Settings):
    #method: sumbre
    """
    ...........................................................................
    This function calculates thethreshold of dF/F0 based on the method developped 
    by Sumbre to compute F0. Fit the Gaussian of the signal only on negative values
    of the dF/F to compute the threshold (3*std of the nag values of the trace)
    Rq: for up-state cells, the noise is computed on the positive values of the 
        trace. 
    ...........................................................................
    Inputs
    ------------
    dF          dF of one ROI
    Settings    uses 'state' to get up-state cells
    
    Output
    ------------
    th_dF       Threshold of the trace (3*std)

    """
    state=Settings['state']
    th_dF = []
    f_inf=np.array(np.shape(dF)[1])
    for roi in range(len(dF)):
        if state[roi]==-1:
            f_bin = (dF[roi] > 0)*1 
        else:
            f_bin = (dF[roi] < 0)*1 
        f_inf=dF[roi][f_bin==1]
        f_tot=np.concatenate((f_inf,f_inf*(-1)))
        th = 3*np.std(f_tot)
        th_dF.append(th)
    return(th_dF)
    

def active_silent(dF, th_dF, Settings, window_length=2.5):
    """
    ...........................................................................
    This function assigns 1 or 0 or -1 to active / silent cells / up-state cells
    (if up-states already detected by the function state_detection, state is re-
    evaluated to -1 or 0) using dF/F0 and the thresholds (computed with Sumbre method). 
    A cell is considered as silent if:
    - 1% of the filtered trace (savgold) is above or below the threshold 
    - some points of the filtered trace (hamming)  are above or below the threshold 
    ...........................................................................
    Methods
    -------------
    'savgold'            
    'hamming'   
     
    Inputs
    -------------
    dF              dF of several ROIs
    th_dF           list of thresholds 
    window_length   (seconds) only used in Hamming filter
    Settings        use of the sampling frequency, time_sec
                    use of 'state' (optional, if Settings['SST']['isSST']=True)
       
    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 
    active         list of state of each ROI
    PER            list of percentages of the trace out of bounds (savgol)
                   list of 1 or 0 if the trace is out of bounds or not (hamming)
    filt_trace      array, filtered trace
    ...........................................................................
    """
    fs=Settings['fs']
    filt=Settings['filt_active_cells']
    active=[0]*len(dF)
    PER=[]
    filt_trace=np.zeros((np.shape(dF)[0], np.shape(dF)[1]))
    if Settings['SST']['isSST']==True:
        active=Settings['state']

    
    for ROI in range(0,np.shape(dF)[0]):
        if filt=='savgol':
            dF_filt = savgol_filter(dF[ROI],29,2)
            inf= (dF_filt < -th_dF[ROI])*1
            dF_inf=dF[ROI][inf==1] #valeurs de dF au dessous du seuil, valeur extreme
            sup=(dF_filt>th_dF[ROI])*1
            dF_sup=dF_filt[sup==1] #valeurs de dF au dessus du seuil, valeur extreme    
            per_outside= (len(dF_inf)+len(dF_sup))/len(dF[ROI])*100

            if active[ROI]!=-1:
                if per_outside >= 2:
                    active[ROI]=1
                else:
                    active[ROI]=0
            else:
                pass
            
            if active[ROI]==-1:
                if per_outside >= 2:
                    active[ROI]=-1
                else:
                    active[ROI]=0
            else:
                pass
            
            PER.append(per_outside)
        elif filt == 'hamming':
            dF_hamm=hamming_filter(dF[ROI], fs, window_length)
            sup=(dF_hamm>th_dF[ROI])*1
            inf=(dF_hamm<-th_dF[ROI])*1
            F_sup=dF_hamm[sup==1]
            F_inf=dF_hamm[inf==1]
            if F_sup.size==0 and F_inf.size==0:
                PER.append(0)
                active[ROI]=0
            else:
                if active[ROI]!=-1:
                    active[ROI]=1
 
                elif active[ROI]!=-1:
                    active[ROI]=1
                PER.append(1)
        filt_trace[ROI]=dF_filt
       
    return( active, PER, filt_trace)

    

def state_detection(F, Settings):
    '''
    ...........................................................................
    Determines if F is mostly high with negative deflexions (low states, expected 
    for classic cells) or if F is mostly low with positive fluctuations (high-states).
    The distribution of Fraw (filtered) is skewed on the right for high-states, resp.
    on the left for low-states.
    This function assigns 1 or 0 to active or silent cells, and -1 to 'up-states'
    cells.
    ...........................................................................
    Inouts
    ------
    F                  array, raw trace 
     
    Settings    
       fs, win, sigma  uses parameters to compute the percentile of a trace
       per             int, fraction of the fluorescence amplitude that will be
                       compared to the same fraction of the distribution (percentile)
                       to determine ROI state. Determines how strict the detection will be.
    Outputs
    ------
    state              list of states (0, 1, -1).

    '''
    win= Settings['F0_settings']['PERCENTILE']['filter_window']
    fs=Settings['fs']
    per=Settings['SST']['per']
    state=[]
    
    for i in range(np.shape(F)[0]):
        F_filt = savgol_filter(F[i],29,2)
        per_inf = percentile(F_filt, per*100,win, fs, 1)[0]
        per_sup = percentile(F_filt, (1- per)*100, win, fs, -1)[0]
        amp=np.max(F_filt)-np.min(F_filt)
        low= np.min(F_filt)+amp*per
        up= np.max(F_filt)-amp*per
        #print(round(per_sup), up, round(per_inf), low)
        if per_sup > up :
            state.append(-1)
        elif per_inf < low:
            state.append(1)
            
        else:
            state.append(0)
    return state
    per=Settings['SST']['per']
    state=[]
    
    for i in range(np.shape(F)[0]):
        F_filt = savgol_filter(F[i],29,2)
        #exclude extreme values of the distribution 
        per_sup = np.percentile(F_filt, 99, 0)
        per_inf= np.percentile(F_filt, 0.1 , 0)
        amp=per_sup-per_inf
        
        per_act = np.percentile(F_filt, per*100,0)
        per_up = np.percentile(F_filt, (1- per)*100,0)
#        amp=np.max(F_filt)-np.min(F_filt)
#        low= np.min(F_filt)+amp*per
#        up= np.max(F_filt)-amp*per
        low= per_inf+amp*per
        up= per_sup-amp*per
        #print(round(per_sup), up, round(per_inf), low)
        if per_up > up :
            state.append(-1)
        elif per_act < low:
            state.append(1)
            
        else:
            state.append(0)
    
    
    return state

def percentile_state(trace, percentile, filter_window, fs, state):
    smooth = hamming_filter(trace, fs, filter_window)
    trace_perc = np.percentile(smooth, percentile, 0)
    if state == -1:
        trace_vals = np.extract(smooth >= 100-trace_perc, smooth)
    elif state == 1:
        trace_vals = np.extract(smooth <= trace_perc, smooth)
    F0 = np.array(np.shape(trace)[0] * [np.mean(trace_vals)])
    return F0



def percentile(trace, percentile, filter_window, fs, state):
    smooth = hamming_filter(trace, fs, filter_window)
    trace_perc = np.percentile(smooth, percentile, 0)
    trace_vals = np.extract(smooth <= trace_perc, smooth)
    F0 = np.array(np.shape(trace)[0] * [np.mean(trace_vals)])
    return F0



def minmax(trace, window, sigma, fs, state):
    if trace.ndim != 2:
        trace = np.expand_dims(trace, 0)
    F0 = filters.gaussian_filter(trace, [0., sigma])
    if state ==-1:
        F0 = filters.maximum_filter1d(F0,    window * fs, mode='wrap')
        F0 = filters.minimum_filter1d(F0,    window * fs, mode='wrap')
    else :
        F0 = filters.minimum_filter1d(F0,    window * fs, mode='wrap')
        F0 = filters.maximum_filter1d(F0,    window * fs, mode='wrap')
    return F0


def lowest_std(trace, window, std_window, filter_window, fs):
    smooth = hamming_filter(trace, fs, filter_window)
    F0 = np.zeros(len(smooth))
    n_extra = int((window*fs)/2)
    smooth_extend = list(np.flip(smooth[0:n_extra]))+list(smooth)+list(np.flip(smooth[-n_extra:]))
    for i in range(0,len(smooth)):
        mean = []
        std = []
        section = smooth_extend[i:((window * fs)+i)]
        for x in range(0,len(section)-(std_window * fs)-1):
            std_section = section[x:(std_window*fs)+x]
            local_std = np.std(std_section)
            local_mean = np.mean(std_section)
            mean.append(local_mean)
            std.append(local_std)
        F0[i] = mean[np.argmin(np.array(std))]
    return F0


def selected():
    
    pass


def calculate_F0(F, Settings):
    method = Settings['F0_method']
    f0_settings = Settings['F0_settings'][method] #method = minmax, percentile ,.. (see in Settings)
    F0 = np.zeros(np.shape(F))
    for i in range(0, np.size(F,0)):
        
        if method == 'PERCENTILE':
            F0_roi = percentile(F[i], f0_settings['percentile'], f0_settings['filter_window'], Settings['fs'], 1)
            F0[i] = F0_roi
            
        elif method == 'MINMAX':
            if Settings['SST']['isSST'] == True and Settings['state'][i] == -1:
                F0_roi = minmax(F[i], f0_settings['window'], f0_settings['sigma'], Settings['fs'], -1)
            else: 
                F0_roi = minmax(F[i], f0_settings['window'], f0_settings['sigma'], Settings['fs'], 1)
            F0[i] = F0_roi
            
        elif method == 'LOWEST STD':
            #F0[i] = F0_roi
            pass
        
        elif method == 'SELECTED':
            #F0[i] = F0_roi
            pass
            
    return F0



    

#def calculate_F0(F, fs, window_length, percentile):
#    '''Calculates F0
#    
#    Parameters:
#        F: array with fluorescence values
#        fs: sampling rate
#        window_length: filtering window length in seconds
#        percentile: percentage of data to take
#    Returns:
#        F_smooth: smoothed trace of F
#        roi_percentile: array with values of fluorescence below percentile
#        F0: average fluorescence of values below percentile
#        
#    '''
#    F_smooth = serial_smoother(F, fs, window_length)
#    roi_percentile = percentile_extract(F_smooth, percentile)
#    F0 = np.mean(roi_percentile, axis=1)
#    return F_smooth, roi_percentile, F0

#def serial_smoother(data, fs, window_length):
#    '''Performs a hamming filter on multiple traces
#    
#    Parameters:
#        data: 2-dimentional array of fluorescence
#        fs: sampling frequency
#        window_length: length of filter window in seconds     
#    Returns:
#        data_smoothed: smoothed array
#        
#    '''
#    data_smoothed = np.copy(data)
#    for i in range(0, np.size(data, 0)):
#        data_smoothed[i] = hamming_filter(data[i,:], fs, window_length)
#    return data_smoothed    

    
#def smooth(x,window_len=11,window='hanning'):
#    """smooth the data using a window with requested size.
#    
#    This method is based on the convolution of a scaled window with the signal.
#    The signal is prepared by introducing reflected copies of the signal 
#    (with the window size) in both ends so that transient parts are minimized
#    in the begining and end part of the output signal.
#    
#    input:
#        x: the input signal 
#        window_len: the dimension of the smoothing window; should be an odd integer
#        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
#            flat window will produce a moving average smoothing.
#
#    output:
#        the smoothed signal
#        
#    example:
#
#    t=linspace(-2,2,0.1)
#    x=sin(t)+randn(len(t))*0.1
#    y=smooth(x)
# 
#    TODO: the window parameter could be the window itself if an array instead of a string
#    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
#    """

#    if x.ndim != 1:
#        raise ValueError, "smooth only accepts 1 dimension arrays."
#
#    if x.size < window_len:
#        raise ValueError, "Input vector needs to be bigger than window size."


#    if window_len<3:
#        return x
#
#
##    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
##        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
#
#
#    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
#    #print(len(s))
#    if window == 'flat': #moving average
#        w=np.ones(window_len,'d')
#    else:
#        w=eval('numpy.'+window+'(window_len)')
#
#    y=np.convolve(w/w.sum(),s,mode='valid')
#    return y

# TODO
    
#def new_F0_calculation(array, fs, win= 60, win_std= 0.5):
#    n_frames_win = int(fs * win)
#    n_frames_win_std = int(fs * win_std)
#    n_win_std = int(round(n_frames_win / n_frames_win_std))
#    
#    for i in range(0, len(array) - n_frames_win):
#        section = array[i:(i + n_frames_win)]
#        local_min_std = []
#        local_min_avg = []
#        for n in range(1, n_win_std):
#            i_start = n * n_frames_win_std
#            i_end = i_start + n_frames_win_std
#            local_min_std.append(np.std(section[i_start: i_end]))
#            local_min_avg.append(np.average(section[i_start: i_end]))
#            
#def moving_F0(array, percentile, fs, win= 60):
#    array_smooth = hamming_filter(array, fs, 0.5)
#    n_frames_win = int(fs * win)
#    F0 = []
#
#    for i in range(0, len(array_smooth) - n_frames_win):
#        section = array_smooth[i:(i + n_frames_win)]
#        percentile_value = np.percentile(section, percentile, axis=0)
#        local_F0 = np.average( section[section <= percentile_value])
#        F0.append(local_F0)
#
#    return F0
#
#    
#def F0_calc_std(array, fs, win_filt = 0.5, win= 60, win_std= 5):
#    data = np.copy(array)
#    # Filter Data
#    hamming_window = np.hamming(int(win_filt*fs))
#    for i in range(0,len(data)):
#        data[i] = np.convolve(data[i], hamming_window/hamming_window.sum(), mode='valid')
#    
#    # Extend Data
#    
#    # Get
#    F0 = []
#    
#    for i in range(0, len(data) - (win * fs)):
#        section = data[i:(i + (win * fs))]
#        local_min_std = []
#        local_min_avg = []
#        for i in range
        
        
        
    
    