#import brewer2mpl
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import skew
from scipy import signal
from sklearn.cluster import KMeans
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable

import spiking_analysis, base_calculation



# Plot Data
def plot_F0(F, F0, ROI, time_sec, percentile):
    t = np.linspace(0,time_sec,np.size(F, 1))
    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(1,1,1)
    ax.plot(t, F[ROI], 'b-', label='Raw Trace')
    #ax.plot(t, F_smooth[ROI], 'y-', label='Smoothed Trace')
    ax.hlines(F0[ROI], 0, time_sec, colors = 'r', label = 'F0 ({}th percentile)'.format(percentile))
    ax.set(xlabel = 'Time')
    ax.legend()
    plt.show()
    

def plot_dF(dF, time_sec, offset, filename, colormap='plasma', show = True):
    plt.ioff()
    t = np.linspace(0,time_sec,np.size(dF, 1))
    color_map = plt.get_cmap(colormap)
    plt.figure()
#    plt.subplot(1,1,1)
    for i in range(len(dF)):
        trace = dF[i] + (i*offset)
        c = color_map(float(i)/len(dF))
        plt.plot(t, trace, color=c, label = 'ROI-{}'.format(i))
        plt.xlabel('Time (s)')
        plt.legend(loc=5, prop={'size': 6})
    plt.title(label='dF/F0 Traces')
    plt.savefig('{}/figure_dF.pdf'.format(filename))
    plt.savefig('{}/figure_dF'.format(filename))
    
    if show == True:
        plt.show()
    else:
        plt.close()
    

def plot_F_ROI(F_raw, Fneu_raw, F, F0, dF, ROI, th_dF, per_outside, dF_filt, Settings, show = True):
    
    ''' 
    ...........................................................................
    Plot
    - Fraw and F0
    - dF/F0 with the thresholds computed with the Sumbre method
    - dF/F0 filtered (savgold or hamming filters)
    Display if the cell is active or silent for each method)
    - skewness of dF/F0
    - - - - - - - - - - - - - - - METHODS - - - - - - - - - - - - - - - - - - -
    'minmax'
    'percentile'
    --
    'hamming'   Hamming filter, cells with points of the trace above or below 
                thresholds are considered active.
    'savgold'   Savgold filter, Cells with less than 2% of dF/F0 above and 
                below threshold are considered silent (2% value is arbitraty)
    -  - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -
    dF                one or several ROI
    F0                baseline
    th_dF             threshold computed with Sumbre method
    window_length     used for Hamming filtering
    Settings          use of sampling frequency, time_sec, state
                        
    '''
    time_sec=Settings['time_seconds']
    percentile=Settings['F0_settings']['PERCENTILE']['percentile']
    method = Settings['F0_method']
    filt= Settings['filt_active_cells']
    state_cells= Settings['state']
    
    plt.ioff()
    t = np.linspace(0,time_sec,np.size(F, 1))
    skewness = skew(dF[ROI])
    plt.figure(figsize=(8,8))
    
    plt.subplot(5,1,1)
    plt.plot(t, F_raw[ROI], 'k-', linewidth=0.5, label='F raw')
    plt.plot(t, Fneu_raw[ROI], 'r-', linewidth=0.5, label='Fneuropil', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.legend(loc=1, prop={'size': 6})
    plt.title(label='ROI {}'.format(ROI))
    
    plt.subplot(5,1,2)
    plt.plot(t, F[ROI], 'b-', linewidth=0.5, label='F corrected')
    if method== 'PERCENTILE':
        plt.hlines(F0[ROI], 0, time_sec, colors = 'r', label = 'F0 ({}th percentile)'.format(percentile))
    else:
        plt.plot(t, F0[ROI], 'r', linewidth=2, label = 'F0')
    plt.xlabel('Time (s)')
    plt.legend(loc=1, prop={'size': 6})
    plt.title(label='ROI {}'.format(ROI))
    
    plt.subplot(5,1,3)
    plt.plot(t,dF[ROI], 'b-', linewidth=0.5, label='dF/F0')
    plt.xlabel('Time (s)')
    plt.hlines(th_dF[ROI],0, time_sec,color='slategrey', linewidth=1, linestyle='--', label='3*std={}'.format(round(th_dF[ROI],2)))
    plt.hlines(- th_dF[ROI],0, time_sec, color='slategrey', linewidth=1, linestyle='--')    
    plt.legend(loc=1, prop={'size': 6})
    
    plt.subplot(5,1,4)
    state='------'
    plt.plot(t,dF_filt[ROI], 'black', linewidth=0.5, label='dF/F0 filtered ({})'.format(filt))
    plt.hlines(th_dF[ROI],0,time_sec, color='slategrey', linewidth=1, linestyle='--', label='3*std={}'.format(round(th_dF[ROI],2)))
    plt.hlines( -th_dF[ROI],0, time_sec, color='slategrey', linewidth=1, linestyle='--')
    plt.xlabel('Time (s)')
    plt.legend(loc=1, prop={'size': 6})
    
    if per_outside[ROI] >= 2 and state_cells[ROI]==1: 
        state='Active'
    elif per_outside[ROI] < 2 and state_cells[ROI]==0:
        state='Silent'
    elif state_cells[ROI]==-1:
        state='Up-state'
    plt.title('{} cell, {}% of dF/F above threshold'.format(state, round(per_outside[ROI],2)))

            
    plt.subplot(5,1,5)
    plt.hist(dF[ROI], bins= 50)
    plt.xlabel('dF/F0')
    plt.ylabel('N')
    plt.title(label='Histogram (Skewness={})'.format(skewness))
    plt.tight_layout()
    plt.savefig('{0}/figure_ROI_{1}'.format(Settings['save_path'],ROI))
            
    if show == True:
        plt.show()
    else:
        plt.close()
        
  
def plot_F_ROI_threshold(F, F0, F0per, dF, dFper, ROI, th_dF_minmax, th_dF_per, Settings, show = True):
    ''' 
    ...........................................................................
    Plot 
    - Fraw and F0 computed with minmax and percentile methods
    - dF/F0 thresholded with the minmax method
    - dF/F0 thresholded with the percentile method
    ...........................................................................

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    F                 one ROI
    F0, F0per         baselines computed with minmax or percentile methods
    dF, dFper         dF/F0 computed with minmax or percentile methods
    ROI
    th_dF_minmax, th_dF_per   thresholds for minmax and percentile methods
    Settings          use of percentile, time_sec 

    '''
    time_sec=Settings['time_seconds']
    percentile=Settings['percentile']
    
    plt.ioff()
    t = np.linspace(0,time_sec,np.size(F, 1))
    plt.figure()
    
    plt.subplot(3,1,1)
    plt.plot(t, F[ROI], 'black', linewidth=0.5, label='Raw Trace')
    plt.plot(t, F0[ROI], 'r', linewidth=2, label = 'F0') #minmax
    plt.hlines(F0per[ROI], 0, time_sec, colors = 'goldenrod', label = 'F0 ({}th percentile)'.format(percentile)) #percentile
    plt.xlabel('Time (s)')
    plt.legend(loc=1, prop={'size': 6})
    plt.title(label='ROI {}'.format(ROI))
        
    #minmax
    plt.subplot(3,1,2)
    plt.plot(t,dF[ROI], 'black', linewidth=0.5, label='dF/F0 minmax')
    plt.hlines(th_dF_minmax[ROI],0, time_sec, color='steelblue', linewidth=1, label='3*std={}'.format(round(th_dF_minmax[ROI],2)))
    plt.hlines(- th_dF_minmax[ROI],0, time_sec, color='steelblue', linewidth=1)
    plt.xlabel('Time (s)')
    plt.legend(loc=1, prop={'size': 6})
    

    #percentile
    plt.subplot(3,1,3)
    plt.plot(t,dFper[ROI], 'black', linewidth=0.5, label='dF/F0 percentile')
    plt.hlines(th_dF_per[ROI],0,time_sec, color='goldenrod', linewidth=2, linestyle = '--', label='3*std={}'.format(round(th_dF_per[ROI],2)))
    plt.hlines( -th_dF_per[ROI],0, time_sec, color='goldenrod', linewidth=2, linestyle = '--')
    plt.xlabel('Time (s)')
    plt.legend(loc=1, prop={'size': 6})
       
    plt.tight_layout()
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_trace_filtering(F,dF, F0, th_dF,Settings,window_length=2.5, show=True):

    ''' 
    ...........................................................................
    Plot
    - Fraw and F0
    - dF/F0 filtered with savgold
    - dF/F0 filtered with hamming
    Display if the cell is active or silent for each method)
    ...........................................................................

    - - - - - - - - - - - - - - - METHODS - - - - - - - - - - - - - - - - - - -
    'hamming'   Hamming filter
    'savgold'   Savgold filter, Cells with less than 2% of dF/F0 above and 
                below threshold are considered silent (2% value is arbitraty)

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    dF                one or several ROI
    F0                baseline
    th_dF             threshold computed with Sumbre method
    window_length     used for Hamming filtering
    Settings          use of sampling frequency, time_sec 

    '''
    time_sec=Settings['time_seconds']
    fs=Settings['fs']
    t = np.linspace(0,time_sec,np.size(dF, 1))
    savepath= Settings['save_path']
    
    for ROI in range(0,np.shape(dF)[0]):
        plt.figure()
    
        plt.subplot(3,1,1)
        plt.plot(t, F[ROI], 'black', linewidth=0.5, label='Raw Trace')
        plt.plot(t, F0[ROI], 'r', linewidth=2, label = 'F0') #minmax
        plt.xlabel('Time (s)')
        plt.legend(loc=1, prop={'size': 6})
        plt.title(label='ROI {}'.format(ROI))
    
        plt.subplot(3,1,2)
        dF_filt = signal.savgol_filter(dF[ROI],29,2)
        inf= (dF_filt < -th_dF[ROI])*1
        dF_inf=dF[ROI][inf==1] #valeurs de dF au dessous du seuil, valeur extreme
        sup=(dF_filt>th_dF[ROI])*1
        dF_sup=dF_filt[sup==1] #valeurs de dF au dessus du seuil, valeur extreme    
        per_outside= (len(dF_inf)+len(dF_sup))/len(dF[ROI])*100
        if per_outside < 1:
            state='Silent'
        else:
            state='Active'  
        plt.plot(t,dF_filt, 'black', linewidth=0.5, label='dF/F0 filtered (Savgold)')
        plt.title('{} cell, {}% of dF/F above threshold'.format(state, round(per_outside,2)))
        plt.hlines(th_dF[ROI],0,time_sec, color='steelblue', linewidth=1)
        plt.hlines( -th_dF[ROI],0, time_sec, color='steelblue', linewidth=1)
        plt.xlabel('Time (s)')
        plt.legend(loc=1, prop={'size': 6})
        
        plt.subplot(3,1,3)
        dF_hamm=base_calculation.hamming_filter(dF[ROI], fs, window_length)
        sup=(dF_hamm>th_dF[ROI])*1
        inf=(dF_hamm<-th_dF[ROI])*1
        F_sup=dF_hamm[sup==1]
        F_inf=dF_hamm[inf==1]
        if F_sup.size==0 and F_inf.size==0:
            state='Silent'
        else:
            state='Active'
        plt.plot(t, dF_hamm , 'black', linewidth=0.5, label='filtered (Hamming {})'.format(window_length))
        plt.title('{} cell'.format(state))
        plt.hlines(th_dF[ROI],0,time_sec, color='steelblue', linewidth=1)
        plt.hlines( -th_dF[ROI],0, time_sec, color='steelblue', linewidth=1)
        plt.xlabel('Time (s)')
        plt.legend(loc=1, prop={'size': 6})
        
        plt.tight_layout()
        plt.savefig('{0}/figure_th_ROI_compare_smoothings_{1}'.format(savepath,ROI))
    if show == True:
        plt.show()
    else:
        plt.close()
        
    
def image_movement(movement, dF, path, show = True):
    plt.ioff()
    fig = plt.figure()
    grid = plt.GridSpec(5, 1, hspace= 1.5, figure=fig)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1,0])
    ax3 = fig.add_subplot(grid[2:5,0])
    ax3.set_xlabel('Time (s)')
    ax1.set_title('Speed (cm/s) - {}% movement'.format(int(movement['percentage']*100)), loc='center')
    ax2.set_title('Binary Movement', loc='center')
    ax3.set_title('dF/F0', loc='center')
    ax1.tick_params(axis='x',labelbottom=False)
    ax2.tick_params(axis='x',labelbottom=False)
    ax2.axes.get_yaxis().set_ticks([])
    ax3.axes.get_yaxis().set_ticks([])
    ax1.plot(movement['t'], movement['speed'])
    ax2.plot(movement['t'], movement['binary_movement'])
    ax2.plot(movement['t'], movement['extended_binary_movement'])
    for i in range(len(dF)):
        trace = dF[i] + i*int(np.amax(dF)+1)
        ax3.plot(movement['t'], trace, lw=1)
    fig.savefig('{}/figure_dF_movement.pdf'.format(path))
    fig.savefig('{}/figure_dF_movement'.format(path))

    if show == True:
        plt.show()
    else:
        plt.close()
    
    
### Speed vs neuronal activity . . . . . . . . . . . . . . . . . . . . . . . .

# In this script, we will plot the neuronal activity versus the speed. We will 
# bin neuronal activity as well as running speed at 5s resolution by default.

# by default, binned every 5s and compute the mean (and not the sum)

def plot_neuron_vs_speed(speed_bin, trace_bin, slope, intercept, r_sq, path, ncats = 10,
                         n_rows = 30, n_columns = 3, show = True):
    
    NbOfROI = trace_bin.shape[0]

    n_rows = min(ceil(NbOfROI/n_columns), n_rows)
    n_columns = min(NbOfROI, n_columns)
    
    axe_x_min, axe_x_max = -0.2, np.nanmax(speed_bin) + 0.3
    axe_y_min, axe_y_max = np.nanmin(trace_bin)-0.2, np.nanmax(trace_bin)+0.2
    
    # ------- speed -----------------------------------------------------------
    min_s, max_s = np.nanmin(speed_bin), np.nanmax(speed_bin)
    dcat = (max_s - min_s)/ncats
    s_cats = [np.nanmin(speed_bin)]+[(i+1)*dcat + min_s for i in range(ncats)]
    
    t_cats = np.zeros((NbOfROI, ncats))
    t_cats = np.zeros((NbOfROI, ncats))
    for i in range(ncats):
        for j in range(NbOfROI):
            t_cats[j,i] = np.mean([f[1] for f in enumerate(trace_bin[j]) if speed_bin[0,f[0]]>s_cats[i] and speed_bin[0,f[0]]<s_cats[i+1]])
    
    s_cats = np.asarray(s_cats[:-1])
    
    plt.ioff()
    fig = plt.figure(figsize=(20,10))
    for roi in range(NbOfROI):
        
        plt.subplot(n_rows, n_columns, roi+1)
        plt.scatter(speed_bin, trace_bin[roi], color='seagreen')
        
        t_cat = np.asarray(t_cats[roi])
        t_mask = np.isfinite(np.array(t_cats[roi]).astype(np.double))
        plt.plot(s_cats[t_mask], t_cat[t_mask], color='red')
        
        plt.xlim(axe_x_min, axe_x_max)
        plt.ylim(axe_y_min, axe_y_max)
        
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept[roi] + slope[roi] * x_vals
        plt.plot(x_vals, y_vals, '--')
        
        plt.xlabel('Speed')
        plt.ylabel('mean dF/F')
        
#        textstr = '\n'.join((
#                r'coefficient of determination =%.2f$' % (r_sq[roi]),
#                r'slope =%.2f$' % (slope[roi])))
#        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#        
#        plt.text(1, 1, textstr, fontsize=14,
#        verticalalignment='top', bbox=props)
        
    fig.suptitle('Speed vs dF/F')
    
    p = path.split('\\')       
    fig.suptitle('Neuronal activity vs speed - {} - {} - Tseries {}'.format(p[-4], p[-5], p[-3][-1]))
    fig.savefig('{}/neuron_vs_speed'.format(path)) 
        
    
    if show == True:
        fig.show()
    else:
        plt.close(fig)
        
    
def plot_run_vs_rest(coor_rest, coor_run, path, show = True):
    
    plt.ioff()
    # plot the coordinates in running and resting states
    fig = plt.figure(figsize=(10,5))
    
    axes_min, axes_max = -0.2, max(coor_rest + coor_run) + 0.3
    
    plt.scatter(coor_rest, coor_run)
    plt.plot([axes_min, axes_max], [axes_min, axes_max], '--', color='k', 
             linewidth=1)
    axes = plt.gca()
    axes.set_xlim(axes_min, axes_max)
    axes.set_ylim(axes_min, axes_max)
    plt.xlabel('Neuronal activity - Resting periods')
    plt.ylabel('Neuronal activity - Running periods')
    
    #p = path.split('\\')       
    #fig.suptitle('Neuronal acitivity in running vs resting state - {} - {} - Tseries {}'.format(p[-4], p[-5], p[-3][-1]))
    fig.savefig('{}/run_vs_rest'.format(path))
        

    if show == True:
        fig.show()
    else:
        plt.close(fig)
    

def compare_lmi(lmi_f, lmi_a, lmi_s, path, show = True):
    plt.ioff()
    axes_min = min(lmi_a + lmi_f + lmi_s) - 0.1
    axes_max = max(lmi_a + lmi_f + lmi_s) + 0.1
    
    fig = plt.figure(figsize=(20,10))
    
    ax1 = fig.add_subplot(1, 3, 1)
    plt.scatter(lmi_f, lmi_a)
    plt.grid(linestyle='dotted')
    ax1.set_xlabel('LMI dF/F')
    ax1.set_ylabel('LMI Amplitude')
    ax1.set_xlim(axes_min, axes_max)
    ax1.set_ylim(axes_min, axes_max)
    
    ax2 = fig.add_subplot(1, 3, 2)
    plt.scatter(lmi_f, lmi_s)
    plt.grid(linestyle='dotted')
    ax2.set_xlabel('LMI dF/F')
    ax2.set_ylabel('LMI spike')
    ax2.set_xlim(axes_min, axes_max)
    ax2.set_ylim(axes_min, axes_max)
    
    ax3 = fig.add_subplot(1, 3, 3)
    plt.scatter(lmi_s, lmi_a)
    plt.grid(linestyle='dotted')
    ax3.set_xlabel('LMI spike')
    ax3.set_ylabel('LMI Amplitude')
    ax3.set_xlim(axes_min, axes_max)
    ax3.set_ylim(axes_min, axes_max)
    
    p = path.split('\\')       
    fig.suptitle('LMI comparaison - {} - {} - Tseries {}'.format(p[-4], p[-5], p[-3][-1]))
    fig.savefig('{}/comparaison_lmi'.format(path))
        
    if show == True:
        fig.show()
    else:
        plt.close(fig)
    
    
def plot_pmi(PMI_all, coor_rest, coor_run, mod_index_pmi, path, 
             nb_of_clusters = 2, show = True):
    
    plt.ioff()
    NbOfROI = len(mod_index_pmi)
    
    fig1 = plt.figure(figsize=(10,20))

    for i in range(NbOfROI):
        ax = plt.subplot(ceil(NbOfROI/3), min(3, NbOfROI), i+1)
        plt.imshow(PMI_all[i,:,:])
        plt.colorbar()
        ax.xaxis.set_ticklabels(['', 'rest', 'run'], rotation = 45) 
    
    p = path.split('\\')        
    fig1.suptitle('PMI - {} - {} - Tseries {}'.format(p[-4], p[-5], p[-3][-1]))
    fig1.savefig('{}/pmi_matrices.pdf'.format(path))
    
    if show == True:
        fig1.show()
    else:
        plt.close(fig1)
        
        
    # plot the coordinates in running and resting states
    fig2 = plt.figure(figsize=(10,5))
        
    # plot the ROIs according to their coordinates in running and resting states
    plt.subplot(1,2,1)
    # set color map without kmeans
    my_c = [0 for _ in range(NbOfROI)]
    for i in range(NbOfROI):
        if mod_index_pmi[i] > 0.9:
            my_c[i] = 'r'
        elif mod_index_pmi[i] < - 0.9:
            my_c[i] = 'g'
        else:
            my_c[i] = 'b'

    axes_min, axes_max = -0.2, max(coor_rest + coor_run) + 0.3
    
    plt.scatter(coor_rest, coor_run, c = my_c)
    plt.plot([axes_min, axes_max], [axes_min, axes_max], '--', color='k', 
             linewidth=1)
    axes = plt.gca()
    axes.set_xlim(axes_min, axes_max)
    axes.set_ylim(axes_min, axes_max)
    plt.xlabel('Modulation index - Resting periods')
    plt.ylabel('Modulation index - Running periods')
        
    # plot the ROIs according to their coordinates in running and resting states
    ax = plt.subplot(1,2,2)
    
    # kmeans: clustering the cells in three groups: positively, negratively and
    # non modulated by runnign speed (I don't use it anymore but just in case)
    nb_of_clusters = min(NbOfROI,nb_of_clusters)        
    coor_rest_run = np.column_stack((coor_rest, coor_run))
    estimator = KMeans(n_clusters = nb_of_clusters)
    # kmeans = estimator.fit(coor_rest_run)
    y_kmeans = estimator.fit_predict(coor_rest_run)
            
    # compute the circles
    
    clusters_centroids = dict()
    clusters_radii = dict()
    for cluster in range(nb_of_clusters):
        clusters_centroids[cluster] = list(zip(estimator.cluster_centers_[:, 0],estimator.cluster_centers_[:,1]))[cluster]
        clusters_radii[cluster] = max([np.linalg.norm(np.subtract(i,clusters_centroids[cluster])) for i in zip(coor_rest_run[y_kmeans == cluster, 0],coor_rest_run[y_kmeans == cluster, 1])])
        

#    plt.scatter(coor_rest, coor_run, c = my_c)
        
    colors = ['red', 'green', 'blue', 'yellow', 'k', 'pink']
    
    for cluster in range(nb_of_clusters):
        plt.scatter(coor_rest_run[y_kmeans == cluster, 0], 
                    coor_rest_run[y_kmeans == cluster, 1], c = colors[cluster])
        art = mpatches.Circle(clusters_centroids[cluster],clusters_radii[cluster], 
                              edgecolor=colors[cluster], linestyle = '--',
                              fill=False)
        ax.add_patch(art)

    plt.plot([axes_min, axes_max], [axes_min, axes_max], '--', color='k', 
             linewidth=1)
    axes = plt.gca()
    axes.set_xlim(axes_min, axes_max)
    axes.set_ylim(axes_min, axes_max)
    plt.xlabel('Modulation index - Resting periods')
    plt.ylabel('Modulation index - Running periods')
    
    fig2.suptitle('Coordinates PMI - {} - {} - Tseries {}'.format(p[-4], p[-5], p[-3][-1]))
    fig2.savefig('{}/coordinates_run_rest_pmi.pdf'.format(path))
        
    if show == True:
        fig2.show()
    else:
        plt.close(fig2)
   
    
# def plot_dF_spikes(dF, spikes, settings, chosen_index, do_filter=False,
#                    method_th='constant', c=0, show=True):
#     plt.ioff()
#     offset = np.amax(dF[chosen_index,:]) + 4
#     offset_spikes = 2.5
#     linelength = [1.1]
#     color_spikes = '#2b8cbe'
#     #color_spikes = 'blue_3b'
    
#     time = np.linspace(0,settings['time_seconds'],np.size(dF, 1))
    
#     normalized = np.copy(dF)
#     for i in range(len(dF)):
#         normalized[i] = (dF[i] - dF[i].min()) / (dF[i].max() - dF[i].min())
    
    
#     fig, ax = plt.subplots(figsize=(10,10))
#     for i in chosen_index:
        
#         trace = dF[i] + (chosen_index.index(i)*offset)
        
#         if do_filter == True:
#             trace = signal.savgol_filter(trace, 5, 2)
            
#         s = np.copy(spikes[i])
#         th = spiking_analysis.threshold(s, dF[i], settings, method = method_th)
#         for c, value in enumerate(s):
#             if value <= th:
#                 s[c] = 0
#         positions = np.nonzero(s)[0]
        
#         ax.eventplot(positions*settings['time_seconds']/np.size(dF, 1), 
#                       lineoffsets=((chosen_index.index(i)*offset)-offset_spikes), 
#                       colors = color_spikes, linewidths = 0.8, linelengths=linelength)
#         ax.plot(time, trace, 'k', linewidth=0.5, label = 'ROI-{}'.format(i))
#         ax.set_xlabel('Time (s)')
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.spines["left"].set_visible(False)
#         ax.spines["bottom"].set_visible(False)
#         ax.set_yticks([])
    
#     plt.setp(ax, yticklabels=[])
#     plt.savefig('{}/dF_spikes.pdf'.format(settings['save_path']))
#     plt.savefig('{}/dF_spikes'.format(settings['save_path']))
    
#     if show == True:
#         plt.show()
#     else:
#         plt.close()
  
def plot_dF_spikes(dF, spikes, thr, settings, chosen_index, do_filter=False,
                   method_th='constant', c=0, show=True, normalize=False):
    plt.ioff()
    if normalize == True:
        offset = 2
        offset_spikes = 0.1
        linelength = [0.2]
        normalized = np.copy(dF)
        for i in range(len(dF)):
            normalized[i] = (dF[i] - dF[i].min()) / (dF[i].max() - dF[i].min())
    
    else:
        offset = np.amax(dF[chosen_index,:]) + 4
        offset_spikes = 2.5
        linelength = [1.1]
        
    color_spikes = '#2b8cbe'
    #color_spikes = 'blue_3b'
    time = np.linspace(0,settings['time_seconds'],np.size(dF, 1))
    
    fig, ax = plt.subplots(figsize=(10,10))
    for i in chosen_index:
        
        if normalize == True:
            trace = normalized[i] + (chosen_index.index(i)*offset)
        else:
            trace = dF[i] + (chosen_index.index(i)*offset)
            
        if do_filter == True:
            trace = signal.savgol_filter(trace, 5, 2)
            
        s = np.copy(spikes[i])       
        for c, value in enumerate(s):
            if value <= thr[i]:
                s[c] = 0
        positions = np.nonzero(s)[0]
        
        ax.eventplot(positions*settings['time_seconds']/np.size(dF, 1), 
                      lineoffsets=((chosen_index.index(i)*offset)-offset_spikes), 
                      colors = color_spikes, linewidths = 0.8, linelengths=linelength)
        ax.plot(time, trace, 'k', linewidth=0.5, label = 'ROI-{}'.format(i))
        ax.set_xlabel('Time (s)')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        #ax.set_yticks([])
    
    #plt.setp(ax, yticklabels=[])
    plt.savefig('{}/dF_spikes.pdf'.format(settings['save_path']))
    plt.savefig('{}/dF_spikes'.format(settings['save_path']))
    
    if show == True:
        fig.show()
    else:
        plt.close(fig)

def plot_matrix_synchro(synchro, settings, show=True):
    plt.ioff()
    plt.matshow(synchro) 
    plt.colorbar()
    
    plt.xlabel('ROI')
    plt.ylabel('ROI')
#    plt.title('Synchrony')
    
    plt.savefig('{}/synchrony.pdf'.format(settings['save_path']))
    plt.savefig('{}/synchrony'.format(settings['save_path']))
      
    if show == True:
        plt.show()
    else:
        plt.close()
        
        
        
def plot_pmi_control_ket(coor_rest, coor_run, coor_rest_k, coor_run_k, path, 
                         nb_of_clusters = 2, show = True):
    plt.ioff()
    coor_run, coor_rest = np.asarray(coor_run), np.asarray(coor_rest)
    coor_run = coor_run[~np.isnan(coor_run)]
    coor_rest = coor_rest[~np.isnan(coor_rest)]
    NbOfROI = len(coor_rest)
    
    coor_run_k, coor_rest_k = np.asarray(coor_run_k), np.asarray(coor_rest_k)
    coor_run_k = coor_run_k[~np.isnan(coor_run_k)]
    coor_rest_k = coor_rest_k[~np.isnan(coor_rest_k)]
    NbOfROI_k = len(coor_rest_k)
    
    fig = plt.figure(figsize=(10,5))

    ax = plt.subplot(1,2,1)

    nb_of_clusters = min(NbOfROI,nb_of_clusters)        
    coor_rest_run = np.column_stack((coor_rest, coor_run))
    estimator = KMeans(n_clusters = nb_of_clusters)
    # kmeans = estimator.fit(coor_rest_run)
    y_kmeans = estimator.fit_predict(coor_rest_run)
            
    # compute the circles
    
    clusters_centroids = dict()
    clusters_radii = dict()
    for cluster in range(nb_of_clusters):
        clusters_centroids[cluster] = list(zip(estimator.cluster_centers_[:, 0],estimator.cluster_centers_[:,1]))[cluster]
        clusters_radii[cluster] = max([np.linalg.norm(np.subtract(i,clusters_centroids[cluster])) for i in zip(coor_rest_run[y_kmeans == cluster, 0],coor_rest_run[y_kmeans == cluster, 1])])
        

    #    plt.scatter(coor_rest, coor_run, c = my_c)
        
    colors = ['red', 'green', 'blue', 'yellow', 'k', 'pink']
    
    for cluster in range(nb_of_clusters):
        plt.scatter(coor_rest_run[y_kmeans == cluster, 0], 
                    coor_rest_run[y_kmeans == cluster, 1], c = colors[cluster])
        art = mpatches.Circle(clusters_centroids[cluster],clusters_radii[cluster], 
                              edgecolor=colors[cluster], linestyle = '--',
                              fill=False)
        ax.add_patch(art)

    plt.plot([axes_min, axes_max], [axes_min, axes_max], '--', color='k', 
             linewidth=1)
    axes = plt.gca()
    axes.set_xlim(axes_min, axes_max)
    axes.set_ylim(axes_min, axes_max)
    plt.xlabel('Modulation index - Resting periods')
    plt.ylabel('Modulation index - Running periods')
    plt.title('control')
  

    ax = plt.subplot(1,2,2)
    
    # kmeans: clustering the cells in three groups: positively, negratively and
    # non modulated by runnign speed (I don't use it anymore but just in case)
    nb_of_clusters = min(NbOfROI_k,nb_of_clusters)        
    coor_rest_run_k = np.column_stack((coor_rest_k, coor_run_k))
    estimator = KMeans(n_clusters = nb_of_clusters)
    # kmeans = estimator.fit(coor_rest_run)
    y_kmeans = estimator.fit_predict(coor_rest_run_k)
            
    # compute the circles
    
    clusters_centroids = dict()
    clusters_radii = dict()
    for cluster in range(nb_of_clusters):
        clusters_centroids[cluster] = list(zip(estimator.cluster_centers_[:, 0],estimator.cluster_centers_[:,1]))[cluster]
        clusters_radii[cluster] = max([np.linalg.norm(np.subtract(i,clusters_centroids[cluster])) for i in zip(coor_rest_run_k[y_kmeans == cluster, 0],coor_rest_run_k[y_kmeans == cluster, 1])])
        

    #    plt.scatter(coor_rest, coor_run, c = my_c)
        
    colors = ['red', 'green', 'blue', 'yellow', 'k', 'pink']
    
    for cluster in range(nb_of_clusters):
        plt.scatter(coor_rest_run_k[y_kmeans == cluster, 0], 
                    coor_rest_run_k[y_kmeans == cluster, 1], c = colors[cluster])
        art = mpatches.Circle(clusters_centroids[cluster],clusters_radii[cluster], 
                              edgecolor=colors[cluster], linestyle = '--',
                              fill=False)
        ax.add_patch(art)

    plt.plot([axes_min, axes_max], [axes_min, axes_max], '--', color='k', 
             linewidth=1)
    axes = plt.gca()
    axes.set_xlim(axes_min, axes_max)
    axes.set_ylim(axes_min, axes_max)
    plt.xlabel('Modulation index - Resting periods')
    plt.ylabel('Modulation index - Running periods')
    plt.title('ketamine')
    
    fig.suptitle('PMI control vs ketamine')
    fig.savefig('{}/PMI_control_vs_ketamine'.format(path))
    
    if show == True:
        fig.show()
    else:
        plt.close(fig)

def spikes_movement(dF, spikes, thr, resting_binary, movement, settings, chosen_index, do_filter=False,
                    method_th='sumbre', c=0, show=True, normalize=False):
    """
    Create figure with speed trace, fluorescence traces, spiking traces and periods of rest and movement.
    
    Marcel van Velze, m.debritovanvelze@icm-institute.org

    Parameters
    ----------
    dF : Two-dimentional array of fluorescence traces. 
    spikes : Raw spiking traces (output of suite2p).
    movement : dictionary containing movement calculations.
    settings : settings folder.
    chosen_index : List of roi indexes to be plotted.
    do_filter : TYPE, optional
        Do filter or not. Filter is Savinsky-Golay. The default is False.
    method_th : TYPE, optional
        Method for filtering spiking data. See 'spiking_analysis.py'. The default is 'sumbre'.
    c : TYPE, optional
        DESCRIPTION. The default is 0.
    show : TYPE, optional
        Whether or not to display generated figure. The default is True.
    normalize : TYPE, optional
        Whether or not to normalize data from [0-1]
    Returns
    -------
    None.

    """
    if normalize == True:
        offset = 2
        offset_spikes = 0.1
        linelength = [0.2]
        normalized = np.copy(dF)
        for i in range(len(dF)):
            normalized[i] = (dF[i] - dF[i].min()) / (dF[i].max() - dF[i].min())
        
    else:
        offset = np.amax(dF[chosen_index,:]) + 4
        offset_spikes = 2.5
        linelength = [1.1]

    color_spikes = '#2b8cbe'
    color_speed = '#2b8cbe'
    color_stimulation = '#e34a33'
    #color_spikes = 'blue_3b'
    
    time = np.linspace(0,settings['time_seconds'],np.size(dF, 1))
    
    plt.ioff()
    
    if 'puff signal' in movement:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios': [1, 1, 6]})
    else:
        f, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios': [1, 6]})
    #f.suptitle('{}'.format(file_name[0]), size=20)
    
    # Speed
    for x, i in enumerate(movement['binary_movement']):
        if i == 1:
            ax1.axvline(time[x], color = '#deebf7', linewidth = 0.1)
                        
    ax1.plot(time, movement['speed'], color = color_speed, label = 'Locomotion')
    ax1.set_title('Speed (cm/s) - {}% movement'.format(int(movement['percentage']*100)), loc='center')
    ax1.legend(loc='upper right', frameon=False)
    ax1.set_ylabel('cm/s')
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    
    if 'puff signal' in movement:
        for x, i in enumerate(movement['binary_movement']):
            if i == 1:
                ax2.axvline(time[x], color = '#deebf7', linewidth = 0.1)
                            
        ax2.plot(time, movement['puff signal'], color = color_stimulation, label = 'Whisker Stimulation')
        ax2.legend(loc='upper right', frameon=False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        
    
    # dF + spikes
    
    for x, i in enumerate(movement['binary_movement']):
        if i == 1:
            ax3.axvline(time[x], color = '#deebf7', linewidth = 0.1)
    
    for x, i in enumerate(resting_binary):
        if i == 0:
            ax3.axvline(time[x], color = '#fee0d2', linewidth = 0.1)
                        
    for i in chosen_index:
        
        if normalize == True:
            trace = normalized[i] + (chosen_index.index(i)*offset) 
        else:
            trace = dF[i] + (chosen_index.index(i)*offset) 
        
        if do_filter == True:
            trace = signal.savgol_filter(trace, 5, 2)
            
        s = np.copy(spikes[i])
        for c, value in enumerate(s):
            if value <= thr[i]:
                s[c] = 0
        positions = np.nonzero(s)[0]
        
        ax3.eventplot(positions*settings['time_seconds']/np.size(dF, 1), 
                      lineoffsets=((chosen_index.index(i)*offset)-offset_spikes), 
                      colors = color_spikes, linewidths = 0.8, linelengths=linelength,
                      zorder = 50)
        ax3.plot(time, trace, 'k', linewidth=0.5)
    
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.yaxis.set_visible(True)
    
    plt.savefig('{}/spikes_movement.pdf'.format(settings['save_path']))
    plt.savefig('{}/spikes_movement'.format(settings['save_path']))
    
    if show == True:
        plt.show()
    else:
        plt.close()
        
def spikes_movement_whisking(dF, spikes, thr, resting_binary, movement, whisking, settings, chosen_index, do_filter=False,
                    method_th='sumbre', c=0, show=True, normalize=False):
    """
    Create figure with speed trace, fluorescence traces, spiking traces and periods of rest and movement.
    
    Marcel van Velze, m.debritovanvelze@icm-institute.org

    Parameters
    ----------
    dF : Two-dimentional array of fluorescence traces. 
    spikes : Raw spiking traces (output of suite2p).
    movement : dictionary containing movement calculations.
    settings : settings folder.
    chosen_index : List of roi indexes to be plotted.
    do_filter : TYPE, optional
        Do filter or not. Filter is Savinsky-Golay. The default is False.
    method_th : TYPE, optional
        Method for filtering spiking data. See 'spiking_analysis.py'. The default is 'sumbre'.
    c : TYPE, optional
        DESCRIPTION. The default is 0.
    show : TYPE, optional
        Whether or not to display generated figure. The default is True.
    normalize : TYPE, optional
        Whether or not to normalize data from [0-1]
    Returns
    -------
    None.

    """
    if normalize == True:
        offset = 2
        offset_spikes = 0.1
        linelength = [0.2]
        normalized = np.copy(dF)
        for i in range(len(dF)):
            normalized[i] = (dF[i] - dF[i].min()) / (dF[i].max() - dF[i].min())
        
    else:
        offset = np.amax(dF[chosen_index,:]) + 4
        offset_spikes = 2.5
        linelength = [1.1]

    color_spikes = '#2b8cbe'
    color_speed = '#2b8cbe'
    color_stimulation = '#e34a33'
    #color_spikes = 'blue_3b'
    
    time = np.linspace(0,settings['time_seconds'],np.size(dF, 1))
    
    plt.ioff()
    
    if 'puff signal' in movement:
        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios': [1, 1, 1, 1, 8]})
    else:
        f, (ax1, ax2, ax3, ax5) = plt.subplots(4, 1, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios': [1, 1, 1, 7]})
    #f.suptitle('{}'.format(file_name[0]), size=20)
    
    # Speed
    # for x, i in enumerate(movement['binary_movement']):
    #     if i == 1:
    #         ax1.axvline(time[x], color = '#deebf7', linewidth = 0.1)
                        
    ax1.plot(time, movement['speed'], color = color_speed, label = 'Locomotion')
    ax1.legend(loc='upper right', frameon=False)
    ax1.set_title('Speed (cm/s) - {}% movement'.format(int(movement['percentage']*100)), loc='center')
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    
    # Whisking
    # for x, i in enumerate(whisking['binary_whisking']):
    #     if i == 1:
    #         ax2.axvline(time[x], color = '#deebf7', linewidth = 0.1)
                        
    ax2.plot(time, whisking['normalized_trace'], color = color_speed, label = 'Whisking')
    ax2.legend(loc='upper right', frameon=False)
    ax2.set_title('Whisking', loc='center')
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    
    # Behavior
        # Locomotion
    for i in movement['events']['location']:
        ax3.fill_between(x=[time[i[0]], time[i[1]-1]], y1= 0, y2= 1, color='y')
        # Whisking only
    if len(whisking['whisking only']) == 0:
        # Whisking
        for i in whisking['location_bouts']:
            ax3.fill_between(x=[time[i[0]], time[i[1]-1]], y1= 0, y2= 1, color='r')
    else:
        for i in whisking['whisking only']['bout_location']:
            ax3.fill_between(x=[time[i[0]], time[i[1]-1]], y1= 0, y2= 1, color='b')

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    
    if 'puff signal' in movement: 
        ax4.plot(time, movement['puff signal'], color = color_stimulation, label = 'Whisker Stimulation')
        ax4.legend(loc='upper right', frameon=False)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        ax4.spines["left"].set_visible(False)
        ax4.spines["bottom"].set_visible(False)
    
    
    # dF + spikes
    # for x, i in enumerate(movement['binary_movement']):
    #     if i == 1:
    #         ax4.axvline(time[x], color = '#deebf7', linewidth = 0.1)
    
    # for x, i in enumerate(resting_binary):
    #     if i == 0:
    #         ax4.axvline(time[x], color = '#fee0d2', linewidth = 0.1)
                        
    for i in chosen_index:
        
        if normalize == True:
            trace = normalized[i] + (chosen_index.index(i)*offset) 
        else:
            trace = dF[i] + (chosen_index.index(i)*offset) 
        
        if do_filter == True:
            trace = signal.savgol_filter(trace, 5, 2)
            
        s = np.copy(spikes[i])
        for c, value in enumerate(s):
            if value <= thr[i]:
                s[c] = 0
        positions = np.nonzero(s)[0]
        
        ax5.eventplot(positions*settings['time_seconds']/np.size(dF, 1), 
                      lineoffsets=((chosen_index.index(i)*offset)-offset_spikes), 
                      colors = color_spikes, linewidths = 0.8, linelengths=linelength,
                      zorder = 50)
        ax5.plot(time, trace, 'k', linewidth=0.5)
    
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.spines["left"].set_visible(False)
    ax5.spines["bottom"].set_visible(False)
    ax5.yaxis.set_visible(True)
    
    plt.savefig('{}/spikes_movement.pdf'.format(settings['save_path']))
    plt.savefig('{}/spikes_movement'.format(settings['save_path']))
    
    if show == True:
        plt.show()
    else:
        plt.close()
        
def figure_one_genotype(title = 'VIP', data_type = 'dF'):
    
    data = pd.read_excel(easygui.fileopenbox(msg='Select Excell file containing the data'))
    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15,6), gridspec_kw={'width_ratios': [1.2, 3, 3, 0.7]})   
    
    f.suptitle(title + ' - ' + data_type)  
             
                            ######################
                            # Rest Run bar graph #
                            ######################
    if data_type == 'dF':
        rest = np.array(data['Rest dF'].dropna().tolist(), dtype=np.float64)
        run = np.array(data['Run dF'].dropna().tolist(), dtype=np.float64)
    elif data_type == 'deconvolved data':
        rest = np.array(data['Rest Nb of spikes'].dropna().tolist(), dtype=np.float64)
        run = np.array(data['Run Nb of spikes'].dropna().tolist(), dtype=np.float64)
    
    restrun = np.zeros((3,2))
    restrun[0, 0] = rest.mean()
    restrun[1, 0] = stats.sem(rest)
    restrun[2, 0] = rest.size
    restrun[0, 1] = run.mean()
    restrun[1, 1] = stats.sem(run)
    restrun[2, 1] = run.size
    
    g = [0.20, 0.35]
    color_bars = ['#a6bddb', '#1c9099']
    width = 0.15
    labels = ['Rest', 'Run']

    label_n_cells = [('n= '+ str(int(i))) for i in restrun[2]]
    height_label = [(restrun[0,i]+restrun[1,i]+ 0.5*restrun[1,i]) for i in range(0,2)]
    
    #ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax1.bar(g, restrun[0], width, yerr=restrun[1], align='center', ecolor='black', capsize=10, color=color_bars)
    for x, y, z in zip([w - (width/2) for w in g], height_label, label_n_cells):
        ax1.text(float(x), y, z, fontsize=6)
    ax1.set_ylim(0, max(height_label) + 0.25 * max(height_label))
    ax1.set_xlim((g[0]-width, g[-1]+width))
    plt.setp(ax1, xticks=g, xticklabels=labels)  
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    if data_type == 'dF':
        ax1.set_ylabel('Locomotion (\u0394F/F0)')
    elif data_type == 'deconvolved data':
        ax1.set_ylabel('Locomotion (Events/s)')
    
                            #######################
                            # Scatter Rest vs Run #
                            #######################   
    
    ax2.scatter(rest, run, s=5, c='grey', alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    x = np.linspace(*ax2.get_xlim())
    ax2.plot(x, x, '--', c='#bdbdbd', alpha = 0.5)
    #ax2.set_xscale('log')
    #ax2.set_yscale('log')
    if data_type == 'dF':
        ax2.set_ylabel('Locomotion (\u0394F/F0)')
        ax2.set_xlabel('Stationary (\u0394F/F0)')
    elif data_type == 'deconvolved data':
        ax2.set_ylabel('Locomotion (Events/s)')
        ax2.set_xlabel('Stationary (Events/s)')
    
                            #################
                            # Histogram LMI #
                            #################   
    
    if data_type == 'dF':
        lmi = np.array(data['LMI dF'].dropna().tolist(), dtype=np.float64)
    elif data_type == 'deconvolved data':
        lmi = np.array(data['LMI Nb of spikes'].dropna().tolist(), dtype=np.float64)
        
    ax3.set_ylabel('Neurons (%)')
    ax3.set_xlabel('Locomotion Modulation Index')
    ax3.axvline(linewidth=2, color='#bdbdbd', ls='--')
    ax3.hist(lmi, weights=(np.zeros_like(lmi) + 1. / lmi.size)*100,
             bins=20, range = (-1.5,1.5), color = '#3182bd',ec = '#ffffff', lw=1, alpha = 0.5)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)   
    
                                ################
                                # LMI bar plot #
                                ################ 
        
    lmi_stat = np.zeros((3,1))
    lmi_stat[0, 0] = lmi.mean()
    lmi_stat[1, 0] = stats.sem(lmi)
    lmi_stat[2, 0] = lmi.size
    
    ax4.bar(0.20, lmi_stat[0], width, yerr=lmi_stat[1], align='center', ecolor='black', capsize=10, color='#3182bd')
    ax4.text(float(0.2 - (width/2)), lmi_stat[0]+lmi_stat[1]+ 0.5*lmi_stat[1], 'n= '+ str(int(lmi_stat[2])), fontsize=6)
    ax4.set_ylim(0, ((lmi_stat[0]+lmi_stat[1]+ 0.5*lmi_stat[1]) + (0.25 * (lmi_stat[0]+lmi_stat[1]+ 0.5*lmi_stat[1]))))
    ax4.set_xlim((0.2-width, 0.2+width))  
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.set_ylabel('LMI')
    ax4.axes.get_xaxis().set_ticks([])
    
    plt.tight_layout(pad = 1.5)


def figure_two_genotype(title = 'VIP vs SST', data_type = 'dF'):
    
    data1 = pd.read_excel(easygui.fileopenbox(msg='Select Excell file containing the data for Genotype 1'))
    data2 = pd.read_excel(easygui.fileopenbox(msg='Select Excell file containing the data for Genotype 2'))
    
   
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15,6), gridspec_kw={'width_ratios': [1.2, 3, 3, 0.7]})   
    
    f.suptitle(title + ' - ' + data_type)  
             
                            ######################
                            # Rest Run bar graph #
                            ######################
    if data_type == 'dF':
        rest1 = np.array(data1['Rest dF'].dropna().tolist(), dtype=np.float64)
        run1 = np.array(data1['Run dF'].dropna().tolist(), dtype=np.float64)
        rest2 = np.array(data2['Rest dF'].dropna().tolist(), dtype=np.float64)
        run2 = np.array(data2['Run dF'].dropna().tolist(), dtype=np.float64)
    elif data_type == 'deconvolved data':
        rest1 = np.array(data1['Rest Nb of spikes'].dropna().tolist(), dtype=np.float64)
        run1 = np.array(data1['Run Nb of spikes'].dropna().tolist(), dtype=np.float64)
        rest2 = np.array(data2['Rest Nb of spikes'].dropna().tolist(), dtype=np.float64)
        run2 = np.array(data2['Run Nb of spikes'].dropna().tolist(), dtype=np.float64)
    
    restrun = np.zeros((3,4))
    restrun[0, 0] = rest1.mean()
    restrun[1, 0] = stats.sem(rest1)
    restrun[2, 0] = rest1.size
    restrun[0, 1] = rest2.mean()
    restrun[1, 1] = stats.sem(rest2)
    restrun[2, 1] = rest2.size
    restrun[0, 2] = run1.mean()
    restrun[1, 2] = stats.sem(run1)
    restrun[2, 2] = run1.size
    restrun[0, 3] = run2.mean()
    restrun[1, 3] = stats.sem(run2)
    restrun[2, 3] = run2.size
    
    g = [0.20, 0.35, 0.55, 0.70]
    color_bars = ['#3182bd', '#e34a33', '#3182bd', '#e34a33']
    width = 0.15
    g_labels = [0.275, 0.625]
    labels = ['Rest', 'Run']

    label_n_cells = [('n= '+ str(int(i))) for i in restrun[2]]
    height_label = [(restrun[0,i]+restrun[1,i]+ 0.5*restrun[1,i]) for i in range(0,4)]
    
    #ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax1.bar(g, restrun[0], width, yerr=restrun[1], align='center', ecolor='black', capsize=10, color=color_bars)
    for x, y, z in zip([w - (width/2) for w in g], height_label, label_n_cells):
        ax1.text(float(x), y, z, fontsize=6)
    ax1.set_ylim(0, max(height_label) + 0.25 * max(height_label))
    ax1.set_xlim((g[0]-width, g[-1]+width))
    plt.setp(ax1, xticks=g_labels, xticklabels=labels)  
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    if data_type == 'dF':
        ax1.set_ylabel('Locomotion (\u0394F/F0)')
    elif data_type == 'deconvolved data':
        ax1.set_ylabel('Locomotion (Events/s)')
    
                            #######################
                            # Scatter Rest vs Run #
                            #######################   
    
    ax2.scatter(rest1, run1, s=5, c='#3182bd', alpha=0.5)
    ax2.scatter(rest2, run2, s=5, c='#e34a33', alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    #ax2.set_xscale('log')
    #ax2.set_yscale('log')
    x = np.linspace(*ax2.get_xlim())
    ax2.plot(x, x, '--', c='#bdbdbd', alpha = 0.5)
    #ax2.autoscale
    if data_type == 'dF':
        ax2.set_ylabel('Locomotion (\u0394F/F0)')
        ax2.set_xlabel('Stationary (\u0394F/F0)')
    elif data_type == 'deconvolved data':
        ax2.set_ylabel('Locomotion (Events/s)')
        ax2.set_xlabel('Stationary (Events/s)')
    
                            #################
                            # Histogram LMI #
                            #################   
    
    if data_type == 'dF':
        lmi1 = np.array(data1['LMI dF'].dropna().tolist(), dtype=np.float64)
        lmi2 = np.array(data2['LMI dF'].dropna().tolist(), dtype=np.float64)
    elif data_type == 'deconvolved data':
        lmi1 = np.array(data1['LMI Nb of spikes'].dropna().tolist(), dtype=np.float64)
        lmi2 = np.array(data2['LMI Nb of spikes'].dropna().tolist(), dtype=np.float64)
    
    ax3.set_ylabel('Neurons (%)')
    ax3.set_xlabel('Locomotion Modulation Index')
    ax3.axvline(linewidth=2, color='#bdbdbd', ls='--')
    ax3.hist(lmi1, weights=(np.zeros_like(lmi1) + 1. / lmi1.size)*100,
             bins=20, range = (-1.5,1.5), color = '#3182bd',ec = '#ffffff', lw=1, alpha = 0.5)
    ax3.hist(lmi2, weights=(np.zeros_like(lmi2) + 1. / lmi2.size)*100,
             bins=20, range = (-1.5,1.5), color = '#e34a33',ec = '#ffffff', lw=1, alpha = 0.5)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)   
    
                                ################
                                # LMI bar plot #
                                ################ 
        
    lmi_stat = np.zeros((3,2))
    lmi_stat[0, 0] = lmi1.mean()
    lmi_stat[1, 0] = stats.sem(lmi1)
    lmi_stat[2, 0] = lmi1.size
    lmi_stat[0, 1] = lmi2.mean()
    lmi_stat[1, 1] = stats.sem(lmi2)
    lmi_stat[2, 1] = lmi2.size
    
    label_n_cells_LMI = [('n= '+ str(int(i))) for i in lmi_stat[2]]
    height_label_LMI = [(lmi_stat[0,i]+lmi_stat[1,i]+ 0.5*lmi_stat[1,i]) for i in range(0,2)]
    
    ax4.bar([0.20, 0.35], lmi_stat[0], width, yerr=lmi_stat[1], align='center', ecolor='black', capsize=10, color=['#3182bd', '#e34a33'])
    for x, y, z in zip([w - (width/2) for w in g], height_label_LMI, label_n_cells_LMI):
        ax4.text(float(x), y, z, fontsize=6)
    ax4.set_ylim(0, max(lmi_stat[0])+0.2*max(lmi_stat[0]))
    ax4.set_xlim((0.2-width, 0.4+width))  
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.set_ylabel('LMI')
    ax4.axes.get_xaxis().set_ticks([])
    
    plt.tight_layout(pad = 1.5)
    
    
def locomotionVSwhisking(movement, whisking, savepath):
    
    color_speed = '#2b8cbe'
    
    time_movement = np.linspace(0,300,len(movement['speed']))
    time_whisking = np.linspace(0,300,len(whisking['resampled_trace']))
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,5), gridspec_kw={'height_ratios': [1, 1]})
    
    for x, i in enumerate(movement['binary_movement']):
        if i == 1:
            ax1.axvline(time_movement[x], color = '#deebf7', linewidth = 0.1)
    for x, i in enumerate(movement['extended_binary_movement']):
        if i == 0:
            ax1.axvline(time_movement[x], color = '#fee0d2', linewidth = 0.1)
                            
    ax1.plot(time_movement, movement['speed'], color = color_speed, linewidth = 1)
    ax1.set_title('Locomotion (cm/s)')
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
        
    for x, i in enumerate(movement['binary_movement']):
        if i == 1:
            ax2.axvline(time_movement[x], color = '#deebf7', linewidth = 0.1)
        
    for x, i in enumerate(movement['extended_binary_movement']):
        if i == 0:
            ax2.axvline(time_movement[x], color = '#fee0d2', linewidth = 0.1)
                            
       
    trace = signal.savgol_filter(whisking['resampled_trace'], 21, 2)
            
    ax2.plot(time_whisking, trace, 'k', linewidth = 1)   
    ax2.set_title('Whisking (AU)')
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    
    plt.savefig('{}/whiskingVSlocomotion.pdf'.format(savepath))
    plt.savefig('{}/whiskingVSlocomotion'.format(savepath))
    
    plt.show()
    
def heatmap_whisking(dF, movement, whisking, settings):
    
    color_speed = '#2b8cbe'
    color_whisk = '#f03b20'
    color_stimulation = '#e34a33'
    
    decimated = signal.decimate(dF, 20, axis = 1)
    
    # Normalize data
    normalized = np.copy(decimated)
    for i in range(len(decimated)):
        normalized[i] = (decimated[i] - decimated[i].min()) / (decimated[i].max() - decimated[i].min())
        
    time = np.linspace(0,settings['time_seconds'],np.size(dF, 1))
    
    plt.ioff()
    
    if 'puff signal' in movement:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios': [1, 1, 1, 6]})
    else:
        f, (ax1, ax2, ax4) = plt.subplots(3, 1, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios': [1, 1, 6]})
    
    # Locomotion
    ax1.plot(time, movement['speed'], color = color_speed, label = 'Locomotion')
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.legend(loc='upper right', frameon=False)
    ax1.set_ylabel('cm/s')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.5)
    cax1.axis('off')
    
    # Whisking
    ax2.plot(time, whisking['normalized_trace'], color = color_whisk, label='Whisking')
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.legend(loc='upper right', frameon=False)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3%", pad=0.5)
    cax2.axis('off')
    
    if 'puff signal' in movement:
        ax3.plot(time, movement['puff signal'], color = color_stimulation, label = 'Whisker Stimulation')
        ax3.legend(loc='upper right', frameon=False)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["left"].set_visible(False)
        ax3.spines["bottom"].set_visible(False)
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="3%", pad=0.5)
        cax3.axis('off')
    
    # Heatmap
    im = ax4.imshow(normalized, cmap='Greys',interpolation="None", aspect='auto', extent = [0, settings['time_seconds'], 0-0.5, np.size(normalized, 0)-0.5])
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    ax4.spines["bottom"].set_visible(False)
    ax4.set_ylabel('Cell Number')
    ax4.set_xlabel('Time (s)')
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="3%", pad=0.5)
    f.colorbar(im, cax=cax4, orientation='vertical')
    
    
    plt.savefig('{}/Heatmap.pdf'.format(settings['save_path']))
    plt.savefig('{}/Heatmap'.format(settings['save_path']))
    if settings['do_show'] == True:
        plt.show()
    else:
        plt.close()
    
def heatmap(dF, movement, settings):
    
    color_speed = '#2b8cbe'
    color_stimulation = '#e34a33'
    
    # Normalize data
    decimated = signal.decimate(dF, 20, axis = 1)
    
    # Normalize data
    normalized = np.copy(decimated)
    for i in range(len(decimated)):
        normalized[i] = (decimated[i] - decimated[i].min()) / (decimated[i].max() - decimated[i].min())
        
    time = np.linspace(0,settings['time_seconds'],np.size(dF, 1))
    
    plt.ioff()
    
    if 'puff signal' in movement:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios': [1, 1, 6]})
    else:
        f, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios': [1, 6]})
    # Locomotion
    ax1.plot(time, movement['speed'], color = color_speed, label = 'Locomotion')
    ax1.legend(loc='upper right', frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.set_ylabel('cm/s')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.5)
    cax1.axis('off')
    
    if 'puff signal' in movement:
        ax2.plot(time, movement['puff signal'], color = color_stimulation, label = 'Whisker Stimulation')
        ax2.legend(loc='upper right', frameon=False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="3%", pad=0.5)
        cax2.axis('off')
    
    # ScatterPlot
    im = ax3.imshow(normalized, cmap='Greys',interpolation="None", aspect='auto', extent = [0, settings['time_seconds'], 0-0.5, np.size(normalized, 0)-0.5])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.set_ylabel('Cell Number')
    ax3.set_xlabel('Time (s)')
    divider2 = make_axes_locatable(ax3)
    cax3 = divider2.append_axes("right", size="3%", pad=0.5)
    f.colorbar(im, cax=cax3, orientation='vertical')
    plt.savefig('{}/Heatmap.pdf'.format(settings['save_path']))
    plt.savefig('{}/Heatmap'.format(settings['save_path']))
    
    if settings['do_show'] == True:
        plt.show()
    else:
        plt.close()
    
