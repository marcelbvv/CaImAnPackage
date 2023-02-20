# Functions for statistical analysis

import numpy as np
import math
import collections
from scipy import signal
from scipy.stats import pearsonr, spearmanr, sem
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import random
import matplotlib.pyplot as plt


def bin_trace(trace, settings, method = 'mean', window = 5):

    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................

    This function aligns speed and neuronal data (to have the same number of
    points). We will only upsample the speed trace and never touch the neuronal
    trace. If the number of point in the speed trace is higher than the number
    of points in the neuronal trace,  we can't use this function. We can use
    two methods upsampling : linear or cubic spline interpolation.

    ...........................................................................

    - - - - - - - - - - - - - - - METHODS - - - - - - - - - - - - - - - - - - -

    'mean'            mean (by default)
    'sum'             sum of the spike amplitude
    'Nb of events'    number of events

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    trace             any trace you want, one or several ROIs
    method            mean, sum or number of events
    settings          (use of the final "sampling frequency)
    window            duration used to binned the trace (5 by default)

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    trace_bin         binned trace

    ...........................................................................
    """

    sampling_rate = settings['fs']
    lg_window = sampling_rate*window

    one_row = len(trace.shape)
    if one_row == 1 :
        trace = trace.reshape((1, len(trace)))

    rows, columns = trace.shape
    Nb_Of_It = int(columns/(sampling_rate*window))
    trace_bin = np.zeros((rows, Nb_Of_It + 1))

    for i in range(Nb_Of_It + 1):

        if method == 'mean':
            trace_bin[:,i] = \
            np.mean(trace[:, i*lg_window:(i+1)*lg_window], axis = 1)

        if method == 'sum':
            trace_bin[:,i] = sum(trace[i*lg_window:(i+1)*lg_window], axis = 1)
    
    trace_bin = trace_bin[:,np.logical_not(np.isnan(trace_bin[0]))]
    
    return trace_bin


def sort_data(F, method = 'StandardScaler', sorted_first_pc = True, 
                 binned_01 = True):

    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ............................................................................

    This function reshape the fluorescence trace (1 or multiple ROIs) : you can
    sort it by the first  PC and bin fluorescence trace between O and 1.

    ............................................................................

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    F                 fluorescence data (F or dF/F)
                      /!\ in rows the fluorescence for each trace in columns the
                          number of ROIs
    sorted_first_pc   do yo want the data to be sorted by the first PC (by
                      default, sorted_first_pc = True)
    binned_01         do yo want the data to be binned between 0 and 1 (by
                      default, binned_01 = True)

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - -

    F                 F in the way you want it
    index_ROI         index of each ROI (the way it's sorted)

    ............................................................................
    """

    # sort data by the first pc
    if sorted_first_pc == True:
        
        pca = PCA(n_components=1)
        
        if method == 'StandardScaler':
            F_normalized = StandardScaler().fit_transform(F.T)
            principalComponent = pca.fit_transform(F_normalized.T)
            
        if method == 'MinMax':
            scaler = MinMaxScaler(feature_range=[0,1])
            F_rescaled = scaler.fit_transform(F.T) 
            principalComponent = pca.fit_transform(F_rescaled.T)
            
        order = np.argsort(principalComponent, axis=0)
        order = order[::-1]
        F_Sorted = np.array([F[i] for i in order])
        F_Sorted = np.reshape(F_Sorted, (F.shape[0], F.shape[1]))
        
        F = F_Sorted
        
    else :
        order = [i for i in len(F)]

    # binned all traces between 0 and 1
    if binned_01 == True:
        scaler = MinMaxScaler(feature_range=[0,1])
        F_binned = scaler.fit_transform(F.T)
        F = F_binned.T
        
    else :
        pass

    return F, order


def neuron_speed_correlation(speed, neuronal_trace, method = 'spearman',
                             do_bin = False,
                             method_speed = 'mean', method_neuron = 'mean',
                             samplingRate = 30, window_bin = 5):

    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................

    This function computes the correlation between a behavioural trace (for
    exemple speed) and neuronal data for each ROI.

    ...........................................................................

    - - - - - - - - - - - - - - - METHODS - - - - - - - - - - - - - - - - - - -

    'pearson'         pearson  correlation 
    'spearman'        spearman correlation (by default)

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    speed             speed trace
    neuronal_trace    any neuronal trace (F, dF/F or spikes)
    method            spearman (by default) or pearson correlation
    do_bin            binned or not (by default) before computing the spearman
                      correlation
    method_speed      mean or sum over periods of time for the speed
    method_neuron     mean or sum for the speed neuronal data
    sampling_rate     (= 30 by default)
    window_bin        duration of the window (= 5 by default)

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    rho               spearman correlation coefficient
    p_val             p-value associated with this correlation coefficient

    ...........................................................................
    """

    if do_bin == True:
        speed = bin_trace(speed, method = method_speed, 
                          sampling_rate = samplingRate, window = window_bin)
        neuronal_trace = bin_trace(neuronal_trace, method = method_neuron, 
                                   sampling_rate = samplingRate, 
                                   window = window_bin)
    else:
        pass
    
    NbOfROI = neuronal_trace.shape[0]
    
    if method == 'pearson':
        rho, p_val = np.zeros(NbOfROI), np.zeros(NbOfROI)
        for i in range(NbOfROI):
            rho[i], p_val[i] = pearsonr(speed, neuronal_trace[i]) 
            
    if method == 'spearman':
        rho, p_val = spearmanr(speed, neuronal_trace, axis = 1)
    
        if NbOfROI > 1:
            rho, p_val = rho[0,1:len(rho)], p_val[0,1:len(p_val)]
        elif NbOfROI == 1:
            rho, p_val = [rho], [p_val] 
        else:
            print('The neuronal trace is empty.')
            
    return rho, p_val


def LMI(movement_active, movement_inactive, neuronal_data, settings,
        method = 'mean dF', method_th = 'constant'):

    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................

    Computation of the LMI : index of the modulation of barrel cortex neuronal
    activity by locomotion.

    ...........................................................................

    - - - - - - - - - - - - - - - METHODS - - - - - - - - - - - - - - - - - - -

    'mean dF'         mean dF/F in running and resting periods (by default)
    'sum amplitude'   sum of the amplitudes of the spikes
    'events rate'     number of events (binary or thresholded trace)

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    neuronal_trace    any neuronal trace (F, dF/F or spikes (binary trace))
                      Rq: use the thresholded spike trace
    settings          all settings by default
    method            mean dF (by default), sum amplitude or events rate
    method_th         method used to threshold the firing trace (constant by
                                                                 default)

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    coor_run          coordinates in the running state
    coor_rest         coordinates in the resting state
    LMI               index of the modulation of barrel cortex neuronal 
                      activity by locomotion

    ...........................................................................

                                  FORMULA

        LMI = (coor_run-coor_rest) / (coor_run+coor_rest) for each ROI

    ...........................................................................
    """
    sampling_rate = settings['fs']

    # reshape the data if there is only one ROI
    one_ROI = len(neuronal_data.shape)
    if one_ROI == 1 :
        neuronal_data = neuronal_data.reshape((1, len(neuronal_data)))

    NbOfROI, _ = neuronal_data.shape

    neuronal_run = np.zeros((NbOfROI,np.count_nonzero(movement_active)))
    neuronal_rest = np.zeros((NbOfROI,np.count_nonzero(movement_inactive<1)))
    LMI = np.zeros(NbOfROI)

    for roi in range(NbOfROI):
        neuronal_run[roi,:] = [neuronal_data[roi,i] for i in np.nonzero(movement_active)[0]]
        neuronal_rest[roi,:] = [neuronal_data[roi,i] for i in np.nonzero(movement_inactive<1)[0]]


    # using the mean(dF/F)
    if method == 'mean dF':
        coor_run = np.mean(neuronal_run, 1)
        coor_rest = np.mean(neuronal_rest, 1)

    # using the sum of the amplitude (only using the spike trace)
    elif method == 'sum amplitude':
        coor_run = np.sum(neuronal_run,1)*sampling_rate/len(neuronal_run[0])
        coor_rest = np.sum(neuronal_rest,1)*sampling_rate/len(neuronal_rest[0])

    # using the number of events (only using the spike trace)
    elif method == 'events rate':
        coor_run, coor_rest = np.zeros(NbOfROI), np.zeros(NbOfROI) 
        for roi in range(NbOfROI):
            coor_run[roi] = np.count_nonzero(neuronal_run[roi] >= 1 ) * sampling_rate / len(neuronal_run[0])  #spikes binary, no need for the threshold
            coor_rest[roi] = np.count_nonzero(neuronal_rest[roi] >= 1) * sampling_rate / len(neuronal_rest[0])

    else:
        pass

    # compute the LMI
    LMI = [(m-s)/(m+s) for m, s in zip(coor_run, coor_rest)]

    return coor_run, coor_rest, LMI


def PMI(neuronal_trace, speed, settings, ncats1, ncats2 = 2, withlog = False):

    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................

    This function takes in one 1-D vectors specified in var1 (neuronal
    activity) and the assciated number of spatial bins ncats1. It then creates
    ncats categories for neuronal acitivity and assigns each datapoint to one
    of them. In var2 the running speed, divided in two states: resting and
    running state (ncats2 = 2). The programme then outputs a pointwise
    mutual-information matrix and displays it. If vectors are of different
    sizes, the shortest one isoversampled to match the length of the other.

    ...........................................................................

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    neuronal_trace    1-D variable of lenght L1 (dF/F, only one ROI)
    speed             1-D variable of length L2 (speed)
    ncats1            number of categories for dF
    ncats2            number of categories for speed (in our case ncats2 = 2,
                                                    running and resting states)
    settings          all parameters
    withlog           log the PMI or not

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    PMI       a ncats1 * ncats2 matrix with point-mutual information

    ...........................................................................

                                  FORMULA

           PMI(A,B) = p(A,B)/(p(A)p(B)), possibly with a log.

    - Here, the probability of any event A, p(A), would be its frequency of
    occurence in the input data.
    - (A,B) is the intersection of events A and B.

    ...........................................................................
    """

#    # if speed and neuronal data have different length, we align speed on the
#    # neuronal trace
#    L1, L2 = len(neuronal_trace), len(speed)
#    
#    if L1 != L2:
#        neuronal_trace_bis = neuronal_trace.reshape((1, len(neuronal_trace)))
#        speed = align_trace(speed, neuronal_trace_bis, method = 'linear')
#        del neuronal_trace_bis
#        
#    L = L1
#    del L1, L2
    
    L = len(neuronal_trace)

    # Finding category boundaries for each 1D vector based on (ncats)

    # ------- var1 : dF -------------------------------------------------------
    min1, max1 = np.min(neuronal_trace), np.max(neuronal_trace)
    dcat = (max1 - min1)/ncats1
    cats1 = [(i+1)*dcat + min1 for i in range(ncats1)]

    # ------- var2 : speed ----------------------------------------------------
    # NB : we can change this function to have more categories inside the speed
    cats2 = [settings['speed threshold'], max(speed)]

    # Assigning labels to each data point of the vectors
    # At the end of this part, we have two 1D vectors of size L where label(i)
    # is the category ([1:ncat]) to which var(i) belongs.

    labels1 = np.zeros(L)
    labels2 = np.zeros(L)

    # ------- var1 : dF -------------------------------------------------------

    for i in range(L):
        cat = 0
        while (cat < ncats1 - 1) and (neuronal_trace[i] > cats1[cat]):
            cat += 1
        labels1[i] = cat

    # ------- var2 : speed ----------------------------------------------------

    for i in range(L):
        cat = 0
        while (cat < ncats2 - 1) and speed[i] > cats2[cat]:
            cat += 1
        labels2[i] = cat


    # PMI
    PMI = np.zeros((ncats1,ncats2))

    for i in range(L):
        lb1, lb2 = int(labels1[i]), int(labels2[i])
        PMI[lb1,lb2] += 1
    PMI /= L

    elements1 = collections.Counter(labels1)
    elements2 = collections.Counter(labels2)

    for k in range(ncats1):
        P1 = elements1[k] / L

        for j in range(ncats2):
            P2 = elements2[j] / L

            if P1 > 0 and P2 > 0:
                PMI[k,j] = PMI[k,j]/(P1*P2)

                if withlog == True:
                    PMI[k,j] = math.log1p(PMI[k,j]);

                else:
                    pass

            else:
                PMI[k,j] = 0

    PMI /= np.max(PMI)

    return PMI


def get_pmi(dF, speed, settings, ncats1=10, ncats2=2, nb_of_clusters = 2):
    
    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................

    This function compute the PMI and different index using it on the all 
    dataset (dF/F). For each ROI, it then creates ncats1 categories for
    neuronal acitivity and assigns each datapoint to oneof them. The running 
    speed is divided in two states: resting and running state (ncats2 = 2).
    The programme then outputs a pointwise mutual-information matrix and 
    displays it for each neuron. It will also computes modulation indexes, 
    k-means and percentage of positively/negatively modulated cells.

    ...........................................................................

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    dF                dF/F for one or several ROIs 
    speed             speed trace
    settings          all parameters
    ncats1            number of categories for dF
    ncats2            number of categories for speed (in our case ncats2 = 2,
                                                    running and resting states)
    nb_of_clusters    number of clusters for the k-mean computation

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    PMI_all           matrix with point-mutual information for each ROI
    coor_rest         coordinates in resting state (mean pmi for the highest 
                      fluorescence bins divided by mean pmi in resting state)
    coor_run          coordinates in running state (mean pmi for the highest 
                      fluorescence bins divided by mean pmi in running state)
    mod_index_pmi     coor_run - coor_rest / coor_run + coor_rest
    pos_pmi           percentage of positively modulated cell (computed on 
                      mod_index_pmi)
    neg_pmi           percentage of negatively modulated cell (computed on 
                      mod_index_pmi)
    neutral_pmi       percentage of non modulated cell (computed on
                      mod_index_pmi)
    pos_pmi_k         percentage of positively modulated cell computed on 
                      k-mean
    neg_pmi_k         percentage of negatively modulated cell computed on 
                      k-mean
    label_states      for each ROI, number of the cluster (using k-mean)

    ...........................................................................
    """
    
    NbOfROI = len(dF)
    
    PMI_all = np.zeros((NbOfROI, ncats1, ncats2))
    for i in range(NbOfROI):
        pmi = PMI(dF[i], speed, settings, ncats1 = 10)
        PMI_all[i,:,:] = pmi
      
        
    # Compute the coordinates in each state (running and resting states). The 
    # coordinates in the 2 axes are the values of the PMI in the 5 highest bins
    # of fluorescence normalized by the probability of the entire state. 
    coor_rest, coor_run = np.zeros(NbOfROI), np.zeros(NbOfROI)
    for i in range(NbOfROI):
        coor_rest[i], coor_run[i] = \
        np.mean(PMI_all[i,round(0.4*ncats1):,:], axis=0)/np.mean(PMI_all[i,:,:])
        
    # compute the modulation index using the PMI, for each ROI
    # mod_index_pmi = [(m-s)/(m+s) for m, s in zip(coor_run, coor_rest)]
    mod_index_pmi = (coor_run - coor_rest) / (coor_run + coor_rest) 
    
    
    # percentages for each cell groups

    # positively modulated cells
    pos_pmi = sum((mod_index_pmi > 0.9)*1) / len(mod_index_pmi) * 100
    # negatively modulated cells
    neg_pmi = sum((mod_index_pmi < -0.9)*1) / len(mod_index_pmi) * 100
    # non modulated cells
    neutral_pmi = 100 - pos_pmi - neg_pmi
    
    # percentages for each cell groups using kmeans
    # kmeans: clustering the cells in three groups: positively, negratively and
    # non modulated by runnign speed (I don't use it anymore but just in case)
    coor_rest_run = np.column_stack((coor_rest, coor_run))
    kmeans = KMeans(n_clusters=min(NbOfROI,nb_of_clusters)).fit(coor_rest_run)

    label_states = kmeans.labels_

    # percentage of modulated cells

    #positively modulated cells
    pos_pmi_k = sum(label_states) / len(label_states) * 100
    # negatively modulated cells
    neg_pmi_k = 100 - pos_pmi_k
    
    return(PMI_all, coor_rest, coor_run, mod_index_pmi, pos_pmi, neg_pmi, 
           neutral_pmi, pos_pmi_k, neg_pmi_k, label_states)
    
    
def compute_synchrony_ES(x, y, setting, w_size = 1, method = 'constant', 
                         derivative = False):
    
    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................

    This function takes two lists x and y of size t (datapoints acquired at a
    given sampling freq.) with binary values, where 1s occur at times of 
    events. These can be peaks or spikes, or anything else you want. This
    algorithm was written fully based on Quiroga et al. (2008) so copyright 
    goes to them. 

    ...........................................................................

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    x                 vector of size t with mx events 
    y                 vector of size t with my events 
    setting           all parameters
    w_size            limit number of frames to consider 2 events as 
                      synchronous
    'derivative'      if you want to compute time-resolved synchrony

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    Q                 synchrony measure, Q \in [0,1]
    q                 delay behavior, q \in [0,1]
                      q = 1 => events in x always precede those in y
    Qprime            time-resolved synchrony with window size dn
    qprime            time-resolved antisynchrony (causality)
    tau_min           minimal tau ever defined

    ...........................................................................
    """

    ## Q (symmetry)       
    
#    th_x = spiking_analysis.threshold(x, settings=setting, method=method_th)
#    th_y = spiking_analysis.threshold(y, settings=setting, method=method_th)
#    tx = [i for i,e in enumerate(x) if e > th_x]
#    ty = [i for i,e in enumerate(y) if e > th_y]
    
    tx = [i for i,e in enumerate(x) if e > 0]
    ty = [i for i,e in enumerate(y) if e > 0]

    mx = len(tx) # nb of x events
    my = len(ty) # nb of y events
    

    if method == 'constant':
        c_xy, _ = get_synchrony_ES_constant(tx, ty, w_size)
        c_yx, _ = get_synchrony_ES_constant(ty, tx, w_size)
        tau_min = w_size
        
    elif method == 'ES':
        c_xy,tau_min_xy, _ = get_synchrony_ES(tx, ty, w_size)
        c_yx,tau_min_yx, _ = get_synchrony_ES(ty, tx, w_size)

        tau_min = min(tau_min_xy,tau_min_yx)
        
    else:
        pass
    
    
    if mx!=0 and my!=0:
        # simple and straightforward
        Q = (c_xy + c_yx)/np.sqrt(mx*my)  
        q = (c_yx - c_xy)/np.sqrt(mx*my)
    else:
        Q, q = 0, 0
        
    
    ## Q'(n)
    if derivative == True:

        dn = 5
        dp = int(len(x)/dn) # nb of points per window
        aside = len(x)%dn # but this has to be added to the last one
        windows = [0 for i in range(dp+1)]
        windows[0] = 1
        
        for w in range(1,dp+1):
            windows[w] = w*dn + (w==dp)*1*aside
            
        Qval = [0 for i in range(dp)]
        qval = [0 for i in range(dp)]
        
        for w in range(dp):
            #now we find synchroneous events up
            c_xy, _, _ = get_synchrony_ES(tx, ty, windows[w+1]) 
            # to t = windows(w+1).
            c_yx, _, _ = get_synchrony_ES(ty, tx, windows[w+1])
            Qval[w] = c_xy + c_yx
            qval[w] = c_xy - c_yx

        Qprime = [0 for i in range(dp-2)]
        qprime = [0 for i in range(dp-2)]
        
        for w in range(dp-2):
            
            # local derivative
            dQ = Qval[w+1] - Qval[w] 
            qprime[w] = qval[w+1] - qval[w]
            
            dnx = len([i for i,e in enumerate(x[windows[w+1]:windows[w+2]]) if e > th_x]) 
            dny = len([i for i,e in enumerate(y[windows[w+1]:windows[w+2]]) if e > th_y]) 
            # finding number of events between the 2 Qvals that we compare
            
            if dnx*dny != 0:
                # if dividing is an option
                Qprime[w] = dQ/np.sqrt(dnx*dny) 
            else:
                Qprime[w] = 0
                
    else:
        Qprime = []
        qprime = [] # if the derivative was not asked for
        
    return (Q, q, Qprime, qprime, tau_min)


def get_synchrony_ES_constant(tx, ty, w_size = 1, use_step = False, *n):
    
    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr
    
    ...........................................................................

    This function takes two lists tx and ty with the index of the spikes. It 
    will return the number of spikes in x preceded by a spike in y in a certain
    window. This window is fixed at 1 frame by default. Avoid changing it for 
    bi-photon data. 
    
    /!\ this measure is not bilateral 
        get_synchrony_ES(x,y) != get_synchrony_ES(y,x)
        
    ...........................................................................

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    tx                vector of size mx with the index of events in x 
    ty                vector of size my with the index of events in y 
    w_size            limit number of frames to consider 2 events as 
                      synchronous (by default = 1)
    'use_step'        check the step

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    cxy               number of spikes in x preceded by a spike in y
    is_synchro        index of synchronous spikes in x

    ...........................................................................
    """
    
    mx, my = len(tx), len(ty)
    cxy = 0 
    is_synchro = np.zeros(mx)
        
    if mx > 1 and my > 1:
        
        for i in range(mx):
            J_ij = 0
            
            # for j in range(my):
            for j,_ in enumerate([v for v in ty if v<=tx[i]]):
                
                dt = tx[i] - ty[j]
                
                if dt > 0 and dt < w_size:
                    J_ij = 1
                elif dt == 0:
                    J_ij = 1/2
                else:
                    J_ij = 0     
                    
            
                if use_step == True:
                    step = (n-tx[i] > 0)*1
                else:
                    step = 1
                    
                
                cxy += J_ij*step
                
                if J_ij > 0:
                    is_synchro[i] = J_ij*step
    
    # if one of the 2 is null         
    else: 
        cxy = 0
        
    return(cxy, is_synchro)


def get_synchrony_ES(tx, ty, w_size, use_step = False, *n):
    
    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr
    
    ...........................................................................

    This function takes two lists tx and ty with the index of the spikes. It 
    will return the number of spikes in x preceded by a spike in y in a certain
    window. This window is changing to avoid counting a spike twice. 
    
    /!\ this measure is not bilateral 
        get_synchrony_ES(x,y) != get_synchrony_ES(y,x)
        
    ...........................................................................

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    tx                vector of size mx with the index of events in x 
    ty                vector of size my with the index of events in y 
    w_size            limit number of frames to consider 2 events as 
                      synchronous
    'use_step'        check the step

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    cxy               number of spikes in x preceded by a spike in y
    tau_min           minimal tau ever defined
    is_synchro        index of synchronous spikes in x

    ...........................................................................
    """

    tau_min = w_size
    mx, my = len(tx), len(ty)
     
    cxy = 0
    is_synchro = np.zeros(mx)
    
        
    if mx == 1 and my == 1: 
        
        # interspike interval
        dt = tx[0] - ty[0]
        tau = w_size
        
        # compute J_ij
        if dt > 0 and dt <= tau:
            J_ij = 1
        elif dt == 0:
            J_ij = 1/2
        else:
            J_ij = 0
            
        ## vérifier la step
        if use_step == True:
            step = (n-tx[0] > 0)*1
        else:
            step = 1
                
        cxy += J_ij*step
        
        is_synchro[0] = J_ij*step
        
     
    elif mx == 1 and my > 1:
        i, all_i  = 0, []
         
        for j,_ in enumerate([v for v in ty if v<=tx[i]]): # and v>=max((tx[i]-w_size),0)]):
             
            # adapting the window size to the spike
            if j==0:
                all_j = [ty[j+1] - ty[j]]
            elif j==my-1:
                all_j = [ty[j] - ty[j-1]]
            else:
                all_j = [ty[j+1] - ty[j], ty[j] - ty[j-1]]
                 
            ALL = all_i + all_j
            tau = min(w_size, min(ALL)/2)
            if tau < tau_min:
                tau_min = tau
             
            # interspike interval
            dt = tx[i] - ty[j]
            
            # compute J_ij
            if dt > 0 and dt <= tau:
                J_ij = 1
            elif dt == 0:
                J_ij = 1/2
            else:
                J_ij = 0
            print(J_ij)
                
            # check the step
            if use_step == True:
                step = (n-tx[i] > 0)*1
            else:
                step = 1
                
            cxy += J_ij*step
              
            is_synchro[i] = J_ij*step
            
            
    elif mx > 1 and my == 1:
        j, all_j  = 0, []
        
        for i,_ in enumerate([v for v in tx if v<=ty[j]]): # and v>=max((ty[j]-w_size),0)]):
             
            # adapting the window size to the spike
            if i==0:
                all_i = [tx[i+1] - tx[i]]
            elif i==mx-1:
                all_i = [tx[i] - tx[i-1]]
            else:
                all_i = [tx[i+1] - tx[i], tx[i] - tx[i-1]]
                
            ALL = all_i + all_j
            tau = min(w_size, min(ALL)/2)
            if tau < tau_min:
                tau_min = tau
            
            # interspike interval
            dt = tx[i] - ty[j]
            
            # compute J_ij
            if dt > 0 and dt <= tau:
                J_ij = 1
            elif dt == 0:
                J_ij = 1/2
            else:
                J_ij = 0
                
            # check the step
            if use_step == True:
                step = (n-tx[i] > 0)*1
            else:
                step = 1
                
            cxy += J_ij*step
            
            is_synchro[i] = J_ij*step
    
        
    elif mx > 1 and my > 1:
        
        for i in range(mx):
            
            # adapting the window size to the spike
            if i==0:
                all_i = [tx[i+1] - tx[i]]
            elif i==mx-1:
                all_i = [tx[i] - tx[i-1]]
            else:
                all_i = [tx[i+1] - tx[i], tx[i] - tx[i-1]]
                 
            for j,_ in enumerate([v for v in ty if v<=tx[i]]): #  and v>=max((tx[i]-w_size),0)]):
                
                # adapting the window size to the spike
                if j==0:
                    all_j = [ty[j+1] - ty[j]]
                elif j==my-1:
                    all_j = [ty[j] - ty[j-1]]
                else:
                    all_j = [ty[j+1] - ty[j], ty[j] - ty[j-1]]
                    
                ALL = all_i + all_j
                tau = min(w_size, min(ALL)/2)
                if tau < tau_min:
                    tau_min = tau
                
                # interspike interval
                dt = tx[i] - ty[j]
            
                # compute J_ij
                if dt > 0 and dt <= tau:
                    J_ij = 1
                elif dt == 0:
                    J_ij = 1/2
                else:
                    J_ij = 0
                
                # check the step
                if use_step == True:
                    step = (n-tx[i] > 0)*1
                else:
                    step = 1
                
                cxy += J_ij*step
                
                if J_ij > 0:
                    is_synchro[i] = J_ij*step
                    
    # if one of the 2 is null          
    else: 
        cxy = 0
        
    return(cxy, tau_min, is_synchro)
            

def compute_ISI(spikes, thr_spikes, settings):
    
    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................

    This function takes two lists x and y of size t (datapoints acquired at a
    given sampling freq.) with binary values, where 1s occur at times of 
    events. We compute the Instantaneous firing rate as the interspike 
    interval. This algorithm was written based on Politi et al. (2007) so
    copyright goes to them. 

    ...........................................................................

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    spikes            matrix of size roi*t with roi*mx events 
    thr_spikes        vector of size roi with thresholds 
    setting           all parameters

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    spikes_ISI        ISI for all traces

    ...........................................................................
    """
    
    NbOfPoints = spikes.shape[1]
    time = np.linspace(0, settings['time_seconds'], NbOfPoints)
    spikes_ISI = np.copy(spikes)
    
    for roi in range(spikes.shape[0]):
        
        x = spikes[roi]
        tx = [0] + [i for i,e in enumerate(x) if e > thr_spikes[roi]] + [NbOfPoints-1]
    
#       tx = [0] + [i for i,e in enumerate(x) if e > 0] + [NbOfPoints-1]
#       time_x = [0] + [time[i] for i,e in enumerate(x) if e > 0] + [settings['time_seconds']]

        X_isi = np.zeros(NbOfPoints)

        # compute the instantaneous firing rates
        for t in range(NbOfPoints-1):        
            X_isi[t] = min(time[i] for i in tx if i>t) - max(time[i] for i in tx if i<=t)
        X_isi[-1] = X_isi[-2] 
            
        spikes_ISI[roi] = X_isi        
    
    return spikes_ISI, 1/spikes_ISI
            
            

def compute_synchrony_ISI(x, y, setting):
    
    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................

    This function takes two lists x and y of size t (datapoints acquired at a
    given sampling freq.) with binary values, where 1s occur at times of 
    events. With the ISI-distance, we extract information from the interspike 
    intervals by evaluating the ratio of the instantaneous firing rates. This
    algorithm was written fully based on Politi et al. (2007) so copyright 
    goes to them. 

    ...........................................................................

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    x                 vector of size t with mx events 
    y                 vector of size t with my events 
    setting           all parameters

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    D                 ISI-distance
    X_isi             instantaneous firing rate in X
    Y_isi             instantaneous firing rate in Y

    ...........................................................................
    """
    
#    th_x = spiking_analysis.threshold(x, settings=setting, method=method_th, 
#                                      constant = 0.8)
#    th_y = spiking_analysis.threshold(y, settings=setting, method=method_th, 
#                                      constant = 0.8)
#
#    NbOfPoints = len(x)
#    
#    tx = [0] + [i for i,e in enumerate(x) if e > th_x] + [NbOfPoints]
#    ty = [0] + [i for i,e in enumerate(y) if e > th_y] + [NbOfPoints]

    NbOfPoints = len(x)
    tx = [0] + [i for i,e in enumerate(x) if e > 0] + [NbOfPoints]
    ty = [0] + [i for i,e in enumerate(y) if e > 0] + [NbOfPoints]
    
    X_isi, Y_isi = np.zeros(NbOfPoints), np.zeros(NbOfPoints)
    I = np.zeros(NbOfPoints)

    # compute the instantaneous firing rates
    for t in range(NbOfPoints):        
        x_isi = min(i for i in tx if i>t) - max(i for i in tx if i<=t)
        y_isi = min(i for i in ty if i>t) - max(i for i in ty if i<=t)
       
        X_isi[t] = x_isi
        Y_isi[t] = y_isi
        
        if x_isi < y_isi:
            I[t] = x_isi/y_isi -1 
        else:
            I[t] = - (y_isi/x_isi - 1)
           
    # compute the ISI-distance
    # time-weighted variant, the ISI-distance is intergated over time
    time = np.linspace(0, setting['time_seconds'], len(x))
    D = np.trapz(abs(I), time)
    
    return(D, X_isi, Y_isi)    
    
    
def compute_synchrony_STTC(x, y, setting, w_size):
    
    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................

    This function computes the synchronicity between two lists x and y of size 
    t (datapoints acquired at a given sampling freq.) with binary values, where
    1s occur at times of events. This function compute the spike time tilling 
    coefficient (spikes in x with fall within +/-dt of a spike from B). This 
    algorithm was written fully based on Cutts & Eglen (2014) so copyright goes
    to them. 

    ...........................................................................

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    x                 vector of size t with mx events 
    y                 vector of size t with my events 
    setting           all parameters
    w_size            limit number of frames to consider 2 events as 
                      synchronous

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    STTC              Spike Time Tilling Coefficient
    
    ...........................................................................

                                  FORMULA

           STTC = ((Pa-Tb)/(1-Pa*Tb) + (Pb-Ta)/(1-Pb*Ta)) / 2

    - Ta, Tb : density of spikes in  x, y
    - Pa, Pb : percentage of spikes in x which fall within +/-dt of a spike 
               from y

    ...........................................................................
    """

    w_size = int(w_size/2)
    
#    th_x = spiking_analysis.threshold(x, settings=setting, method=method_th)
#    th_y = spiking_analysis.threshold(y, settings=setting, method=method_th)   
#    x_spikes = (x > th_x)*1
#    y_spikes = (y > th_y)*1

    tx = [i for i,e in enumerate(x) if e > 0]
    ty = [i for i,e in enumerate(y) if e > 0]

    # nb of x and y events   
    mx, my = len(tx), len(ty)   
    
    
    x_extend = np.copy(x)
    for i in tx:
        if i < w_size:
            x_extend[0:i+w_size] = [1 for j in range(i+w_size)]
        elif i > len(x)-w_size-1:
            x_extend[i-w_size:-1] = [1 for j in range(w_size+len(x)-i-1)]
        else:
            x_extend[i-w_size:i+w_size] = [1 for j in range(2*w_size)]
       
    
    y_extend = np.copy(y)
    for j in ty:
        if j < w_size:
            y_extend[0:j+w_size] = [1 for i in range(j+w_size)]
        elif j > len(y)-w_size-1:
            y_extend[j-w_size:-1] = [1 for i in range(w_size+len(x)-j-1)]
        else:
            y_extend[j-w_size:j+w_size] = [1 for i in range(2*w_size)]
            
            
    Ta, Tb = sum(x_extend)/len(x), sum(y_extend)/len(x)
    Pa, Pb = sum((x*y_extend)*1)/mx, sum((y*x_extend)*1)/my
    
    STTC = ((Pa-Tb)/(1-Pa*Tb) + (Pb-Ta)/(1-Pb*Ta)) / 2
    
    return(STTC)        

    
def synchrony(dataset, settings, th, method='cross corr', w_size=1, 
              derivative=False):
    
    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................

    This function will compute the synchronicity inside a data set of n 
    neurons, using the deconvolved trace (amplitude and not 0 and 1). The 
    synchronicity is computed between each neuron so we get a matrix of 
    correlation neuron per neuron. The user can use different method to compute 
    this synchronicity.

    ...........................................................................

    - - - - - - - - - - - - - - - METHODS - - - - - - - - - - - - - - - - - - -

    'cross corr'      cross correlation (by default)  
                      /!\ computed on dF/F 
    'ES constant'     event synchronicity using a constant window (1 frame)
    'ES'              event synchronicity using a changing window
    'ISI'             Inter-Spike Interval
    'STTC'            Spike Time Tilling Coefficient

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    dataset           dataset of deconvolved firing trace (several ROIs) or dF 
    settings          (use of the final "sampling frequency)
    th                threshold for each ROI
    method            'cross corr', 'ES constant', 'ES', 'ISI' or 'STTC'
                      ('constant' by default)
    w_size            limit number of frames to consider 2 events as 
                      synchronous (1 by default)
    'derivative'      if you want to compute time-resolved synchrony

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    synchro           matrix of synchronicity (neuron per neuron)

    ...........................................................................
    """
    
    NbOfROI = len(dataset)
    NbOfPoints = len(dataset[0])
    synchro = np.eye(NbOfROI)
    synchro[:] = np.NaN

    for i in range(NbOfROI):
        for j in range(i+1, NbOfROI):
            
            if method == 'cross corr':
                Q = np.correlate(dataset[i], dataset[j])/NbOfPoints
                
            elif method == 'ES constant':
                x, y = (dataset[i]>th[i])*1, (dataset[j]>th[j])*1
                Q, _, _, _, _ = compute_synchrony_ES(x, y, settings, w_size,
                                                     method='constant')
            
            elif method == 'ES':
                x, y = (dataset[i]>th[i])*1, (dataset[j]>th[j])*1
                Q, _, _, _, _ = compute_synchrony_ES(x, y, settings, w_size, 
                                                     method='ES')
                
                
            elif method == 'ISI':
                D, _, _ = compute_synchrony_ISI(dataset[i], dataset[j], 
                                                settings, method_th)
                Q = D
                
            elif method == 'STTC':
                x, y = (dataset[i]>th[i])*1, (dataset[j]>th[j])*1
                Q = compute_synchrony_STTC(x, y, settings, w_size)
            
            else:
                Q = 0
                
            synchro[i,j], synchro[j,i] = Q, Q 
       
#        /!\ absolutly not working for now !      
#        start=datetime.now() 
#        if method == 'ISI':
#            pool = mp.Pool(mp.cpu_count())
#            results = [pool.apply(compute_synchrony_ISI, args=(dataset[i], dataset[j], settings, method_th, w_size)) for j in range(i+1, NbOfROI)]
#            pool.close()
#        print(datetime.now() -  start) 
            
    return(synchro)           
            
            
def test_synchrony(x,y, w_size, settings, method):
    
    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................

    This function takes two lists x and y of size t (datapoints acquired at a
    given sampling freq.) with binary values, where 1s occur at times of 
    events. The user can use different method to compute this synchronicity.

    ...........................................................................

    - - - - - - - - - - - - - - - METHODS - - - - - - - - - - - - - - - - - - -

    'cross corr'      cross correlation  
                      /!\ normally computed on dF/F => not so usefull
    'ES constant'     event synchronicity using a constant window (1 frame)
    'ES'              event synchronicity using a changing window
    'ISI'             Inter-Spike Interval
    'STTC'            Spike Time Tilling Coefficient

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    x, y              firing traces (binary values)  
    w_size            limit number of frames to consider 2 events as 
                      synchronous 
                      /!\ with 'ES constant', use w_size = 1
    settings          all parameters
    method            'cross corr', 'ES constant', 'ES', 'ISI' or 'STTC'
                      ('constant' by default)

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    /!\ not all values will be meaningfull for all methods

    Q                 synchrony measure, Q \in [0,1]
    q                 delay behavior, q \in [0,1]
                      q = 1 => events in x always precede those in y
    c_xy, c_yx        number of spikes in x, y preceded by a spike in y, x
    tau_min           minimal tau ever defined
    D                 ISI-distance
    is_synchro_xy, yx index of synchronous spikes in x, y
    
    ...........................................................................
    """
    
    ## x, y : trains of 0 and 1 
    
    if method == 'ES':
        
        tx = [i for i,e in enumerate(x) if e > 0]
        ty = [i for i,e in enumerate(y) if e > 0]

        mx = len(tx) # nb of x events
        my = len(ty) # nb of y events
        
        c_xy,tau_min_xy, is_synchro_xy = get_synchrony_ES(tx, ty, w_size)
        c_yx,tau_min_yx, is_synchro_yx = get_synchrony_ES(ty, tx, w_size)

        tau_min = min(tau_min_xy,tau_min_yx)
    
        if mx!=0 and my!=0:
            # simple and straightforward
            Q = (c_xy + c_yx)/np.sqrt(mx*my)  
            q = (c_yx - c_xy)/np.sqrt(mx*my)
        else:
            Q, q = 0, 0
            
        D = 0
        
        
    elif method == 'ES constant':
        
        tx = [i for i,e in enumerate(x) if e > 0]
        ty = [i for i,e in enumerate(y) if e > 0]

        mx = len(tx) # nb of x events
        my = len(ty) # nb of y events
        
        c_xy, is_synchro_xy = get_synchrony_ES(tx, ty)
        c_yx, is_synchro_yx = get_synchrony_ES(ty, tx)
    
        if mx!=0 and my!=0:
            # simple and straightforward
            Q = (c_xy + c_yx)/np.sqrt(mx*my)
            q = (c_yx - c_xy)/np.sqrt(mx*my)
        else:
            Q, q = 0, 0
            
        D, tau_min = 0, 1
            
    
    elif method == 'ISI':
        
        NbOfPoints = len(x)
    
        tx = [0] + [i for i,e in enumerate(x) if e > 0] + [NbOfPoints]
        ty = [0] + [i for i,e in enumerate(y) if e > 0] + [NbOfPoints]
    
        I = np.zeros(NbOfPoints)

        # compute the ISI
        for t in range(NbOfPoints):        
            x_isi = min(i for i in tx if i>t) - max(i for i in tx if i<=t)
            y_isi = min(i for i in ty if i>t) - max(i for i in ty if i<=t)
        
            if x_isi < y_isi:
                I[t] = x_isi/y_isi -1 
            else:
                I[t] = - (y_isi/x_isi - 1)
            
        time = np.linspace(0, settings['time_seconds'], len(x))
        D = np.trapz(abs(I), time)
        Q, q, tau_min, c_xy, c_yx = 0, 0, 0, 0, 0 
        is_synchro_xy, is_synchro_yx = [], []
        
    
    elif method == 'STTC':
        tx = [i for i,e in enumerate(x) if e > 0]
        ty = [i for i,e in enumerate(y) if e > 0]

        mx, my = len(tx), len(ty) # nb of x events # nb of y events     
        NbOfPoints = len(x)
    
    
        x_extend, y_extend = np.copy(x), np.copy(y)
        
        for i in tx:
            if i < w_size:
                x_extend[0:i+w_size] = [1 for j in range(i+w_size)]
            elif i > len(x)-w_size-1:
                x_extend[i-w_size:-1] = [1 for j in range(w_size+len(x)-i-1)]
            else:
                x_extend[i-w_size:i+w_size] = [1 for j in range(2*w_size)]
       
        for j in ty:
            if j < w_size:
                y_extend[0:j+w_size] = [1 for i in range(j+w_size)]
            elif j > len(y)-w_size-1:
                y_extend[j-w_size:-1] = [1 for i in range(w_size+len(x)-j-1)]
            else:
                y_extend[j-w_size:j+w_size] = [1 for i in range(2*w_size)]
            
            
        Ta, Tb = sum(x)/NbOfPoints, sum(y)/NbOfPoints
        Pa, Pb = sum((x*y_extend)*1)/mx, sum((y*x_extend)*1)/my
        is_synchro_xy, is_synchro_yx = (x*y_extend)*1, (y*x_extend)*1
    
        Q = ((Pa-Tb)/(1-Pa*Tb) + (Pb-Ta)/(1-Pb*Ta)) / 2
        q, tau_min, D, c_xy, c_yx = 0, 0, 0, 0, 0
        
        
    elif method == 'cross corr':
        print('Attention, pas de de sens avec cette trace')
        Q = np.correlate(x, y)/len(x)
        q, tau_min, D, c_xy, c_yx = 0, 0, 0, 0, 0
        is_synchro_xy, is_synchro_yx = [], []
        
    
    else:
        pass
        
    return(Q, q, c_xy, c_yx, tau_min, D, is_synchro_xy, is_synchro_yx)
    
    
def nb_firing_cells(spikes, w_size, method = 'sliding'):

    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr

    ...........................................................................
    
    This function takes all firing traces of size t (datapoints acquired at a
    given sampling freq.) with binary or thresholded values, where 1s occur at 
    times of events. These can be peaks or spikes, or anything else you want. 
    This algorithm count the number of cells firing in a certain window.

    ...........................................................................
    
    - - - - - - - - - - - - - - - METHODS - - - - - - - - - - - - - - - - - - -

    'sliding'         using a sliding window (by default)
    'jumping'         using a jumping window (binning the trace)

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    spikes            all firing traces
    w_size            duration of the sliding window 
    method            'sliding' or 'jumping'

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - - 

    Nb_firing         proportion of cells firing in the window at each point

    ...........................................................................
    """

    one_row = len(spikes.shape)
    if one_row == 1 :
        trace = spikes.reshape((1, len(spikes)))   

    if method == 'sliding':
        Nb_firing = np.zeros(spikes.shape[1])
        for i in range(spikes.shape[1]-w_size-1):
            Nb_firing[i] = sum((np.sum(spikes[:, i:i+w_size], axis=1) > 0)*1)
            
    elif method == 'jumping':
        Nb_firing = np.zeros(int(spikes.shape[1]/w_size))
        ind = 0
        for i in range(0, spikes.shape[1]-w_size-1, w_size):
            Nb_firing[ind] = sum((np.sum(spikes[:, i:i+w_size], axis=1) > 0)*1)
            ind += 1
            
    else:
        pass

    return Nb_firing/spikes.shape[0]
    
def create_cell(n_points, n_spikes):
    """
    Create synthetic array of dimension 'n_points' and with 'n_spikes' number
    of spikes.
    Array consists of 0s and 1s.
    Made to simulate a spiking trace.

    Marcel van Velze, m.debritovanvelze@icm-institute.org

    Parameters
    ----------
    n_points : Length of the desired array;
    n_spikes : Number of spikes in array.

    Returns
    -------
    cell : Synthetic spiking trace.

    """
    spikes = random.sample(range(int(n_points * 0.1), int(n_points * 0.9)), n_spikes)
    cell = np.zeros(n_points)
    for i in spikes:
        cell[i] = 1
    return cell    
   
    
def make_offset(array, offset):
    """
    Offset data by n values.
    Adds n ammount of zeros to the opposite side of the shift.
    Final array is same size as original.
    
    Marcel van Velze, m.debritovanvelze@icm-institute.org

    Parameters
    ----------
    array : One-dimentional array of data;
    offset : Number of points to offset.

    Returns
    -------
    shifted_array : Array with same dimentions as 'array' but shifted n units.

    """
    if offset > 0:
        shifted_array = np.pad(array, (offset, 0), 'constant')
        shifted_array = shifted_array[0:len(array)]
    elif offset < 0:
        shifted_array = np.pad(array, (0, abs(offset)), 'constant')
        shifted_array = shifted_array[abs(offset)-1:-1]
    else:
        shifted_array = array
    return shifted_array
    
def linreg(bin_speed, bin_traces):
    """
    Performs a simple linear regression with the given data
    
    Marcel van Velze, m.debritovanvelze@icm-institute.org

    Parameters
    ----------
    bin_speed : One dimensional array containing speed data;
    bin_traces : One or multidimentional array with data to be fitted;

    Returns
    -------
    r_sq : Confidence of the linear regression (from 0 - 1);
    slope : Slope of the linear regression;
    intercept : Interception at y axis.

    """
    x = np.copy(bin_speed).reshape((-1, 1))
    r_sq = []
    slope = []
    intercept = []
    for i in range(0, bin_traces.shape[0]):
        y = np.copy(bin_traces[i])
        model = LinearRegression().fit(x, y)
        r_sq.append(model.score(x, y))
        slope.append(model.coef_[0])
        intercept.append(model.intercept_)
    return r_sq, slope, intercept

def lin_reg_moving(speed_data, dF_data, speed_threshold = 1):
    """
    Removes the values where speed is lower than speed threshold and calculates the linear regression
    
    Marcel van Velze, m.debritovanvelze@icm-institute.org

    Parameters
    ----------
    speed_data : One dimensional array containing speed data
    dF_data : One dimensional or multidimentional array of fluorescence data
    speed_threshold : float64, optional
        Threshold speed above which data will be excluded. The default is 1.

    Returns
    -------
    r_sq : Confidence of the linear regression (from 0 - 1);
    slope : slope of the linear regression;
    intercept : interception at y axis.

    """
    speed = np.squeeze(np.copy(speed_data).reshape((-1, 1)))
    
    if np.size(speed) != np.size(dF_data,1):
        print('Problem: Datasets are not the same size!')
    else:
        r_sq = []
        slope = []
        intercept = []
    
        selection = speed > speed_threshold
        s_speed = speed[selection]
        s_data = dF_data[:,selection]
        
        x = s_speed.reshape((-1, 1))
        if len(x) > 0:
            for i in range(0, s_data.shape[0]):
                y = np.expand_dims(np.copy(s_data[i]), 1)
                model = LinearRegression().fit(x, y)
                r_sq.append(model.score(x, y))
                slope.append(model.coef_[0][0])
                intercept.append(model.intercept_)  
            return r_sq, slope, intercept 
        else:
            return [], [], []
            
def linearize_data(data_dictionary):
    """
    Concatenate all speed and fluorescence data in dictionary 
    
    Marcel van Velze, m.debritovanvelze@icm-institute.org
    
    Parameters
    ----------
    data_dictionary : dictionary containing a series of keys each containing one speed file and one fluorescence file

    Returns
    -------
    combined_points : array with speed in row 0 and df in row 1
    
    Call function: combined_points = linearize_data(save_session.load_variable(easygui.fileopenbox(title='Select file containing speed vs fluo data (.pyckle)')))
    """
    combined_points = np.zeros((2,1))
    for key in data_dictionary:
        s = data_dictionary[key]['speed']
        fluo = data_dictionary[key]['dF']
        for i in range(0, len(fluo)):
            points = np.concatenate((np.expand_dims(s, 0), np.expand_dims(fluo[i],0)), axis=0)
            combined_points = np.concatenate((combined_points, points), axis=1)
            
    combined_points = combined_points[:, 1:-1]
    
    return combined_points


def plot_section(combined_points, min_speed, max_speed, bin_size=0.1):
    """
    Plots speed vs fluorescence in the defined speed ranges
    
    Marcel van Velze, m.debritovanvelze@icm-institute.org

    Parameters
    ----------
    combined_points : array with speed in row 0 and df in row 1. See function: linearize_data
    min_speed : an integer or 'full' if minimum speed desired.
    max_speed : an integer or 'full' if maximum speed desired.
    bin_size : size of bin. The default is 0.1.

    Returns
    -------
    Graph with section of data plotted and mean in designated bins.

    """
    
    if max_speed == 'full':
        max_speed = np.amax(combined_points[0])
    if min_speed == 'full':
        min_speed = np.amin(combined_points[0])
        
    selection = np.where(np.logical_and(combined_points[0]>=min_speed, combined_points[0]<=max_speed))
    
    selected_speed = np.array([combined_points[0,i] for i in selection[0]])
    selected_fluo = np.array([combined_points[1,i] for i in selection[0]])
    
    bins = np.arange(min_speed, max_speed, bin_size)
    digitized_speed = np.digitize(selected_speed, bins)
    bin_means = [selected_fluo[digitized_speed == i].mean() for i in range(1, len(bins))]
    bin_sem = [sem(selected_fluo[digitized_speed == i]) for i in range(1, len(bins))]
    #bin_means = (np.histogram(selected_fluo, bins, weights=selected_fluo)[0] / np.histogram(selected_fluo, bins)[0])
    
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6), sharex=True)
    fig.suptitle('Range: {} to {} cm/s'.format(min_speed, max_speed))
    ax1.set_ylabel('\u0394F/F0')
    ax1.set_xlabel('Speed (cm/s)')
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    #Red
    #ax1.scatter(selected_speed, selected_fluo, color = '#de2d26', alpha=0.4, marker='o', s=2)
    #Blue
    ax1.scatter(selected_speed, selected_fluo, color = '#3182bd', alpha=0.4, marker='o', s=2)
    ax1.plot(bins[0:len(bin_means)], bin_means, color='r')
    ax2.set_ylabel('\u0394F/F0')
    ax2.set_xlabel('Speed (cm/s)')
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.plot(bins[0:len(bin_means)], bin_means, color='r')
    ax2.fill_between(bins[0:len(bin_means)], [m_i + s_i for m_i, s_i in zip(bin_means, bin_sem)], [m_i - s_i for m_i, s_i in zip(bin_means, bin_sem)], color = 'gray', alpha = 0.2)
    #plt.tight_layout()
    plt.show()
    
    
def plot_2section(combined_points1, combined_points2, min_speed, max_speed, bin_size=0.1, normalize = False, n_start=3, n_end=5):
    """
    Plots speed vs fluorescence in the defined speed ranges
    
    Marcel van Velze, m.debritovanvelze@icm-institute.org

    Parameters
    ----------
    combined_points : array with speed in row 0 and df in row 1. See function: linearize_data
    min_speed : an integer or 'full' if minimum speed desired.
    max_speed : an integer or 'full' if maximum speed desired.
    bin_size : size of bin. The default is 0.1.

    Returns
    -------
    Graph with section of data plotted and mean in designated bins.

    """
    # blue dark: #3182bd
    # blue light: #deebf7


    # red dark: #de2d26
    # red light:#fee0d2


    # orange dark: #d95f0e
    # orange light: #fec44f
    
    
    color1 = '#3182bd'
    color2 = '#de2d26'
    label1 = 'VIP'
    label2 = 'VIPnr1' 
    
    if max_speed == 'full':
        max_speed = min([np.amax(combined_points1[0]), np.amax(combined_points2[0])])
    if min_speed == 'full':
        min_speed = max([np.amin(combined_points1[0]), np.amin(combined_points2[0])])
     
    bins = np.arange(min_speed, max_speed, bin_size)                     
    
    norm_select1 = np.where(np.logical_and(combined_points1[0]>=n_start, combined_points1[0]<=n_end))
    norm_val1 = np.mean(np.array([combined_points1[1,i] for i in norm_select1[0]]))
    norm_select2 = np.where(np.logical_and(combined_points2[0]>=n_start, combined_points2[0]<=n_end))
    norm_val2 = np.mean(np.array([combined_points2[1,i] for i in norm_select2[0]]))
    
    selection1 = np.where(np.logical_and(combined_points1[0]>=min_speed, combined_points1[0]<=max_speed))
    selected_speed1 = np.array([combined_points1[0,i] for i in selection1[0]])
    if normalize == True:
        selected_fluo1 = np.array([combined_points1[1,i] for i in selection1[0]])/norm_val1
    else:
        selected_fluo1 = np.array([combined_points1[1,i] for i in selection1[0]])
    digitized_speed1 = np.digitize(selected_speed1, bins)
    bin_means1 = [selected_fluo1[digitized_speed1 == i].mean() for i in range(1, len(bins))]
    bin_sem1 = [sem(selected_fluo1[digitized_speed1 == i]) for i in range(1, len(bins))]
    
    selection2 = np.where(np.logical_and(combined_points2[0]>=min_speed, combined_points2[0]<=max_speed))
    selected_speed2 = np.array([combined_points2[0,i] for i in selection2[0]])
    if normalize == True:
        selected_fluo2 = np.array([combined_points2[1,i] for i in selection2[0]])/norm_val2
    else:
        selected_fluo2 = np.array([combined_points2[1,i] for i in selection2[0]])
    digitized_speed2 = np.digitize(selected_speed2, bins)
    bin_means2 = [selected_fluo2[digitized_speed2 == i].mean() for i in range(1, len(bins))]
    bin_sem2 = [sem(selected_fluo2[digitized_speed2 == i]) for i in range(1, len(bins))]
        
    fig, ax1 = plt.subplots(1,1, figsize=(6, 6))
    fig.suptitle('Range: {} to {} cm/s'.format(min_speed, max_speed))
    ax1.set_ylabel('\u0394F/F0')
    ax1.set_xlabel('Speed (cm/s)')
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.fill_between(bins[0:len(bin_means1)], [m_i + s_i for m_i, s_i in zip(bin_means1, bin_sem1)], [m_i - s_i for m_i, s_i in zip(bin_means1, bin_sem1)], color = color1, alpha = 0.3)
    ax1.fill_between(bins[0:len(bin_means2)], [m_i + s_i for m_i, s_i in zip(bin_means2, bin_sem2)], [m_i - s_i for m_i, s_i in zip(bin_means2, bin_sem2)], color = color2, alpha = 0.3)
    ax1.plot(bins[0:len(bin_means1)], bin_means1, color=color1, label= label1)
    ax1.plot(bins[0:len(bin_means2)], bin_means2, color=color2, label = label2)
    ax1.legend()
    #plt.tight_layout()
    plt.show()
    
    
def zero_row_bars(matrix, condition):
    z_matrix = np.copy(matrix)
    # Check condition array size
    if len(condition) != z_matrix.shape[0] or len(condition) != z_matrix.shape[1]:
        print('Condition array is not matching the size of the matrix')
    else:
        # Zero rows
        for i, c in enumerate(condition):
            if c:
                z_matrix[i,:] = np.zeros(z_matrix.shape[1])
                z_matrix[:,i] = np.zeros(z_matrix.shape[1])
        return z_matrix
    

def pearson_shuffle(speed, F, smoothing_window=15, downsampling=3, N=10000):
    '''
    Calculates the probability of the Pearson value.
    After filtering and downsampling the speed trace (speed) and the fluroescence trace (F)
    the speed trace is shuffled (circular shuffling) N times and the pearson correlation 
    is calculated. 
    The Pearson value is then compared with the 99th, 98th and 95th percentile of the 
    shuffled Pearson values. 
    
    Input:
        speed - (1,x) array of speed
        F - (n,x) array of fluorescence
        smoothing_window - number of frames to use in gaussian filter
        downsampling - number of times to downsample (at 30Hz, 3 results in 10Hz signal)
        N - number of repetitions of circular shuffling
        
    Output:
        shuffle - dictionary containing:
            speed - (1,x) array of smoothed, downsampled speed trace
            F - (n,x) array of smoothed and downsampled F traces
            Pearson - (n,1) array of pearson correlation of speed and F
            shuffled pearson - (n,N) array of pearson correlation of shuffled speed and F
            stats - (n,2) array with highest percentile values
            sig - (n,1) array of string with signifficance value:
            
                        non sig - no signifficance found       
                        *       - 0.05
                        **      - 0.02
                        ***     - 0.01
            
    '''
    # Filter and downsample speed trace
    hamming_window = np.hamming(smoothing_window) #Dippopa 5frames à 10Hz -> 15frames à 30Hz
    speed_smoothed=  np.convolve(speed, hamming_window/hamming_window.sum(), mode='same') 
    decimated= signal.decimate(speed_smoothed, downsampling)
    
    if np.size(decimated[decimated != 0]) !=0:
        # Downsample and filter Fcalculate Pearson Correlation
        F_decimated=np.zeros(( len(F),len(decimated)))
        pearson=np.zeros((len(F),1))
        for roi in range(len(F)):
            F_smoothed= np.convolve(F[roi], hamming_window/hamming_window.sum(), mode='same')
            F_decimated[roi]= signal.decimate(F_smoothed, 3)
            # Calculate Pearson Correlation
            pearson[roi]= pearsonr(decimated, F_decimated[roi])[0]
    
        # Shuffle speed
        pearson_shuf=np.zeros(((len(F)), N))
        for i in range(N):
            split=random.randint(10, len(decimated)-10) #circular shuffling sup than 1s. (F aand speed at 10Hz)
            speed_shuf= np.hstack((decimated[split:len(decimated)],decimated[0:split]))
            # Calculate Pearson Correlation of shuffled speed trace
            for roi in range(len(F_decimated)):
                pearson_shuf[roi,i]=pearsonr(speed_shuf, F_decimated[roi])[0]
        
        # Calculate stats
        stats_percentile=np.zeros((len(pearson),2))
        sig=['']*len(pearson)
        for roi in range(len(F)):
            pval_sup= np.percentile(pearson_shuf[roi], 100-0.01*100) #one-sided
            pval_inf= np.percentile(pearson_shuf[roi], 0.01*100)
            if (pearson[roi]>0 and pearson[roi]>pval_sup) or (pearson[roi]<0 and pearson[roi]<pval_inf):
                stats_percentile[roi]=[pval_sup,pval_inf]
                sig[roi]='***'
            else:
                pval_sup= np.percentile(pearson_shuf[roi], 100-0.02*100) #one-sided
                pval_inf= np.percentile(pearson_shuf[roi], 0.02*100)
                if (pearson[roi]>0 and pearson[roi]>pval_sup) or (pearson[roi]<0 and pearson[roi]<pval_inf):
                    stats_percentile[roi]=[pval_sup,pval_inf]
                    sig[roi]='**'
                else:
                    pval_sup= np.percentile(pearson_shuf[roi], 100-0.05*100) #one-sided
                    pval_inf= np.percentile(pearson_shuf[roi], 0.05*100)
                    if (pearson[roi]>0 and pearson[roi]>pval_sup) or (pearson[roi]<0 and pearson[roi]<pval_inf):
                        stats_percentile[roi]=[pval_sup,pval_inf]
                        sig[roi]='*'
                    else:
                        stats_percentile[roi]=[pval_sup,pval_inf]
                        sig[roi]='Non sig'
                        
    else:
        F_decimated = []
        pearson_shuf = []
        pearson=np.zeros((len(F),1))
        for roi in range(len(F)):
            pearson[roi]='NaN'
        pearson_shuf = []
        stats_percentile = []
        sig=['NaN']*len(pearson)
    

    # Save results
    shuffle={}
    shuffle['speed']=decimated  
    shuffle['F']=F_decimated
    shuffle['pearson']=pearson
    shuffle['shuffled_pearson']=pearson_shuf
    shuffle['stats']=stats_percentile
    shuffle['sig']=sig
    
    return(shuffle)          
                
                
            
        
        
        
        
        
        