import sys
import statistics
from scipy import signal
import scipy.stats as stats
import os, easygui
import numpy as np
import pandas as pd
from datetime import date
import glob
import time

#sys.path.append('C:/Users/m.debritovanvelze/Desktop/Analysis/2P_Analysis/CalciumAnalysisPackage')
import settings_gui, base_calculation, list_preprocessing, save_session, plot_figures, get_dataRE
import spiking_analysis, compute_stats, cross_correlation, get_metadata, whisking_analysis, binary_calculations, bin_speed
import event_based_analysis, puff_analysis


"""
To do:
    - Correct xml parser
    - Check what Lin regression (plotting settings) does
    - Include locomotion info in Results
    - Remove use of variable - movement_inactive
    - Remove positions and positions_absolute
    - Check if state cell is still needed
    - Check linear regression (line 388)
    - Check cross correlation with Yann
    - Calculate correlation of whisking and locomotion
    - Include settings of function into main settings
    - Make run vs rest plots
    - Correct single value extraction to account for exclusion criteria
    
"""
    
# def define_settings():
#     settings = {
#             # Experiment Information    
#             'fs': 30,
#             'time_seconds': 300,
#             'define_cells': False,
#             'cell_group': 'Green + Red', #'Green' #'Green + Red'
#             'red_cells': {'separate':False,
#                           'cell_type': 'Green+Red'}, #'Green' and 'Green+Red'
#             'filt_active_cells': 'savgol',
#             'SST' :             {'isSST' : False,
#                                  'per' : 1/3
#                                  },
            
#             # Running settings
#             'locomotion':{
#                 'speed threshold': 0.1, 
#                 'time_before':0.5,
#                 'time_after':2.5,
#                 'remove_short_events': False, 
#                 'min_event_duration': 2 # in seconds            
#                 },
                    
#             # Whisking settings
#             'whisking':{
#                 'sigma': 3,
#                 'percentile': 10,
#                 'remove short bouts': True,
#                 'whisk_min_duration': 0.5,          # in seconds    
#                 'join bouts': True, 
#                 'whisk_max_inter_bout': 1,        # in seconds
#                 'threshold method': 'Percentile of normalized'
#                 },
            
#             # Baseline parameters
#             'F0_method': 'PERCENTILE',
#             'F0_settings':      {'MINMAX':
#                                      {'sigma': 60,
#                                       'window': 60},
#                                  'PERCENTILE':
#                                      {'filter_window': 0.5,
#                                       'percentile': 5},
#                                  'LOWEST_STD':
#                                      {'window': 60,
#                                       'std_win': 5},
#                                  'SELECTED':
#                                      {}
#                                 },
                
#             # Skewness Calculation
#             'filt_window_size': 29,
#             'filt_order': 2,
            
#             # Pearson Correlation
#             'smoothing_window': 15, # number of frames
#             'downsampling': 3,
#             'N shuffling': 10000,
            
#             # Speed binning
#             'speed binning': {'bin_size': 0.1,
#                               'min_val': 0,
#                               'max_val': 20},
            
#             # # Aligned events
#             # 'align locomotion': {
#             #             'onset t_before':8,
#             #             'onset t_after':5,
#             #             'baseline': [-8, -1], # in seconds
#             #             'response': [0, 4] # in seconds
#             #             },
#             # 'align whisking': {
#             #             'onset t_before':8,
#             #             'onset t_after':5,
#             #             'baseline': [-8, -1],
#             #             'response': [0, 4]
#             #             },

                
#             # Analysis settings
#             'Min locomotion percentage': 5,
#             'Min F0 baseline percentage': 1.5,
#             'remove low fluo cells': False,
#             'remove negative F0 cells': True,
#             'Subtract neuropil' : True,
#             'neuropil factor': 0.7,
                
#             # Deconvolution settings
#             'threshold_method' : 'sumbre', 
            
#             # Cross Correlation settings
#             'CC shift (s)' : 60,
            
#             # # Binned speed
#             # 'Speed Binning':    {
#             #                     'min value': 0,
#             #                     'max value': 20,
#             #                     'n bins': 41,
#             #                     },

#             # Plotting settings
#             'do_show':False,  
#             'graphtrace_offset': 8,
#             'linreg_percentile': 5,
#             'sort 1st PC': True,
#             'select_cells':     {'do':True,
#                      }
#             }
#     return settings

def define_dataset(settings={}):
    if not settings:
        print('Error: no settings available')
    else:
        dataset = {}
        dataset['files'] = list_preprocessing.first_decision()
        # if settings['define_cells']:
        #     path_excel = easygui.fileopenbox(msg='Select Excel file with cell selection')
        #     df_selection = pd.read_excel(path_excel)
        #     dataset['files'], dataset['selected_cells'] = base_calculation.cell_select(dataset['files'], df_selection, cell_group=settings['cell_group'])
        list_preprocessing.check_data(dataset['files'])
    return dataset

def run_Ca_analysis(settings = {}, dataset={}):
    
    save_directory = save_session.manage_save_location()
    try:
        mouse_dictionary = np.load('C:/Users/m.debritovanvelze/Desktop/Mouse_list.npy', allow_pickle= True).item()
    except:
        mouse_dictionary = np.load(easygui.fileopenbox(title='Select Mouse dictionary:', multiple=False), allow_pickle='TRUE').item()
    t0 = time.time()
    Complete_data = {}
    CC_locomotion_traces = {}
    CC_whisking_traces = {}
    Combined_Data = pd.DataFrame()
    Whisker_stimulation = {}
    Whisker_stimulation['files'] = []

    list_files = dataset['files'] 
    # Remove files without cells
    if settings['red_cells']['separate'] == True:
        if settings['red_cells']['cell_type'] == 'Green':
            list_files = list_preprocessing.remove_empty_lists(list_files, type_cell= 'Green')
        elif settings['red_cells']['cell_type'] == 'Green+Red':
            list_files = list_preprocessing.remove_empty_lists(list_files, type_cell= 'Green+Red')
        else:
            print('Red cells settings not correct')
            
    for file in range(0,np.size(list_files)):
        
        ti = time.time()
        
        Settings = settings.copy()
        name = list_files[file].split('\\')[-3]+'/'+ list_files[file].split('\\')[-2]+'/'+list_files[file].split('\\')[-1]
        save_data = {}
        path = list_files[file] + '\suite2p\plane0'
        save_data['Calcium_Data'] = {}
        save_data['Calcium_Data']['data_path'] = path
        os.chdir(path)
        print('\n\nOpening File: '+ name)
        save_location = os.path.normpath(os.path.join(save_directory, name))
        save_data['Analysis_save_location'] = save_location
        os.makedirs(save_location)
        
        Settings['path'] = path
        Settings['save_path'] = save_location
        print('-- Using defined settings')
        
        # Create complete data table
        columns = []
        data = []
        columns.append('File')
        data.append(name)
        
        # Get metadata from Bruker acquisition
        acquisition_parameters = get_metadata.bruker_xml_parser(glob.glob(list_files[file]+'/*.xml')[0])
        columns.extend(('Acquisition Start','power','depth','frame_period','optical_zoom'))
        data.extend((acquisition_parameters['StartTime'],
                     float(acquisition_parameters['settings']['laserPower']['Imaging']),
                     abs(float(acquisition_parameters['settings']['positionCurrent']['ZAxis'])),
                     float(acquisition_parameters['settings']['framePeriod']),
                     float(acquisition_parameters['settings']['opticalZoom'])))
        save_data['Calcium_Data']['Acquisition_parameters'] = acquisition_parameters
        print('-- Extracting acquisition settings')
        
        # Get lab journal data
        if list_files[file].split('\\')[-2] in mouse_dictionary:
            mouse_info = mouse_dictionary[list_files[file].split('\\')[-2]] 
            recording_date = list_files[file].split('\\')[-3]
            if mouse_info['dob']:
                delta_age = get_metadata.calc_days(recording_date, mouse_info['dob'])
                columns.append('mouse_age')
                data.append(delta_age)
            if mouse_info['doi']:
                delta_injection = get_metadata.calc_days(recording_date, mouse_info['doi'])
                columns.append('days_after_injection')
                data.append(delta_injection)
            if mouse_info['cw']:
                delta_cw = get_metadata.calc_days(recording_date, mouse_info['cw'])
                columns.append('days_after_surgery')
                data.append(delta_cw)
            columns.extend(('mouse code','sex','genotype','date_window','date_injection',
                            'date_birth','date_recording','location',
                            'mouse_number','virus'))
            data.extend((list_files[file].split('\\')[-2],
                         mouse_info['sex'],
                         mouse_info['line'],
                         mouse_info['cw'],
                         mouse_info['doi'],
                         mouse_info['dob'],
                         recording_date, 
                         mouse_info['location'], 
                         mouse_info['number'],
                         mouse_info['virus']))
            save_data['Mouse_info'] = {'dic': mouse_info,
                                       'mouse code': list_files[file].split('\\')[-2], 
                                       'genotype': mouse_info['line'],
                                       'sex': mouse_info['sex'],
                                       'DOB': mouse_info['dob'],
                                       'Mouse_number': mouse_info['number'], 
                                       'Cranial_window_date': mouse_info['cw'],
                                       'Location_implant': mouse_info['location'],
                                       'Injection_date': mouse_info['doi'],
                                       'Injected_virus': mouse_info['virus']}
            if mouse_info['dob']:
                save_data['Mouse_info']['Age'] = get_metadata.calc_days(recording_date, mouse_info['dob'])
            if mouse_info['doi']:
                save_data['Mouse_info']['Days_after_injection'] = get_metadata.calc_days(recording_date, mouse_info['doi'])
            if mouse_info['cw']:           
                save_data['Mouse_info']['Days_after_surgery'] = get_metadata.calc_days(recording_date, mouse_info['cw'])
            print('-- Extracting mouse information')
            
        # Import raw data
        ops = np.load(glob.glob(path+'/*ops.npy')[0], allow_pickle=True).item()
        if Settings['define_cells'] == True and dataset['selected_cells'][file]:
            F_raw, Fneu_raw, spikes, cells = base_calculation.load_specific_data(path, dataset['selected_cells'][file]) 
        elif Settings['red_cells']['separate'] == True:
            F_raw, Fneu_raw, spikes, cells = base_calculation.load_red_cells(path, Settings) 
        else:
            F_raw, Fneu_raw, spikes, cells = base_calculation.load_data(path) 
        save_data['Calcium_Data']['Suite2P']={'file_location': path,
                                              'F': F_raw,
                                              'Fneu': Fneu_raw,
                                              'spikes': spikes,
                                              'cells': cells, 
                                              'ops': ops}
        print('-- Importing data from '+ path)
        
        ################################################################
        
        # Import and process locomotion data
        locomotion_file = glob.glob(list_files[file]+'/*.abf')
        if locomotion_file == []:
            locomotion_file = glob.glob(list_files[file]+'/*.txt')
        if locomotion_file == []:
            locomotion = []
            print('-- No running file available')
        else:
            
            locomotion = get_dataRE.process_locomotion(locomotion_file[0], settings, np.size(F_raw,1))
            
            # Save locomotion data
            save_data['Locomotion_data'] = locomotion
            locomotion['file'] = locomotion_file[0]
            print('-- Processing movement data')
            
            # Calculate mean speed on active periods
            sp = np.extract(locomotion['bool_binary_movement'], locomotion['speed'])
            speed_mean = np.mean(sp)
            columns.extend(('Percentage Locomotion', 'Mean speed',
                            'Min locomotion percentage', 'Locomotion - Average bout duration',
                            'Locomotion - Max bout duration'))
            data.extend((locomotion['percentage'], speed_mean,
                         Settings['Min locomotion percentage'],
                         locomotion['events']['mean duration'],
                         locomotion['events']['max duration']))
            save_data['Locomotion_data']['mean_speed'] = speed_mean
            
            # Save whisker stimulation data
            if 'puff signal' in locomotion:
                whisker_stim = puff_analysis.get_info(locomotion['puff signal'])
                print('-- File contains whisker stimulation')
                Whisker_stimulation['files'].append(Settings['save_path']+'/Analysis_Data')
                save_data['Whisker_stim_data'] = whisker_stim
                
        # Import and process whisking data
        whisking_file = glob.glob(list_files[file]+'/*proc.npy')
        if whisking_file == []:
            video_file = glob.glob(list_files[file]+'/*.avi')
            if video_file == []:    
                print('-- No whisking video or file available')
            else:
                print('-- Whisking video available but no analysis file available')
            whisking = {}
        else:
            whisking = whisking_analysis.process_whisking(whisking_file[0], Settings, rec_points=np.size(F_raw,1))
            #whisking['path_movie'] = glob.glob(list_files[file]+'/*.avi')[0]
            save_data['Whisking_data'] = whisking
            columns.extend(('Percentage Whisking',
                            'Whisking - Average bout duration', 
                            'Whisking - Max bout duration'))
            data.extend((whisking['percentage_whisking'],
                         whisking['mean event duration'],
                         whisking['max event duration']))
            
            # Define whisking only
            if locomotion:
                whisking['whisking only'] = whisking_analysis.whisking_only(whisking['binary_whisking'], 
                                                                            locomotion['binary_movement'], 
                                                                            Settings) 
                if whisking['whisking only'] == []:
                    columns.append('Percentage Whisking Only')
                    data.append(0)
                else:
                    whisking['whisking only']['percentage'] = np.count_nonzero(whisking['whisking only']['binary'])/len(whisking['whisking only']['binary'])
                    whisking['whisking only']['mean event duration'] = statistics.mean(whisking['whisking only']['bout_duration'])/settings['fs']
                    whisking['whisking only']['max event duration'] = max(whisking['whisking only']['bout_duration'])/settings['fs']
                    columns.extend(('Percentage Whisking Only',
                                    'Whisking only - Average bout duration', 
                                    'Whisking only - Max bout duration'))
                    data.extend((whisking['whisking only']['percentage'],
                                 whisking['whisking only']['mean event duration'], 
                                 whisking['whisking only']['max event duration']))
            print('-- Processing whisking data')
            
        # Define resting state
        if locomotion:
            if whisking:
                movement_inactive = binary_calculations.get_inactive(locomotion['binary_movement'], whisking['binary_whisking'])
    
            else:
                movement_inactive = locomotion['extended_binary_movement']
                
            percentage_resting = np.count_nonzero(movement_inactive)/len(movement_inactive)
            columns.append('Percentage Inactivity')
            data.append(percentage_resting)
        
        positions, positions_absolute = base_calculation.full_trace(F_raw, Settings)
        print('-- Calculating F0 using the full trace')
        
        # Subtract neuropil and calculate F0 and dF/F
        if Settings['Subtract neuropil']:
            m = Settings['neuropil factor']
            F = F_raw - (m * Fneu_raw)
            print('-- Using pre-defined neuropil (%s) percentage value'%m)
            columns.append('neuropil factor')
            data.append(m)
            
        else:
            F = np.copy(F_raw)
            print('-- Skipping neuropil substraction')
        save_data['F'] = F
        
        # Calculate F0
        F0 = base_calculation.calculate_F0(F, Settings)
        save_data['F0'] = F0
    
        # Calculate dF/F
        dF = base_calculation.deltaF_calculate(F, F0)
        F0_raw = base_calculation.calculate_F0(F_raw, Settings)
        F0neu_raw = base_calculation.calculate_F0(Fneu_raw, Settings)
        save_data.update(dF=dF, F0_raw=F0_raw, F0neu_raw=F0neu_raw)  
        print('-- Calculating dF/F')
        
        # Detect negative F0 cells
        negative_F0_cells = []
        for i, v in enumerate(F0):
            if any(v <= 0):
                negative_F0_cells.append(True)
            else:
                negative_F0_cells.append(False) 
        save_data['negative_F0_cells'] = negative_F0_cells
        print('-- %s cells out of %s have negative F0' %(sum(negative_F0_cells), len(F_raw)))
        
        # Detect traces with low fluorescence
        low_fluo_cells = []
        for i, v in enumerate(F0_raw.min(axis=1)):
            if v < Settings['Min F0 baseline percentage'] * F0neu_raw.min(axis=1)[i]:
                low_fluo_cells.append(True)
            else:
                low_fluo_cells.append(False)
        save_data['low_fluo_cells'] = low_fluo_cells
        columns.append('Min F0 baseline percentage')
        data.append(Settings['Min F0 baseline percentage'])
        print('-- %s cells out of %s have low baseline fluorescence' %(sum(low_fluo_cells), len(F_raw)))
        
        #Find up-state cells
        Settings['state']=[0]*len(F)    
        if Settings['SST']['isSST']== True:
            Settings['state']=base_calculation.state_detection(F, Settings)
            print('-- Identifying "up-state" cells')
        
        # Sort by the 1st PC
        if len(dF) > 1 and Settings['sort 1st PC'] == True:
            dF, order = compute_stats.sort_data(dF, method='StandardScaler',
                                                sorted_first_pc=True, 
                                                binned_01=False)
            
            cells = np.array([cells[i] for i in order[:,0]])
            cells = np.reshape(cells, (dF.shape[0], 1))
            
            negative_F0_cells = np.array([negative_F0_cells[i] for i in order[:,0]])
            negative_F0_cells = np.reshape(negative_F0_cells, (dF.shape[0], 1))
            
            low_fluo_cells = np.array([low_fluo_cells[i] for i in order[:,0]])
            low_fluo_cells = np.reshape(low_fluo_cells, (dF.shape[0], 1))
            
            F = np.array([F[i] for i in order])
            F = np.reshape(F, (dF.shape[0], dF.shape[1]))
            
            F0 = np.array([F0[i] for i in order])
            F0 = np.reshape(F0, (dF.shape[0], dF.shape[1]))
            
            F0_raw = np.array([F0_raw[i] for i in order])
            F0_raw = np.reshape(F0_raw, (dF.shape[0], dF.shape[1]))
            
            F0neu_raw = np.array([F0neu_raw[i] for i in order])
            F0neu_raw = np.reshape(F0neu_raw, (dF.shape[0], dF.shape[1]))
            
            F_raw = np.array([F_raw[i] for i in order])
            F_raw = np.reshape(F_raw, (dF.shape[0], dF.shape[1]))
            
            Fneu_raw = np.array([Fneu_raw[i] for i in order])
            Fneu_raw = np.reshape(Fneu_raw, (dF.shape[0], dF.shape[1]))
            
            positions = np.array([positions[i] for i in order])
            positions = np.reshape(positions, (dF.shape[0], 2))
            
            positions_absolute = np.array([positions_absolute[i] for i in order])
            positions_absolute = np.reshape(positions_absolute, (dF.shape[0], 2))
            
            spikes = np.array([spikes[i] for i in order])
            spikes = np.reshape(spikes, (dF.shape[0], dF.shape[1]))
            
            save_data.update(dF=dF, order=order, negative_F0_cells=negative_F0_cells,
                             low_fluo_cells=low_fluo_cells, F=F, F0=F0, F0_raw=F0_raw,
                             F0neu_raw=F0neu_raw, F_raw=F_raw, Fneu_raw=Fneu_raw,
                             spikes=spikes)  
            
            print('-- Sorting traces by  the 1st PC')
        else:
            pass
        
        # Define active cells using sumbre threshold
        th_dF = base_calculation.dF_std(dF, Settings)
        state_cells, per_outside, filt_trace = base_calculation.active_silent(dF, th_dF, Settings, window_length=2.5)
        Settings['state']=state_cells
        save_data.update(th_dF=th_dF, state_cells=state_cells, per_outside=per_outside,
                         filt_trace=filt_trace)
    
        if Settings['SST']['isSST']==True:
            sup=' ({}% of weird cells)'.format( int( np.sum([1 for e in state_cells if e==-1])/np.size(state_cells)*100 ) )
            print('-- {0}% of the cells detected are active, {1}% of the cells detected are silent{2} over a total of {3} cells.'.format( \
                int(np.sum([1 for e in state_cells if e==1])/np.size(state_cells)*100), int(np.sum([1 for e in state_cells if e==0])/np.size(state_cells)*100), \
                sup, len(dF)))

        # Calculate skewness
        skewness_F = []
        skewness_dF = []
        for i in range(len(F)):
            skewness_F.append(stats.skew(signal.savgol_filter(F[i],Settings['filt_window_size'],Settings['filt_order'])))
            skewness_dF.append(stats.skew(signal.savgol_filter(dF[i],Settings['filt_window_size'],Settings['filt_order'])))
        save_data.update(skewness_F=skewness_F, skewness_dF=skewness_dF)
        
        # Extract mean and min values
        average_dF = np.mean(dF, axis=1) 
        average_F0_raw = np.mean(F0_raw, axis=1)
        min_F0_raw = np.min(F0_raw, axis=1) 
        average_F0neu_raw = np.mean(F0neu_raw, axis=1) 
        min_F0neu_raw = np.min(F0neu_raw, axis=1) 
        save_data.update(average_dF=average_dF, average_F0_raw=average_F0_raw,
                         min_F0_raw=min_F0_raw, average_F0neu_raw=average_F0neu_raw,
                         min_F0neu_raw=min_F0neu_raw)
        
        # Create table with data and include data
        data = np.array(data)
        data = np.expand_dims(data, axis=0)
        data = np.repeat(data, len(F), axis=0)
        
        columns.extend(('Cell state', 'cell number', 'Negative F0 cells', 'Low fluo cells', 'Threshold dF',
                        'Average dF', 'Average rawF F0', 'Min rawF F0', 'Average rawNeu F0', 'Min rawNeu F0',
                        'Skewness F', 'Skewness dF'))
        rows = ['ROI %d' % x for x in range(0, len(dF))]
        data = np.concatenate([data,
                               np.reshape(state_cells, (len(state_cells),1)), #Cell state
                               np.reshape(cells, (len(cells),1)), #cell number
                               np.reshape(negative_F0_cells, (len(negative_F0_cells),1)), #negative F0 cells
                               np.reshape(low_fluo_cells, (len(low_fluo_cells),1)), #low fluo cells
                               np.reshape(th_dF, (len(th_dF),1)), 
                               np.reshape(average_dF, (len(average_dF),1)),
                               np.reshape(average_F0_raw, (len(average_F0_raw),1)),
                               np.reshape(min_F0_raw, (len(min_F0_raw),1)),
                               np.reshape(average_F0neu_raw, (len(average_F0neu_raw),1)),
                               np.reshape(min_F0neu_raw, (len(min_F0neu_raw),1)),
                               np.reshape(skewness_F, (len(skewness_F),1)),
                               np.reshape(skewness_dF, (len(skewness_dF),1))
                               ], axis =1)
        
        # Spiking Analysis 
        amplitude, Nb_of_events, spikes_binary, thr_spikes = spiking_analysis.combined(spikes, dF, Settings, state_cells)
        columns.extend(('Amplitude', 'Nb of events', 'thr spike'))
        data=np.hstack((data, 
                        np.reshape(amplitude, (len(amplitude),1)), #Amplitude
                        np.reshape(Nb_of_events, (len(Nb_of_events),1)), #Nb of events
                        np.reshape(thr_spikes, (len(thr_spikes),1)))) #th spike
        save_data.update(amplitude=np.copy(amplitude), Nb_of_events=np.copy(Nb_of_events), 
                         spikes_binary=np.copy(spikes_binary), thr_spikes=np.copy(thr_spikes))
        print('-- Analysing deconvolved data')
        
        if whisking:
            # Cross correlation with whisking
            CC_whisking_array, CC_whisking_shift, CC_whisking_lag = cross_correlation.CC_calculate(dF, whisking['resampled_trace'], Settings, 'whisking')
            columns.append('CC whisking lag')
            data=np.hstack((data, np.reshape(CC_whisking_lag, (len(CC_whisking_lag),1))))
            CC_whisking_traces[name] = {}
            CC_whisking_traces[name]['time'] = CC_whisking_shift
            if len(dF)>1:
                if Settings['remove negative F0 cells']:    
                    CC_whisking_traces[name]['traces'] = np.delete(CC_whisking_array, [i for i, x in enumerate(list(negative_F0_cells)) if x[0]==True], 0)
                    CC_whisking_traces[name]['lag'] = np.delete(CC_whisking_lag, [i for i, x in enumerate(list(negative_F0_cells)) if x[0]==True], 0)
                else:
                    CC_whisking_traces[name]['traces'] = CC_whisking_array
                    CC_whisking_traces[name]['lag'] = CC_whisking_lag
                save_data.update(CC_whisking_array=CC_whisking_array, CC_whisking_shift=CC_whisking_shift, CC_whisking_lag=CC_whisking_lag)
            
        if locomotion:
            if locomotion['percentage']*100 < Settings['Min locomotion percentage'] or percentage_resting*100 > 100-Settings['Min locomotion percentage']:
                print('---- Only one state (resting or running) in this session ----')
    
            else :
                trace_bin = compute_stats.bin_trace(dF, Settings, method = 'mean', 
                                                    window = 2)
                speed_bin = compute_stats.bin_trace(locomotion['speed'], Settings, 
                                                    method = 'mean', window = 2)
                save_data.update(speed_bin=speed_bin, trace_bin=trace_bin)
                
                # Calculate Linear regression of speed and dF
                #r_sq, slope, intercept = compute_stats.linreg(speed_bin, trace_bin)
                r_sq, slope, intercept = compute_stats.lin_reg_moving(locomotion['speed'], dF, speed_threshold = Settings['locomotion']['speed threshold'])
                
                if not intercept:
                    print('-- Speed slope not calculated: no speed points above threshold') 
                else:
                    save_data.update(r_sq=r_sq, slope=slope, intercept=intercept)
                    # if len(dF) <= 50:
                    #     plot_figures.plot_neuron_vs_speed(speed_bin, trace_bin, slope,
                    #                                       intercept, r_sq,                                                                   
                    #                                       Settings['save_path'], 
                    #                                       show=Settings['do_show'])
                    columns.extend(('Speed Slope', 'R value Slope'))
                    data=np.hstack((data,np.reshape(np.array(slope), (len(slope),1)), #Speed Slope
                                    np.reshape(np.array(r_sq), (len(r_sq),1)))) #R value Slope
                
                # Calculate Pearson
                pearson_shuffle = compute_stats.pearson_shuffle(locomotion['speed'], F, smoothing_window = Settings['smoothing_window'], downsampling = Settings['downsampling'], N = Settings['N shuffling'])
                #pearson_shuffle['sign'] = [pearson_shuffle['pearson']>0]
                save_data.update(pearson_shuffle=pearson_shuffle)
                columns.extend(('Pearson Correlation', 'Pearson p-value',)) # 'Pearson sign'))
                data=np.hstack((data,
                                np.reshape(np.array(pearson_shuffle['pearson']), (len(pearson_shuffle['pearson']),1)), #Pearson Correlation
                                np.reshape(np.array(pearson_shuffle['sig']), (len(pearson_shuffle['sig']),1)) #Sig value
                                #np.reshape(np.array(pearson_shuffle['sign']), (len(pearson_shuffle['sign']),1)) # Pearson Sign
                                ))
                if whisking:
                    if len(dF)>1:
                        if Settings['remove negative F0 cells']:    
                            CC_whisking_traces[name]['pearson'] = np.delete(pearson_shuffle['pearson'], [i for i, x in enumerate(list(negative_F0_cells)) if x[0]==True], 0)
                        else:
                            CC_whisking_traces[name]['pearson'] = pearson_shuffle['pearson']
                
                # Movement correlation
                rho, p_val = compute_stats.neuron_speed_correlation(locomotion['speed'], 
                                                                    dF,
                                                                    method='spearman',
                                                                    do_bin=False,
                                                                    method_speed='mean',
                                                                    method_neuron='mean',
                                                                    samplingRate=Settings['fs'],
                                                                    window_bin=5)
                save_data.update(rho=rho, p_val=p_val)
                columns.extend(('spearman coeff','spearman p value'))
                data=np.hstack((data,np.reshape(rho, (len(rho),1)),  #spearman coeff
                                np.reshape(p_val, (len(p_val),1)) )) #p value
                    
                
                #LMI dF
                active_f, inactive_f, lmi_f = compute_stats.LMI(locomotion['binary_movement'], 
                                                                movement_inactive,
                                                                dF, Settings,
                                                                method = 'mean dF')
                columns.extend(('Run dF', 'Rest dF', 'LMI dF'))
                data=np.hstack((data, 
                                np.reshape(active_f, (len(active_f),1)), # Run dF
                                np.reshape(inactive_f,(len(inactive_f),1)), #Rest dF
                                np.reshape(np.asarray(lmi_f), (len(lmi_f),1)) )) #LMI dF
    
                #LMI Amplitude of spikes          
                active_a, inactive_a, lmi_a = compute_stats.LMI(locomotion['binary_movement'], 
                                                                movement_inactive,
                                                                spikes,  Settings,
                                                                method='sum amplitude')
                columns.extend(( 'Run Amplitude', 'Rest Amplitude', 'LMI Amplitude'))
                data=np.hstack((data, 
                                np.reshape(active_a, (len(active_a),1)), #Run Amplitude
                                np.reshape(inactive_a, (len(inactive_a),1)), #Rest Amplitude
                                np.reshape(np.asarray(lmi_a), (len(lmi_a),1)) )) #LMI Amplitude 
                                          
                #LMI Nb of Spikes
                active_s, inactive_s, lmi_s = compute_stats.LMI(locomotion['binary_movement'], 
                                                                movement_inactive,
                                                                spikes_binary, 
                                                                Settings,
                                                                method='events rate',
                                                                method_th=Settings['threshold_method'])
                columns.extend(('Run Nb of spikes', 'Rest Nb of spikes','LMI Nb of spikes'))
                data=np.hstack((data, 
                                np.reshape(active_s, (len(active_s),1)), #Run Nb of spikes
                                np.reshape(inactive_s, (len(inactive_s),1)), #Rest Nb of spikes
                                np.reshape(np.asarray(lmi_s), (len(lmi_s),1)) )) #LMI Nb of spikes
                plot_figures.compare_lmi(lmi_f, lmi_a, lmi_s, Settings['save_path'], 
                                         show = Settings['do_show'])
                
                if whisking:
                    #WMI dF
                    whisk_f, no_whisk_f, wmi_f = compute_stats.LMI(whisking['binary_whisking'], 
                                                                    movement_inactive,
                                                                    dF, Settings,
                                                                    method = 'mean dF')
                    columns.extend(('Whisk dF', 'No whisk dF', 'WMI dF'))
                    data=np.hstack((data, 
                                    np.reshape(whisk_f, (len(whisk_f),1)), # Whisk dF
                                    np.reshape(no_whisk_f,(len(no_whisk_f),1)), # No whisk dF
                                    np.reshape(np.asarray(wmi_f), (len(wmi_f),1)))) #WMI dF
                    
                    # Whisking only modulation dF
                    if whisking['whisking only'] == []:
                        pass
                    else:
                        WO_f, _, WOMI_f = compute_stats.LMI(whisking['whisking only']['binary'], 
                                                                        movement_inactive,
                                                                        dF, Settings,
                                                                        method = 'mean dF')
                        columns.extend(('WO dF', 'WOMI dF'))
                        data=np.hstack((data, 
                                        np.reshape(WO_f, (len(WO_f),1)), # Whisk-only dF
                                        np.reshape(np.asarray(WOMI_f), (len(WOMI_f),1)))) #WOMI dF
                    
                # # PMI        
                # pmi_calc = compute_stats.get_pmi(dF, locomotion['speed'], Settings)
                # PMI_all = pmi_calc[0]
                # coor_rest = pmi_calc[1]
                # coor_run = pmi_calc[2]
                # mod_index_pmi = pmi_calc[3]
                # columns.extend(('Run coor pmi', 'Rest coor pmi', 'Mod index pmi'))
                # data=np.hstack((data, 
                #                 np.reshape(coor_run, (len(coor_run),1)), #Run coor pmi
                #                 np.reshape(coor_rest, (len(coor_rest),1)), #Rest coor pmi
                #                 np.reshape(mod_index_pmi, (len(mod_index_pmi),1)) )) #Mod index pmi
                # columns.append('index kmeans')
                # data=np.hstack((data, np.reshape(pmi_calc[9], (len(pmi_calc[9]),1)) )) #index kmeans          
                
                # Cross correlation with speed
                #CC_array, CC_shift, CC_lag = cross_correlation.CC_calculate(dF, locomotion['speed'], Settings['CC shift (s)'], float(acquisition_parameters['settings']['framePeriod']), save_location)
                CC_array, CC_shift, CC_lag = cross_correlation.CC_calculate(dF, locomotion['speed'], Settings, 'locomotion')
                columns.append('CC lag')
                data=np.hstack((data, np.reshape(CC_lag, (len(CC_lag),1))))
                CC_locomotion_traces[name] = {}
                CC_locomotion_traces[name]['time'] = CC_shift
                if len(dF) > 1:
                    if Settings['remove negative F0 cells']:    
                        CC_locomotion_traces[name]['traces'] = np.delete(CC_array, [i for i, x in enumerate(list(negative_F0_cells)) if x[0]==True], 0)
                        CC_locomotion_traces[name]['lag'] = np.delete(CC_lag, [i for i, x in enumerate(list(negative_F0_cells)) if x[0]==True], 0)
                        CC_locomotion_traces[name]['pearson'] = np.delete(pearson_shuffle['pearson'], [i for i, x in enumerate(list(negative_F0_cells)) if x[0]==True], 0)
                    else:
                        CC_locomotion_traces[name]['traces'] = CC_array
                        CC_locomotion_traces[name]['lag'] = CC_lag
                        CC_locomotion_traces[name]['pearson'] = pearson_shuffle['pearson']
                save_data.update(active_dF=active_f, inactive_dF=inactive_f, 
                                 lmi=lmi_f, active_a=active_a, inactive_a=inactive_a,
                                 lmi_a=lmi_a, active_s=active_s, inactive_s=inactive_s,
                                 lmi_s=lmi_s, 
                                 #pmi_calc=pmi_calc, PMI_all=PMI_all,
                                 #coor_rest=coor_rest, coor_run=coor_run,
                                 #mod_index_pmi=mod_index_pmi,
                                 CC_array=CC_array, CC_shift=CC_shift, CC_lag=CC_lag)
                
                
                
                print('-- Performing Spiking Analysis')
                
        # Grouped analysis
                
        # Data exclusion condition
        if len(dF) > 1:
            if Settings['remove low fluo cells'] and Settings['remove negative F0 cells']:
                condition = list(np.concatenate([a and b for a, b in zip(low_fluo_cells, negative_F0_cells)]).flat)
            elif Settings['remove low fluo cells']:
                condition = list(np.concatenate(low_fluo_cells).flat)
            elif Settings['remove negative F0 cells']:
                condition = list(np.concatenate(negative_F0_cells).flat)
            else:
                condition = [True]*len(dF)
            save_data['condition'] = condition
        
        # Synchrony analysis
        if len(dF) > 1:
            matrix_synchro = compute_stats.synchrony(spikes, Settings, thr_spikes, 
                                                      method='STTC', w_size=10)
            if len(condition)-sum(condition)>=2:
                # Apply Condition
                selected_matrix = compute_stats.zero_row_bars(matrix_synchro, condition)
                lower_matrix = np.tril(selected_matrix, k=-1)
                synchrony_values = list(lower_matrix[np.nonzero(lower_matrix)])
                mean_synchrony = np.full((len(dF),1), statistics.mean(synchrony_values))
                columns.append('Mean Synchrony')
                data=np.hstack((data, np.reshape(mean_synchrony, (len(mean_synchrony),1)) ))
                save_data.update(selected_matrix=selected_matrix,
                                 synchrony_values=synchrony_values,
                                 mean_synchrony=mean_synchrony)
                print('-- Calculating synchrony')  
            
            plot_figures.plot_matrix_synchro(matrix_synchro, Settings, 
                                             show=Settings['do_show'])
            print('-- Plotting synchrony')   
        else :
            matrix_synchro = [[np.nan]]
        save_data['matrix_synchro'] = matrix_synchro
        
        if locomotion:        
            if locomotion['percentage']*100 >= Settings['Min locomotion percentage'] and percentage_resting*100 <= 100-Settings['Min locomotion percentage']:
                # synchronicity in running and resting state  
                if len(dF) > 1 and len(condition)-sum(condition)>=2:
                    
                    n_run,n_rest = base_calculation.get_run_rest(spikes, 
                                                                 locomotion['binary_movement'], 
                                                                 movement_inactive)
                    synchro_run = compute_stats.synchrony(n_run, Settings, 
                                                          thr_spikes, 
                                                          method='STTC', 
                                                          w_size=10)
                    run_selected_matrix = compute_stats.zero_row_bars(synchro_run, condition)
                    run_lower_matrix = np.tril(run_selected_matrix, k=-1)
                    run_synchrony_values = list(run_lower_matrix[np.nonzero(run_lower_matrix)])
                    run_mean_synchrony = np.full((len(dF),1), statistics.mean(run_synchrony_values))
                
                    synchro_rest = compute_stats.synchrony(spikes, Settings, 
                                                           thr_spikes, 
                                                           method='STTC', 
                                                           w_size=10)
                    rest_selected_matrix = compute_stats.zero_row_bars(synchro_rest, condition)
                    rest_lower_matrix = np.tril(rest_selected_matrix, k=-1)
                    rest_synchrony_values = list(rest_lower_matrix[np.nonzero(rest_lower_matrix)])
                    rest_mean_synchrony = np.full((len(dF),1), statistics.mean(rest_synchrony_values))
                    save_data.update(n_run=n_run, n_rest=n_rest, synchro_run=synchro_run,
                                     run_selected_matrix=run_selected_matrix,
                                     run_synchrony_values=run_synchrony_values,
                                     run_mean_synchrony=run_mean_synchrony,
                                     synchro_rest=synchro_rest, 
                                     rest_selected_matrix=rest_selected_matrix,
                                     rest_synchrony_values=rest_synchrony_values,
                                     rest_mean_synchrony=rest_mean_synchrony)
                    
                    columns.extend(('Mean synchrony run', 'Mean synchrony rest'))
                    data=np.hstack((data, run_mean_synchrony, rest_mean_synchrony)) #% synchrony run, % synchrony rest
            

        # Plot data
        for n in range(0,np.size(dF,0)):
            plot_figures.plot_F_ROI(F_raw, Fneu_raw, F, F0, dF, n, th_dF, per_outside, filt_trace, Settings, show=Settings['do_show']) 
                
        plot_figures.plot_dF_spikes(dF, spikes, thr_spikes, Settings,
                                    chosen_index=[i for i in range(len(dF))], 
                                    method_th=Settings['threshold_method'], 
                                    c=0, show=Settings['do_show'])  
        print('-- Plotting traces') 
        
        if locomotion:
            if len(dF) >1:
                if whisking:
                    plot_figures.spikes_movement_whisking(dF, spikes, thr_spikes, movement_inactive, locomotion, whisking, Settings,
                             chosen_index=[i for i in range(len(dF))],
                             do_filter=False, 
                             method_th=Settings['threshold_method'],
                             c=0, show=Settings['do_show'])
                    plot_figures.heatmap_whisking(dF, locomotion, whisking, Settings)
                    print("-- Plotting spikes_movement")
                    
                else:
                    plot_figures.spikes_movement(dF, spikes, thr_spikes, movement_inactive, locomotion, Settings,
                                                 chosen_index=[i for i in range(len(dF))],
                                                 do_filter=False, 
                                                 method_th=Settings['threshold_method'],
                                                 c=0, show=Settings['do_show'])  
                    plot_figures.heatmap(dF, locomotion, Settings)
                    print("-- Plotting spikes_movement")
            
            if locomotion['percentage']*100 >= Settings['Min locomotion percentage'] and percentage_resting*100 < 100-Settings['Min locomotion percentage']:
                # Plot
                plot_figures.plot_run_vs_rest(inactive_f, active_f,
                                              Settings['save_path'], show=Settings['do_show'])
                plot_figures.plot_run_vs_rest(inactive_a, active_a, 
                                              Settings['save_path'], show=Settings['do_show'])
                plot_figures.plot_run_vs_rest(inactive_s, active_s, 
                                              Settings['save_path'], show=Settings['do_show'])
                # plot_figures.plot_pmi(PMI_all,coor_rest, coor_run, mod_index_pmi, 
                #                       Settings['save_path'], nb_of_clusters=2, 
                #                       show=Settings['do_show'])
                print('-- Plotting results from the movement data')
            
        ##################################################################
        # Saving variables (results) and compiling them in an excel file #
        ##################################################################
        
        # Save settings
        Settings['analysis_date'] = date.today().isoformat()
        save_data['Settings'] = Settings   
        
        table_data = pd.DataFrame(data, rows, columns)
        save_data['table_data'] = table_data   
        Combined_Data = Combined_Data.append(table_data,sort =True)
        save_session.save_variable(Settings['save_path']+'/Analysis_Data', save_data)
        table_data.to_excel(Settings['save_path']+'/Experiment_Results.xlsx') 
    
        Complete_data[name] = Settings['save_path']+'/Analysis_Data'
        
        print('time %4.2f sec. Finished analyzing file'%(time.time()-(ti)))
    
    
    low_fluo_selected_table = Combined_Data.loc[Combined_Data['Low fluo cells'] == 'False']
    negative_F0_selected_table = Combined_Data.loc[Combined_Data['Negative F0 cells'] == 'False']
    combined_selected_table = Combined_Data.loc[(Combined_Data['Low fluo cells'] == 'False')&(Combined_Data['Negative F0 cells'] == 'False')]
    
    if len(low_fluo_selected_table.axes[0]) != 0:
        low_fluo_selected_table.to_excel(save_directory+'/LowFluoSelected.xlsx')
    
    if len(negative_F0_selected_table.axes[0]) != 0:
        negative_F0_selected_table.to_excel(save_directory+'/NegF0Selected.xlsx')
        df_negative_F0 = pd.read_excel(save_directory+'/NegF0Selected.xlsx')
        fov_data = df_negative_F0.groupby(['File']).mean()
        fov_data.to_excel((save_directory+'/NegF0Selected.xlsx')[:-5]+'_FOV.xlsx')
        
    if len(combined_selected_table.axes[0]) != 0:
        combined_selected_table.to_excel(save_directory+'/LowFluo_NegF0_Selected.xlsx')
    
    Combined_Data.to_excel(save_directory+'/Combined_Data.xlsx')
    save_session.save_variable((save_directory+'\Path_Analysis_files'), Complete_data)
        
    if len(CC_locomotion_traces) == 0:
        pass
    else:
        #cross_correlation.plot_combined_CC(CC_locomotion_traces, 'Locomotion', save_directory)
        save_session.save_variable((save_directory+'\Combined_CC_locomotion_traces'), CC_locomotion_traces)
        combined_traces_loc = cross_correlation.plot_combined_CC_pearson(CC_locomotion_traces, 'Locomotion', save_directory)
        
    if len(CC_whisking_traces) == 0:
        pass
    else:
        #cross_correlation.plot_combined_CC(CC_whisking_traces, 'Whisking',save_directory)
        save_session.save_variable((save_directory+'\Combined_CC_whisking_traces'), CC_whisking_traces)
        combined_traces_whisk = cross_correlation.plot_combined_CC_pearson(CC_whisking_traces, 'Whisking',save_directory)

    combined_dic = {'Data table': Combined_Data,
                    'Path to files': Complete_data}
    
    if len(CC_locomotion_traces) != 0:
        combined_dic.update({'CC locomotion': CC_locomotion_traces,
                             'CC combined traces locomotion': combined_traces_loc
                             })
    
    if len(CC_whisking_traces) != 0:
        combined_dic.update({'CC whisking': CC_whisking_traces,
                             'CC combined traces whisking': combined_traces_whisk
                             })      

    save_session.save_variable((save_directory+'\Combined_Results'), combined_dic)
    
    save_settings = open(save_directory+'\Settings.txt',"w")
    save_settings.write( str(settings) )
    save_settings.close()    
    
    #combined_figure.combined_figure(combined_dic, save_directory)
    #combined_figure.combined_figure(np.load(easygui.fileopenbox(), allow_pickle=True), 'C:/Users/m.debritovanvelze/Desktop')
    event_data = event_based_analysis.check_data(list_files=Complete_data, save_path=save_directory)
    if Whisker_stimulation['files']:
        puff_analysis.run_analysis(list_files= Whisker_stimulation['files'], save_path=save_directory)
    bin_speed.calc_binned_speed(list_files=Complete_data, settings=settings['speed binning'], plot_data=True, save_path=save_directory)
    event_based_analysis.figure_traces(event_data, save_path= save_directory)
    
    print('time %4.2f sec. Finished analyzing all files'%(time.time()-(t0)))
    return

def main():
    settings = settings_gui.get_settings()
    print(settings)
    #settings = define_settings()
    #dataset = define_dataset(settings=settings)
    #list_files = list_preprocessing.first_decision()
    # combined_dic, save_dir = run_Ca_analysis(settings=settings, dataset=dataset)
    #combined_figure.combined_figure(combined_dic, save_dir)
    
# if __name__=='__main__':
#     main()