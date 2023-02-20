# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:32:22 2022

@author: m.debritovanvelze
"""
import sys
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QGridLayout, QGroupBox, QLabel, QLineEdit,
        QPushButton, QMainWindow)

class settings_UI(QDialog):
#class settings_UI(QMainWindow):
    def __init__(self, parent=None):
        super(settings_UI, self).__init__(parent)
        
        self.originalPalette = QApplication.palette()
        
        # Settings box
        self.settingsGroupBox = QGroupBox("Base Settings")
        self.sampling_label = QLabel("Frame rate:")
        self.sampling_input = QLineEdit('30')
        self.duration_label = QLabel("Recording duration (s):")
        self.duration_input = QLineEdit('300')
        self.loc_perc_label = QLabel("Min Locomotion percentage:")
        self.loc_perc_input = QLineEdit('5')
        self.f0_perc_label = QLabel("Min F0 baseline percentage:")
        self.f0_perc_input = QLineEdit('1.5')
        self.low_fluo_CheckBox = QCheckBox("&Remove low fluorescence cells")
        self.f0_cells_CheckBox = QCheckBox("&Remove negative F0 cells")
        self.f0_cells_CheckBox.setChecked(True)
        self.neuropil_CheckBox = QCheckBox("&Subtract neuropil")
        self.neuropil_CheckBox.setChecked(True)
        self.neuropil_label = QLabel("Neuropil factor:")
        self.neuropil_input = QLineEdit('0.7')
        def check_neuropil(state):
            if state == Qt.Checked:
                self.neuropil_label.setEnabled(True)
                self.neuropil_input.setEnabled(True)
            else:
                self.neuropil_label.setEnabled(False)
                self.neuropil_input.setEnabled(False)
        
        self.neuropil_CheckBox.stateChanged.connect(check_neuropil)
        layout_settings = QGridLayout()
        layout_settings.addWidget(self.sampling_label, 0, 0)
        layout_settings.addWidget(self.sampling_input, 0, 1)
        layout_settings.addWidget(self.duration_label, 1, 0)
        layout_settings.addWidget(self.duration_input, 1, 1)
        layout_settings.addWidget(self.loc_perc_label, 2, 0)
        layout_settings.addWidget(self.loc_perc_input, 2, 1)
        layout_settings.addWidget(self.f0_perc_label, 3, 0)
        layout_settings.addWidget(self.f0_perc_input, 3, 1)
        layout_settings.addWidget(self.low_fluo_CheckBox, 4, 0)
        layout_settings.addWidget(self.f0_cells_CheckBox, 5, 0)
        layout_settings.addWidget(self.neuropil_CheckBox, 6, 0)
        layout_settings.addWidget(self.neuropil_label, 7, 0)
        layout_settings.addWidget(self.neuropil_input, 7, 1)
        self.settingsGroupBox.setLayout(layout_settings)    
        
        # Baseline box
        self.baselineGroupBox = QGroupBox("Baseline Parameters")
        self.percentileGroupBox = QGroupBox('Percentile')
        self.minmaxGroupBox = QGroupBox('MinMax')
        self.stdGroupBox = QGroupBox('Lowest STD')
        def switch_baseline(index):
            if index == 0:
                self.percentileGroupBox.setEnabled(True)
                self.minmaxGroupBox.setDisabled(True)
                self.stdGroupBox.setDisabled(True)
            elif index == 1:
                self.percentileGroupBox.setDisabled(True)
                self.minmaxGroupBox.setEnabled(True)
                self.stdGroupBox.setDisabled(True)
            elif index == 2:
                self.percentileGroupBox.setDisabled(True)
                self.minmaxGroupBox.setDisabled(True)
                self.stdGroupBox.setEnabled(True)
            else:
                pass
        self.baseline_methods = ['Percentile', 'MinMax', 'Lowest STD']
        self.baseline_ComboBox = QComboBox()
        self.baseline_ComboBox.addItems(self.baseline_methods)
        self.baseline_Label = QLabel("Baseline method:")
        self.baseline_Label.setBuddy(self.baseline_ComboBox)
        self.baseline_ComboBox.currentIndexChanged.connect(switch_baseline)
            # Percentile box
        self.p_window_label = QLabel("Filter window (s):")
        self.p_window_input = QLineEdit('0.5')
        self.p_percentile_label = QLabel("Percentile:")
        self.p_percentile_input = QLineEdit('5')
        self.layout_percentile = QGridLayout()
        self.layout_percentile.addWidget(self.p_window_label, 0, 0)
        self.layout_percentile.addWidget(self.p_window_input, 0, 1)
        self.layout_percentile.addWidget(self.p_percentile_label, 1, 0)
        self.layout_percentile.addWidget(self.p_percentile_input, 1, 1)
        self.percentileGroupBox.setLayout(self.layout_percentile)
        self.percentileGroupBox.setEnabled(True)
            # MinMax box
        self.mm_sigma_label = QLabel("Sigma:")
        self.mm_sigma_input = QLineEdit('60')
        self.mm_window_label = QLabel("Window:")
        self.mm_window_input = QLineEdit('60')
        self.layout_minmax = QGridLayout()
        self.layout_minmax.addWidget(self.mm_sigma_label, 0, 0)
        self.layout_minmax.addWidget(self.mm_sigma_input, 0, 1)
        self.layout_minmax.addWidget(self.mm_window_label, 1, 0 )
        self.layout_minmax.addWidget(self.mm_window_input, 1, 1)
        self.minmaxGroupBox.setLayout(self.layout_minmax)
        self.minmaxGroupBox.setEnabled(False)
            # STD box
        self.std_window_label = QLabel("Window (s):")
        self.std_window_input = QLineEdit('60')
        self.std_s_window_label = QLabel("STD window (s):")
        self.std_s_window_input = QLineEdit('5')
        self.layout_std = QGridLayout()
        self.layout_std.addWidget(self.std_window_label, 0, 0)
        self.layout_std.addWidget(self.std_window_input, 0, 1)
        self.layout_std.addWidget(self.std_s_window_label, 1, 0)
        self.layout_std.addWidget(self.std_s_window_input, 1, 1)
        self.stdGroupBox.setLayout(self.layout_std)
        self.stdGroupBox.setEnabled(False)
            # Baselines - Combined layout 
        baseline_layout = QGridLayout()
        baseline_layout.addWidget(self.baseline_Label, 0, 0)
        baseline_layout.addWidget(self.baseline_ComboBox, 0, 1)
        baseline_layout.addWidget(self.percentileGroupBox, 1, 0, 1, 2)
        baseline_layout.addWidget(self.minmaxGroupBox, 2, 0, 1, 2)
        baseline_layout.addWidget(self.stdGroupBox, 3, 0, 1, 2)
        self.baselineGroupBox.setLayout(baseline_layout)
        
        # Plotting box
        self.plottingGroupBox = QGroupBox("Plotting Settings")
        self.show_figures_CheckBox = QCheckBox("&Show Figures")
        self.sort_PC_CheckBox = QCheckBox("&Sort by 1st PC")
        self.trace_offset_label = QLabel("Offset between traces:")
        self.trace_offset_input = QLineEdit('8')
        plotting_layout = QGridLayout()
        plotting_layout.addWidget(self.show_figures_CheckBox, 0, 0)
        plotting_layout.addWidget(self.sort_PC_CheckBox, 1, 0)
        plotting_layout.addWidget(self.trace_offset_label, 2, 0)
        plotting_layout.addWidget(self.trace_offset_input, 2, 1 )
        self.plottingGroupBox.setLayout(plotting_layout)
        
        # Locomotion box
        self.locomotionGroupBox = QGroupBox("Locomotion Settings")
        self.speed_thr_label = QLabel("Speed Threshold:")
        self.speed_thr_input = QLineEdit('0.1')
        self.time_before_label = QLabel("Time before:")
        self.time_before_input = QLineEdit('0.5')
        self.time_after_label = QLabel("Time after:")
        self.time_after_input = QLineEdit('2.5')
        self.loc_short_event_CheckBox = QCheckBox("&Remove short events")
        self.min_event_label = QLabel("Minimim event duratin (s):")
        self.min_event_label.setEnabled(False)
        self.min_event_input = QLineEdit('2')
        self.min_event_input.setEnabled(False)
        self.loc_short_event_CheckBox.toggled.connect(self.min_event_label.setEnabled)
        self.loc_short_event_CheckBox.toggled.connect(self.min_event_input.setEnabled)
        loc_layout = QGridLayout()
        loc_layout.addWidget(self.speed_thr_label, 0, 0)
        loc_layout.addWidget(self.speed_thr_input, 0, 1)
        loc_layout.addWidget(self.time_before_label, 1, 0)
        loc_layout.addWidget(self.time_before_input, 1, 1 )
        loc_layout.addWidget(self.time_after_label, 2, 0)
        loc_layout.addWidget(self.time_after_input, 2, 1)
        loc_layout.addWidget(self.loc_short_event_CheckBox, 3, 0)
        loc_layout.addWidget(self.min_event_label, 4, 0)
        loc_layout.addWidget(self.min_event_input, 4, 1)
        self.locomotionGroupBox.setLayout(loc_layout)
        
        # Whisking box
        self.whiskingGroupBox = QGroupBox("Whisking Settings")
        self.sigma_label = QLabel("Sigma:")
        self.sigma_input = QLineEdit('3')
        self.percentile_label = QLabel("Baseline percentile:")
        self.percentile_input = QLineEdit('10')
        self.whisk_short_event_CheckBox = QCheckBox("&Remove short events")
        self.min_event_label = QLabel("Minimim event duratin (s):")
        self.min_event_label.setEnabled(False)
        self.min_event_input = QLineEdit('0.5')
        self.min_event_input.setEnabled(False)
        self.whisk_short_event_CheckBox.toggled.connect(self.min_event_label.setEnabled)
        self.whisk_short_event_CheckBox.toggled.connect(self.min_event_input.setEnabled)
        self.join_event_CheckBox = QCheckBox("&Join neighbouring events")
        self.inter_event_label = QLabel("Maximum inter-event duratin (s):")
        self.inter_event_label.setEnabled(False)
        self.inter_event_input = QLineEdit('1')
        self.inter_event_input.setEnabled(False)
        self.join_event_CheckBox.toggled.connect(self.inter_event_label.setEnabled)
        self.join_event_CheckBox.toggled.connect(self.inter_event_input.setEnabled)
        whisk_layout = QGridLayout()
        whisk_layout.addWidget(self.sigma_label, 0, 0)
        whisk_layout.addWidget(self.sigma_input, 0, 1)
        whisk_layout.addWidget(self.percentile_label, 1, 0)
        whisk_layout.addWidget(self.percentile_input, 1, 1 )
        whisk_layout.addWidget(self.whisk_short_event_CheckBox, 2, 0)
        whisk_layout.addWidget(self.min_event_label, 3, 0)
        whisk_layout.addWidget(self.min_event_input, 3, 1)
        whisk_layout.addWidget(self.join_event_CheckBox, 4, 0)
        whisk_layout.addWidget(self.inter_event_label, 5, 0)
        whisk_layout.addWidget(self.inter_event_input, 5, 1)
        self.whiskingGroupBox.setLayout(whisk_layout)
        
        # Other box
        self.otherGroupBox = QGroupBox("Other Settings")
        self.skewnessGroupBox = QGroupBox('Skewness')
        self.pearsonGroupBox = QGroupBox('Pearson')
            # Skewness
        self.skew_win_label = QLabel('Filter Window size:')
        self.skew_win_input = QLineEdit('29')
        self.skew_filt_label = QLabel('Filter order:')
        self.skew_filt_input = QLineEdit('2')
        self.skew_layout = QGridLayout()
        self.skew_layout.addWidget(self.skew_win_label, 0, 0)
        self.skew_layout.addWidget(self.skew_win_input, 0, 1)
        self.skew_layout.addWidget(self.skew_filt_label, 1, 0)
        self.skew_layout.addWidget(self.skew_filt_input, 1, 1)
        self.skewnessGroupBox.setLayout(self.skew_layout)
            # Pearson Correlation
        self.pearson_window_label = QLabel('Smoothing window (frames):')
        self.pearson_window_input = QLineEdit('15')
        self.pearson_downsample_label = QLabel('Downsampling:')
        self.pearson_downsample_input = QLineEdit('3')
        self.pearson_shuffle_label = QLabel('Number shuffles:')
        self.pearson_shuffle_input = QLineEdit('10000')
        self.pearson_layout = QGridLayout()
        self.pearson_layout.addWidget(self.pearson_window_label, 0, 0)
        self.pearson_layout.addWidget(self.pearson_window_input, 0, 1)
        self.pearson_layout.addWidget(self.pearson_downsample_label, 1, 0)
        self.pearson_layout.addWidget(self.pearson_downsample_input, 1, 1)
        self.pearson_layout.addWidget(self.pearson_shuffle_label, 2, 0)
        self.pearson_layout.addWidget(self.pearson_shuffle_input, 2, 1)
        self.pearsonGroupBox.setLayout(self.pearson_layout)
            # Cross Correlation
        self.cc_label = QLabel('Cross-correlation shift (s):')
        self.cc_input = QLineEdit('60')
            # Deconvolution
        self.deconvolution_label = QLabel('Deconvolution method:')
        self.deconvolution_methods = ['sumbre', 'Something', 'Other thing']
        self.deconvolution_ComboBox = QComboBox()
        self.deconvolution_ComboBox.addItems(self.deconvolution_methods)
        other_layout = QGridLayout()
        other_layout.addWidget(self.skewnessGroupBox, 0, 0, 1, 2)
        other_layout.addWidget(self.pearsonGroupBox, 1, 0, 1, 2)
        other_layout.addWidget(self.cc_label, 2, 0)
        other_layout.addWidget(self.cc_input, 2, 1)
        other_layout.addWidget(self.deconvolution_label, 3, 0)
        other_layout.addWidget(self.deconvolution_ComboBox, 3, 1)
        self.otherGroupBox.setLayout(other_layout)
        
        
        self.save_button = QPushButton("Save and Launch Analysis")
        self.save_button.setDefault(True)
        settings = self.save_button.clicked.connect(self.get_results)
        self.save_button.clicked.connect(QCoreApplication.instance().quit)
        
        mainLayout = QGridLayout()
        mainLayout.addWidget(self.settingsGroupBox, 0, 0)
        mainLayout.addWidget(self.otherGroupBox, 0, 1)
        mainLayout.addWidget(self.baselineGroupBox, 1, 0, 2, 1)
        mainLayout.addWidget(self.locomotionGroupBox, 1, 1)
        mainLayout.addWidget(self.whiskingGroupBox, 2, 1)
        mainLayout.addWidget(self.plottingGroupBox, 3, 0)
        mainLayout.addWidget(self.save_button, 3, 1)

        self.setLayout(mainLayout)

        self.setWindowTitle("Analysis Settigns")
        #self.changeStyle('Fusion')
        
        return settings
      
    def get_results(self):
        self._output = {
                        # Experiment Information    
                        'fs': int(self.sampling_input.text()),
                        'time_seconds': int(self.duration_input.text()),
                        # 'define_cells': False,
                        # 'cell_group': 'Green + Red', #'Green' #'Green + Red'
                        # 'red_cells': {'separate':False,
                        #               'cell_type': 'Green+Red'}, #'Green' and 'Green+Red'
                        # 'filt_active_cells': 'savgol',
                        # 'SST' :             {'isSST' : False,
                        #                       'per' : 1/3
                        #                       },
                        
                        # Running settings
                        'locomotion':{
                            'speed threshold': float(self.speed_thr_input.text()), 
                            'time_before':float(self.time_before_input.text()),
                            'time_after':float(self.time_after_input.text()),
                            'remove_short_events': self.loc_short_event_CheckBox.isChecked(), 
                            'min_event_duration': float(self.min_event_input.text()) # in seconds            
                            },
                                
                        # Whisking settings
                        'whisking':{
                            'sigma': int(self.sigma_input.text()),
                            'percentile': int(self.percentile_input.text()),
                            'remove short bouts': self.whisk_short_event_CheckBox.isChecked(),
                            'whisk_min_duration': float(self.min_event_input.text()),          # in seconds    
                            'join bouts': self.join_event_CheckBox.isChecked(), 
                            'whisk_max_inter_bout': float(self.inter_event_input.text()),        # in seconds
                            'threshold method': 'Percentile of normalized'
                            },
                        
                        # Baseline parameters
                        'F0_method': self.baseline_ComboBox.currentText(),
                        'F0_settings':      {'Percentile':
                                                  {'sigma': int(self.mm_sigma_input.text()),
                                                  'window': int(self.mm_window_input.text())},
                                              'MinMax':
                                                  {'filter_window': float(self.p_window_input.text()),
                                                  'percentile': int(self.p_percentile_input.text())},
                                              'Lowest STD':
                                                  {'window': int(self.std_window_input.text()),
                                                  'std_win': int(self.std_s_window_input.text())}
                                            },
                            
                        # Skewness Calculation
                        'filt_window_size': int(self.skew_win_input.text()),
                        'filt_order': int(self.skew_filt_input.text()),
                        
                        # Pearson Correlation
                        'smoothing_window': int(self.pearson_window_input.text()), # number of frames
                        'downsampling': int(self.pearson_downsample_input.text()),
                        'N shuffling': int(self.pearson_shuffle_input.text()),
                        
                        # # Speed binning
                        # 'speed binning': {'bin_size': 0.1,
                        #                   'min_val': 0,
                        #                   'max_val': 20},
                        
                        # # Aligned events
                        # 'align locomotion': {
                        #             'onset t_before':8,
                        #             'onset t_after':5,
                        #             'baseline': [-8, -1], # in seconds
                        #             'response': [0, 4] # in seconds
                        #             },
                        # 'align whisking': {
                        #             'onset t_before':8,
                        #             'onset t_after':5,
                        #             'baseline': [-8, -1],
                        #             'response': [0, 4]
                        #             },
        
                            
                        # Analysis settings
                        'Min locomotion percentage': int(self.loc_perc_input.text()),
                        'Min F0 baseline percentage': float(self.f0_perc_input.text()),
                        'remove low fluo cells': self.low_fluo_CheckBox.isChecked(),
                        'remove negative F0 cells': self.f0_cells_CheckBox.isChecked(),
                        'Subtract neuropil' : self.neuropil_CheckBox.isChecked(),
                        'neuropil factor': float(self.neuropil_input.text()),
                            
                        # Deconvolution settings
                        'threshold_method' : self.deconvolution_ComboBox.currentText(), 
                        
                        # Cross Correlation settings
                        'CC shift (s)' : float(self.cc_input.text()),
                        
                        # # Binned speed
                        # 'Speed Binning':    {
                        #                     'min value': 0,
                        #                     'max value': 20,
                        #                     'n bins': 41,
                        #                     },
        
                        # Plotting settings
                        'do_show':self.show_figures_CheckBox.isChecked(),  
                        'graphtrace_offset': int(self.trace_offset_input.text()),
                        #'linreg_percentile': int(self.sampling_input.text()),
                        'sort 1st PC': self.sort_PC_CheckBox.isChecked(),
                        # 'select_cells':     {'do':True,
                        #           }
                        }
            
        super(settings_UI, self).accept()

            
    def get_output(self):
        return self._output

def get_settings():
    
    app = QApplication(sys.argv)
    gui = settings_UI()
    
    if gui.exec_() == settings_UI.Accepted:
        settings = gui.get_output()
    
    return settings
