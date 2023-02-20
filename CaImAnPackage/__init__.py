# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:39:13 2022

@author: m.debritovanvelze
"""

# import RUN_2P

# def main():
#     settings = RUN_2P.define_settings()
#     dataset = RUN_2P.define_dataset(settings=settings)
#     combined_dic, save_dir = RUN_2P.run_Ca_analysis(settings=settings, dataset=dataset)
#     #combined_figure.combined_figure(combined_dic, save_dir)
    
# if __name__=='__main__':
#     main()

import settings_gui, base_calculation, list_preprocessing, save_session, plot_figures, get_dataRE
import spiking_analysis, compute_stats, cross_correlation, get_metadata, whisking_analysis, binary_calculations, bin_speed
import event_based_analysis, puff_analysis

