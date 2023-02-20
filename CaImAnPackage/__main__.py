# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:32:14 2022

@author: m.debritovanvelze
"""
import time
import RUN_2P
import settings_gui
import easygui

def main():
    settings = settings_gui.get_settings()
    if not settings:
        print('Error: No settings available')
    else:
        print('- Settings defined')
        print(settings)
    
    dataset = RUN_2P.define_dataset(settings=settings)
    if not dataset:
        print('Error: No dataset defined')
    else:
        print('- Dataset defined')
    
    if settings and dataset:
        print('- Starting Analysis....')
        combined_dic, save_dir = RUN_2P.run_Ca_analysis(settings=settings, dataset=dataset)
        #combined_figure.combined_figure(combined_dic, save_dir)
    else:
        print('ERROR: Settings and dataset not defined')
    
if __name__=='__main__':
    main()
