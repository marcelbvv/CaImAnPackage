
import pickle
import numpy as np
import easygui
import os


def save_variable(name, variable):
    pickle_name = name+'.pickle'
    pickle_out = open(pickle_name,"wb")
    pickle.dump(variable, pickle_out)
    pickle_out.close()
    return


def load_variable(variable_name):
    variable = pickle.load(open(variable_name, 'rb'))
    return variable


def add_pathfile(new_filepath, savepath):
    with open(savepath, 'rb') as f:
        file_list = pickle.load(f)
    if new_filepath not in file_list:
        file_list.append(new_filepath)
    with open(savepath, 'wb') as f:
        pickle.dump(file_list, f)
    return
    

def export_dF(dF, time_sec, name_file):
    dF_array = dF.transpose()
    t_array = np.linspace(0, time_sec, np.size(dF, 1))
    #L1 = np.arange(np.size(dF,0)).reshape(1,np.size(dF,0))
    #L2 = np.append('Time', L1).reshape(1,(np.size(dF,0)+1))
    L3 = np.append(t_array.reshape(np.size(dF,1),1), dF_array, axis=1)
    #export_file = np.append(L2, L3, axis=0)
    np.savetxt(name_file, L3, delimiter=',')
    return


def load_session():
    file_path = easygui.fileopenbox(title='Select session to import:', multiple=False)
    session_results = load_variable(file_path)
    return session_results


def manage_save_location():
    save_directory = easygui.filesavebox(msg = 'Where to save data?')
    os.mkdir(save_directory)
    return save_directory