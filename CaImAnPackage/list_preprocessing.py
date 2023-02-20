
import os
import easygui
import numpy as np

# class main_list_UI(object):
    
# 	def show_popup(self):
# 		msg = QMessageBox()
# 		msg.setWindowTitle('List preprocessing')
# 		msg.setText('What do you want to analyze?')
# 		msg.setIcon(QMessageBox.Question)
# 		msg.setStandardButtons(QMessageBox.Cancel|QMessageBox.Retry|QMessageBox.Ignore|)
# 		msg.setDefaultButton(QMessageBox.Retry)

# 		msg.setDetailedText("details")

# 		msg.buttonClicked.connect(self.popup_button)

# 	def single_file_button(self, i):
# 		print(i.text())




def second_decision():
    msg = 'Add folder to list?\n\n'
    decision = easygui.ynbox(msg)
    path_list = []
    while decision == True:
        path = easygui.diropenbox()
        path_list.append(path)
        msg = msg + '\n--> ' + path.split('\\')[-1]
        decision = easygui.ynbox(msg)
    return path_list
            
def first_decision():
    msg = 'What do you want to analyze?'
    choices = ['A single file', 'An existing list', 'A new list']
    decision = easygui.buttonbox(msg, choices=choices)
    if decision == 'A single file':
        file_path = [easygui.diropenbox()]
        analysis_list = np.asarray(file_path)
    elif decision == 'An existing list':
        file_path = easygui.fileopenbox(title='Select list to analyze:', multiple=False)
        analysis_list = np.load(file_path)
    elif decision == 'A new list':
        path_list = second_decision()
        analysis_list = np.asarray(path_list)
        np.save(easygui.filesavebox(msg= 'Choose name and save location for list'), analysis_list)
    else:
        pass
    return analysis_list

def check_data(list_files):
    
    check_files = {}
    check_files['locomotion'] = ['.abf', '.txt']
    check_files['microscope'] = ['.xml']
    check_files['behaviour'] = ['.mp4']
   
    failed = {}
    failed['path'] = []
    failed['missing'] = []
    for file in list_files:
        if os.path.isdir(file):
            fail = []
            for f in check_files:
                files_found = 0
                for form in check_files[f]:
                    r = [i for i in os.listdir(file) if i.endswith(form)]
                    files_found += len(r)
                if files_found == 0:
                    fail.append(f)
            if len(fail) > 0:
                failed['path'].append(file)
                failed['missing'].append(fail)
        else:
            print('-- Folder {} not found'.format(file))
    if failed['path']:
        for file, fail in zip(failed['path'], failed['missing']):
            if len(fail) == 1:
                text = fail[0]
            elif len(fail) == 2:
                text = fail[0]+' and '+fail[1]
            elif len(fail) == 3:
                text = fail[0]+', '+fail[1]+' and '+fail[2]
            print('-- Folder {} is missing {} files'.format('/'.join(file.rsplit('\\', 3)[-3:]), text))
        easygui.msgbox('When files are corrected, press OK to continue analysis')
    else:
        print('-- All folders complete')
        
        
def remove_empty_lists(list_files, type_cell= 'Green'):
    remove = []
    for i, file in enumerate(list_files):
        path = file+'\suite2p\plane0'
        try:
            iscell = np.load(path+'/iscell.npy', allow_pickle=True)
            red_cell = np.load(path+'/redcell.npy', allow_pickle=True)
            
            if type_cell == 'Green':
                index = [i for i, (cell, red) in enumerate(zip(iscell, red_cell)) if (cell[0]==1 and red[0]==0)]
            else:
                index = [i for i, (cell, red) in enumerate(zip(iscell, red_cell)) if (cell[0]==1 and red[0]==1)]
            if not index:
                remove.append(i)
        except:
           remove.append(i)
    print('-- Removed %s files from the list'%len(remove))
    new_list = np.delete(list_files, remove, 0)
        
    return new_list
    
