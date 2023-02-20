import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime

def bruker_xml_parser(filename):
    """
    function to parse the xml metadata file produced by the Prairie software
    TODO:
    - find automated ways to count channels
    """
    
    System_Bacci = '4E50-A99B-124D-DB04-704C-3554-E575-991F'
    System_AnimalFacility = 'B6D0-C5DD-AB1E-5EBC-6AB1-B9D9-9DF2-275E'
    
    mytree = ET.parse(filename)
    root = mytree.getroot()

    data = {'settings':{}}
    data['System ID'] = root[0].attrib['SystemID']
    settings = root[1]
    for setting in settings:
        if 'value' in setting.attrib:
            data['settings'][setting.attrib['key']] = setting.attrib['value']
        else:
            data['settings'][setting.attrib['key']] = {}
            for s in setting:
                if s.tag == 'IndexedValue':
                    if 'description' in s.attrib:
                        data['settings'][setting.attrib['key']][s.attrib['description']] = s.attrib['value']
                    else:
                        data['settings'][setting.attrib['key']][s.attrib['index']] = s.attrib['value']
                elif s.tag == 'SubindexedValues':
                    if len(list(s)) == 1:
                        data['settings'][setting.attrib['key']][s.attrib['index']] = s[0].attrib['value']
                    else:
                        data['settings'][setting.attrib['key']][s.attrib['index']] = {}
                        for sub in s:
                            data['settings'][setting.attrib['key']][s.attrib['index']][sub.attrib['description']] = [sub.attrib['value']]
    
    frames = root[2]
    for channel in ['Ch1', 'Ch2']:
        data[channel] = {'relativeTime':[],
                         'absoluteTime':[],
                         'tifFile':[]}
    data['StartTime'] = frames.attrib['time']
    
    for x in frames:  
        if x.tag == 'Frame':
            for f in x:
                for channel in ['Ch1', 'Ch2']:
                    if f.tag == 'File' and f.attrib['channelName'] == channel:
                        data[channel]['tifFile'].append(f.attrib['filename'])
                        for key in ['relativeTime', 'absoluteTime']:
                            data[channel][key].append(float(x.attrib[key]))

    # translation to numpy arrays
    for channel in ['Ch1', 'Ch2']:
        for key in ['relativeTime', 'absoluteTime']:
            data[channel][key] = np.array(data[channel][key], dtype=np.float64)
        for key in ['tifFile']:
            data[channel][key] = np.array(data[channel][key], dtype=str)
            
    # system specific extraction
    if data['System ID'] == System_AnimalFacility:
        data['settings']['positionCurrent']['Piezo'] = data['settings']['positionCurrent']['ZAxis']['Bruker 400 μm Piezo'][0]
        data['settings']['positionCurrent']['ZAxis'] = data['settings']['positionCurrent']['ZAxis']['Z Focus'][0]
        data['settings']['laserPower']['Imaging'] = data['settings']['laserPower']['Tunable']
        
                        
    return data

def get_journal(file_path, dic_journal):
    """
    Using file_path (path to experiment) and dic_journal (dictionary containing 
    the journal of the experiments) it extracts the information for this experiment.

    """
    pass


def calc_days(date1, date2):
    """
    Calculates the interval between the 2 dates in days. 
    data should be in format: yyyy.mm.dd    
    """
    date_format = "%Y.%m.%d"
    a = datetime.strptime(date1, date_format)
    b = datetime.strptime(date2, date_format)
    delta = b - a
    
    return abs(delta.days)

def new_bruker_xml_parser(filename):
    
    mytree = ET.parse(filename)
    root = mytree.getroot()

    data = {'settings':{}, 'date':root.attrib['date']}
    
    settings = root[1]
    for setting in settings:
        if 'value' in setting.attrib:
            data['settings'][setting.attrib['key']] = setting.attrib['value']
        else:
            data['settings'][setting.attrib['key']] = {}
            for s in setting:
                if s.tag == 'IndexedValue':
                    if 'description' in s.attrib:
                        data['settings'][setting.attrib['key']][s.attrib['description']] = s.attrib['value']
                    else:
                        data['settings'][setting.attrib['key']][s.attrib['index']] = s.attrib['value']
                elif s.tag == 'SubindexedValues':
                    if len(list(s)) == 1:
                        data['settings'][setting.attrib['key']][s.attrib['index']] = s[0].attrib['value']
                    else:
                        data['settings'][setting.attrib['key']][s.attrib['index']] = {}
                        for sub in s:
                            data['settings'][setting.attrib['key']][s.attrib['index']][sub.attrib['description']] = [sub.attrib['value']]
    
    for channel in ['Ch1', 'Ch2']:
        data[channel] = {'relativeTime':[],
                         'absoluteTime':[],
                         'depth_index':[],
                         'tifFile':[]}
    data['StartTime'] = root[2].attrib['time']

    depths = {}
    for frames in root[2:]:
        for x in frames:
            if x.tag == 'Frame':
                for f in x:
                    for channel in ['Ch1', 'Ch2']:
                        if f.tag == 'File' and (channel in f.attrib['channelName']):
                            data[channel]['tifFile'].append(f.attrib['filename'])
                            for key in ['relativeTime', 'absoluteTime']:
                                data[channel][key].append(float(x.attrib[key]))
                            if len(root)>3:
                                data[channel]['depth_index'].append(int(x.attrib['index'])-1)
                            else:
                                data[channel]['depth_index'].append(0)
                        # depth
                        if f.tag == 'PVStateShard':
                            for d in f:
                                if d.attrib['key']=='positionCurrent':
                                    for e in d:
                                        if e.attrib['index']=='ZAxis':
                                            for g in e:
                                                if g.attrib['description'] not in depths:
                                                    depths[g.attrib['description']] = []
                                                try:
                                                    depths[g.attrib['description']].append(float(g.attrib['value']))
                                                except ValueError:
                                                    pass

    # dealing with depth  --- MANUAL for piezo plane-scanning mode because the bruker xml files don't hold this info...
    if np.sum(['Piezo' in key for key in depths.keys()]):
        Ndepth = len(np.unique(data['Ch2']['depth_index'])) # SHOULD ALWAYS BE ODD
        try:
            for key in depths.keys():
                if 'Piezo' in key:
                    depth_start_piezo = depths[key][0]
            depth_middle_piezo = 200 # SHOULD BE ALWAYS CENTER AT 200um
            data['depth_shift'] = np.linspace(-1, 1, Ndepth)*(depth_middle_piezo-depth_start_piezo)
        except BaseException as be:
            print(be)
            print(' /!\ plane info was not found /!\ ')
            data['depth_shift'] = np.arange(1, Ndepth+1)
    else:
        data['depth_shift'] = np.zeros(1)

    # translation to numpy arrays
    for channel in ['Ch1', 'Ch2']:
        for key in ['relativeTime', 'absoluteTime']:
            data[channel][key] = np.array(data[channel][key], dtype=np.float64)
        for key in ['tifFile']:
            data[channel][key] = np.array(data[channel][key], dtype=str)
        n_planes = np.unique(data[channel]['depth_index'])
        data[channel]['plane_time'] = {}
        for plane in n_planes:
            data[channel]['plane_time']['plane_'+str(plane)] = [data[channel]['relativeTime'][i] for i in range(len(data[channel]['relativeTime'])) if data[channel]['depth_index'][i]==plane]
    
    return data



