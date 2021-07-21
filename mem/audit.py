# function for decision confirmation

from pandas.core.frame import DataFrame


def confirm():
    """
    output: user confirmation; Yes = True,  No = False
    """
    
    while True:
        answer = input('Are you sure? [Yes/No]: ').strip()

        if answer.lower() == 'yes':
            confirm = True
            break
        elif answer.lower() == 'no':
            confirm = False
            break
        else:
            print('Please confirm again!')

    return confirm

# function for getting file path from user input definition

def defpath(**kwargs):
    """
    input:  base = path base,
            file = file,
            filetype = file type
    ...
    output: file path from user
    """

    base = kwargs.get('base')
    file = kwargs.get('file')
    filetype = kwargs.get('filetype')

    import os.path as op

    while True:
        print('Which one is your %s directory?' %file)
        folder = input('Please indicate your %s directory name%s: ' %(file, filetype)).strip()

        if folder == '':
            print('Please type the directory name!')
        else:
            folderpath = op.join(base, folder)
            if op.isdir(folderpath):
                break
            else:
                print('Please try again, your directory \'%s\' is not found!' %folder)
    
    return folderpath

# function for well name extraction from file path

def filename(filelist):
    """
    input:  filelist = list of file paths
    ...
    output: list of file names
    """
    
    import os.path as op

    return [op.basename(file).split('_', 1)[0].lower() for file in filelist]

# function for grouping the files by well name of prefix

def grouping(**kwargs):
    """
    input:  las = list of well logging file path,
            dev = list of deviation file path,
            top = list of formation top file path,
            pres = list of pressure file path,
            core = list of core file path,
            drill = list of drilling test file path,
            mud = list of mud weight log file path
    ...
    output: grouped data by well name in dictionary
    """
    las = kwargs.get('las')
    dev = kwargs.get('dev')
    top = kwargs.get('top')
    pres = kwargs.get('pres')
    core = kwargs.get('core')
    drill = kwargs.get('drill')
    mud = kwargs.get('mud')

    wellname = filename(las)

    groups = {}
    
    for name in wellname:
        A = (name in filename(las))
        B = (name in filename(dev))
        C = (name in filename(top))
        D = (name in filename(pres))
        E = (name in filename(core))
        F = (name in filename(drill))
        G = (name in filename(mud))

        if A and B and C and D and E and F and G:
            groups[name] = {'las':las[filename(las).index(name)], 
                            'dev':dev[filename(dev).index(name)],
                            'top':top[filename(top).index(name)],
                            'pres':pres[filename(pres).index(name)],
                            'core':core[filename(core).index(name)],
                            'drill':drill[filename(drill).index(name)],
                            'mud':mud[filename(mud).index(name)]}

    return groups

# fucnction for random bright color list (color map)

def default_colors(n_color):
    """
    input:  n_color = number of color want to be generated
    ...
    output: list of color code (hex code) following 25 defaults.
    """

    import random

    defaults = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
                '#808080', '#FF0000', '#00FFFF', '#800000', '#008080',
                '#FFFF00', '#FFBF00', '#0000FF', '#808000', '#000080',
                '#00FF00', '#FF00FF', '#008000', '#800080', '#C0C0C0']

    if int(n_color) > 25:
        colors = defaults.copy()

        while len(colors) != int(n_color):
            scrap_code = [''.join([random.choice('0123456789ABCDEF') for i in range(2)]), '00', 'FF']
            random.shuffle(scrap_code)
            color = '#' + ''.join(scrap_code)
            if color not in colors:
                colors.append(color)
    else:
        colors = defaults[0:int(n_color)]
    
    return colors

# function for ordering formations from all well data

def merge_sequences(seq1,seq2):
    """
    This function is imported from another developer.
    Ref: https://stackoverflow.com/questions/14241320/interleave-different-length-lists-elimating-duplicates-and-preserve-order/49651477#49651477
    """
    
    from difflib import SequenceMatcher

    sm = SequenceMatcher(a = seq1, b = seq2)
    res = []
    for (op, start1, end1, start2, end2) in sm.get_opcodes():
        if op == 'equal' or op == 'delete': 
            #This range appears in both sequences, or only in the first one.
            res += seq1[start1:end1]
        elif op == 'insert':
            #This range appears in only the second sequence.
            res += seq2[start2:end2] 
        elif op == 'replace':
            #There are different ranges in each sequence - add both.
            res += seq1[start1:end1]
            res += seq2[start2:end2]
    return res

"""

True vertical depth calculation

"""

# function for depth extension

def df_exten(**kwargs):
    """
    input:  dataframe = well logging data in data frame
            toprange = start logging point
            step = step or rate of logging
    ...
    output: extended data frame (to measured depth = 0) filled with nan
    """

    dataframe = kwargs.get('dataframe')
    toprange = kwargs.get('toprange')
    step = kwargs.get('step')

    import pandas as pd
    import numpy as np

    # create empty data frame
    
    ex_depth = pd.DataFrame(np.arange(0, toprange, step), columns = ['MD'])

    # merge with well logging data

    dataframe.reset_index(inplace = True)
    dataframe = pd.concat([ex_depth, dataframe]).sort_values(by = ['MD'])
    dataframe.set_index('MD', inplace = True)

    return dataframe

# function for transfering the deviation data to another data frame

def merge_dev(**kwargs):
    """
    input:  dataframe = data in data frame
            dev = deviation data in data frame
    ...
    output: merged data with deviation data
    """
    
    dataframe = kwargs.get('well')
    dev = kwargs.get('dev')

    import pandas as pd

    # merge deviation file

    dataframe = pd.concat([dev, dataframe]).sort_values(by = ['MD'])
    print(dataframe)
    dataframe = dataframe.groupby('MD').max()

    # fill-in data using linear interpolation

    for col in ['AZI', 'ANG']:
        dataframe[col].interpolate(method = 'linear', limit_area = 'inside', inplace = True)
    dataframe.reset_index(inplace = True)

    return dataframe

# function for True Vertical Depth (TVD) computation by minimum curvature method

def mini_cuv(**kwargs):
    """
    input:  dataframe = data in data frame
            ag = air gap
    ...
    output: true vertical depth column
    """

    dataframe = kwargs.get('dataframe')
    ag = kwargs.get('ag')

    import numpy as np

    # setup parameters
    
    md = dataframe.MD
    prev_md = md.shift(periods = 1, fill_value = 0)
    diff_md = md - prev_md
    
    ang = dataframe.ANG
    prev_ang = ang.shift(periods = 1, fill_value = 0)
    diff_ang = ang - prev_ang
    
    azi = dataframe.AZI
    prev_azi = azi.shift(periods = 1, fill_value = 0)
    diff_azi = azi - prev_azi

    I1 = np.radians(ang)
    I2 = np.radians(prev_ang)

    dI = np.radians(diff_ang)
    dA = np.radians(diff_azi)

    # computation
    
    cos_theta = np.cos(dI) - (np.sin(I1) * np.sin(I2) * (1 - np.cos(dA)))
    theta = np.arccos(cos_theta)
    rf = ((2 / theta) * np.tan(theta/2)).fillna(0)

    dataframe['TVD'] = np.cumsum((diff_md / 2) * (np.cos(I1) + np.cos(I2) * rf))

    # remove air gab (ag)

    dataframe.TVD -= ag

    return dataframe

# function for column alignment of well logging data

def df_alignment(**kwargs):
    """
    input:  dataframe = well logging data in data frame
    ...
    output: arranged data frame of well logging 
    """
    dataframe = kwargs.get('dataframe')

    cols = dataframe.columns.tolist()
    cols = cols[:1] + cols[-2:] + cols[1:-2]
    dataframe = dataframe[cols]
    dataframe.set_index('MD', inplace = True)

    return dataframe

# function for formation top alignment

def setuptop(**kwargs):
    """
    input:  dataframe = formation top data in data frame
    ...
    output: setup formation top data
    """
    dataframe = kwargs.get('dataframe')

    last_tvd = dataframe.TVD.max()
    last_tvdss = dataframe.TVDSS.max()

    dataframe.dropna(inplace = True)
    dataframe.reset_index(drop = True, inplace = True)

    dataframe.rename(columns = {'TVD':'TVD_TOP', 'TVDSS':'TVDSS_TOP'}, inplace = True)

    dataframe['TVD_BOTTOM'] = dataframe['TVD_TOP'].shift(periods = -1)
    dataframe['TVDSS_BOTTOM'] = dataframe['TVDSS_TOP'].shift(periods = -1)

    dataframe.fillna(value = {'TVD_BOTTOM': last_tvd, 'TVDSS_BOTTOM': last_tvdss}, inplace = True)

    return dataframe





