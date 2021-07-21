# import downloaded modules

import os, glob
import lasio # well logging management
import pandas as pd 

# import devoloped modules

from mem.note import announce
from mem.well import *
from mem.audit import *

"""

1.Data Audit & 2.Framework Model

"""

# define directories

files = {'las':['well logging file', ' (.las)'], 
         'dev':['deviation file', ' (.csv)'],
         'top':['formation top file', ' (.csv)'],
         'pres':['pressure test file', ' (.csv)'],
         'core':['core test file', ' (.csv)'],
         'drill':['drilling test file', ' (.csv)'],
         'mud':['mud-weight log file', ' (.csv)']}

# announce()

# dirpaths = {}

# print('According to your working directory,')
# print(', '.join(os.listdir(os.getcwd())))

# while True:
#     datapath = defpath(base=os.getcwd(), file='data', filetype='')
    
#     print('These sub-directories are found.')
#     print(', '.join(os.listdir(datapath)))

#     for name in files:
#         dirpaths[name] = defpath(base=datapath, file=files[name][0], filetype=files[name][1])

#     for name in dirpaths:
#         print('The path of your %s is: %s.' %(files[name][0], dirpaths[name]))

#     print('Please confirm your paths')
    
#     if confirm():
#         break
#     else:
#         continue

"""

Temporary code 1

"""

def temppath(directory):
    return os.path.join(os.getcwd(), 'Sirikit field', directory)

dirpaths = {'las': temppath('Las files'), 
            'dev': temppath('Deviations'),
            'top': temppath('Formation tops'),
            'pres': temppath('Pressures'),
            'core': temppath('Cores'),
            'drill': temppath('Drilling test'),
            'mud': temppath('Mud weights')}

for name in dirpaths:
    print('Your %s is: %s.' %(files[name][0], dirpaths[name]))

"""

Temporary code 1

"""

# generate file path

filepaths = {}

for name in dirpaths:
    if name == 'las':
        filepaths[name] = glob.glob(os.path.join(dirpaths[name], '*.las'))
    else:
        filepaths[name] = glob.glob(os.path.join(dirpaths[name], '*.csv'))

# group the files by well name or prefix

groupfiles = grouping(las=filepaths['las'], dev=filepaths['dev'], top=filepaths['top'], 
                        pres=filepaths['pres'],core=filepaths['core'], drill=filepaths['drill'], 
                        mud=filepaths['mud'])

# import all files as borehole object

print('The number of wells is %d.' %len(groupfiles))

wells = []

for name, color in zip(groupfiles, default_colors(len(groupfiles))):
    well = borehole(las=lasio.read(groupfiles[name]['las']), dev=pd.read_csv(groupfiles[name]['dev']),
                    top=pd.read_csv(groupfiles[name]['top']), pres=pd.read_csv(groupfiles[name]['pres']),
                    core=pd.read_csv(groupfiles[name]['core']), drill=pd.read_csv(groupfiles[name]['drill']),
                    mud=pd.read_csv(groupfiles[name]['mud']), color=color)
    if well.completion == True:
        wells.append(well)
        print('The data of well %s are imported' %well.name)
    else:
        print('The data of well %s are not imported' %well.name)

# define field parameters

# while True:

#     field_type = input('What is this oil field type [Onshore/Offshore]: ').strip()

#     if field_type.lower() not in ['onshore', 'offshore']:
#         print('Please type only either \"Onshore\" or \"Offshore\"')
#         continue
    
#     for well in wells:
#         print('Please type basic information for well %s' %well.name)
        
#         well.type = field_type.lower()
#         well.kb = float(input('Kelly Bushing depth (KB level to sea level) [m]: ').strip())
        
#         if field_type.lower() == 'onshore':
#             well.gl = float(input('Ground elevetion (ground level to sea level) [m]: ').strip())
#             well.ag = well.kb - well.gl
#         else:
#             well.wl = float(input('Water depth (sea level to seafloor level) [m]: ').strip())
#             well.ag = well.kb
        
#         well.ml = float(input('Mudline density (density at ground level) [g/c3]: ').strip())

#     print('Please confirm these inputs; Field type = %s' %(field_type))
#     for well in wells:
#         if field_type.lower() == 'onshore':
#             print('Well %s; Ground level = %.2f, Mudline density = %.2f, Air gap = %.2f' %(well.name, well.gl, well.ml, well.ag))
#         else:
#             print('Well %s; Water level = %.2f, Mudline density = %.2f, Air gap = %.2f' %(well.name, well.wl, well.ml, well.ag))

#     if confirm():
#         break
#     else:
#         continue

"""

Temporary code 2

"""

for well in wells:
    well.type = 'onshore'
    well.kb = 62.91
    well.gl = 53.62
    well.ml = 1.32
    well.ag = well.kb - well.gl
    print('Well %s; Ground level = %.2f, Mudline density = %.2f, Air gap = %.2f' %(well.name, well.gl, well.ml, well.ag))

"""

Temporary code 2

"""

# extract formation names

formnames = []

for well in wells:
    if formnames == []:
        for form in well.top.dropna().FORM:
            formnames.append(form)
    else:
        formnames = merge_sequences(formnames, list(well.top.dropna().FORM))

# Setup identity color for all formations

allforms = {}

for form, color in zip(formnames, default_colors(len(formnames))):
    allforms[form] = color

print('All formations in this field are: %s.' %', '.join(allforms))

"""

True vertical depth calculation

"""

for well in wells:
    
    # extend well logging data frame

    well.df = df_exten(dataframe=well.df, toprange=well.range[0], step=well.range[2])
    
    # setup MD column

    well.df.reset_index(inplace = True) # well logging data
    well.top['MD'] = well.top.TOP # formation top

    for dataframe in [well.df, well.top, well.pres, well.core, well.drill, well.mud]:

        print(dataframe)
        print('\n')

        # merge deviation data to another data frame

        dataframe = merge_dev(dataframe=dataframe, dev=well.dev)

        # True Vertical Depth (TVD) calculation using minimum curvature method

        dataframe = mini_cuv(dataframe=dataframe, ag=well.ag)

        # True Vertical Depth Sub-Sea (TVDSS) calculation

        if well.type == 'onshore':
            dataframe['TVDSS'] = dataframe.TVD - well.gl
        else:
            dataframe['TVDSS'] = dataframe.TVD
    
        # print(well.df)
        # print(well.top)
        # print(well.pres)
        # print(well.core)
        # print(well.drill)
        # print(well.mud)

    # well logging data alignment

    well.df = df_alignment(dataframe=well.df)

    # setup formation top data after TVD calculation

    well.top = setuptop(dataframe=well.top)

    # setup for the rest of data after TVD calculation

    for dataframe in [well.pres, well.core, well.drill, well.mud]:
        dataframe.dropna(inplace = True)
        dataframe.reset_index(drop = True, inplace = True)

# set directory to save files

save_folder = 'Saved files'
save_path = os.path.join(os.getcwd(), save_folder)

if not os.path.isdir(save_path):
    os.makedirs(save_path)

if __name__ == '__main__':



    
    pass
