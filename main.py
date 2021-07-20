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

files = {'las':['well logging file', ' (.las)'], 
         'dev':['deviation file', ' (.csv)'],
         'top':['formation top file', ' (.csv)'],
         'pres':['pressure test file', ' (.csv)'],
         'core':['core test file', ' (.csv)'],
         'drill':['drilling test file', ' (.csv)'],
         'mud':['mud-weight log file', ' (.csv)']}

# dirpaths = {}

# announce()

# print('According to your working directory,')
# print(', '.join(os.listdir(os.getcwd())))

# confirm = 'no'
# while confirm.lower() == 'no':

#     datapath = defpath(base=os.getcwd(), file='data', filetype='')
    
#     print('These sub-directories are found.')
#     print(', '.join(os.listdir(datapath)))

#     for name in files:
#         dirpaths[name] = defpath(base=datapath, file=files[name][0], filetype=files[name][1])

#     print('Please confirm your paths')
    
#     for name in dirpaths:
#         print('Your %s is: %s.' %(files[name][0], dirpaths[name]))

#     confirm = answer()

"""

Temporary code

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

Temporary code

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

# set directory to save files

save_folder = 'Saved files'
save_path = os.path.join(os.getcwd(), save_folder)

if not os.path.isdir(save_path):
    os.makedirs(save_path)

if __name__ == '__main__':
    
    # print(wells[0].las.curves)
    # print(wells[0].df)
    # print(wells[0].dev)
    # print(wells[0].top)
    # print(wells[0].pres)
    # print(wells[0].core)
    # print(wells[0].drill)
    # print(wells[0].mud)

    # for well in wells:
    #     print(well.name)


    
    pass
