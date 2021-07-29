# import downloaded modules

import os, glob
import lasio # well logging management
import pandas as pd
import numpy as np

# import devoloped modules

from mem.note import announce
from mem.well import *
from mem.audit import *
from mem.mstati import *

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

    # merge deviation data to another data frame
    
    dataframes = [well.df, well.top, well.pres, well.core, well.drill, well.mud] # setup dataframe list
    well.df, well.top, well.pres, well.core, well.drill, well.mud = [merge_dev(dataframe=dataframe, dev=well.dev) for dataframe in dataframes]

    # True Vertical Depth (TVD) calculation using minimum curvature method

    dataframes = [well.df, well.top, well.pres, well.core, well.drill, well.mud] # reset dataframe list
    well.df, well.top, well.pres, well.core, well.drill, well.mud = [mini_cuv(dataframe=dataframe, ag=well.ag) for dataframe in dataframes]

    # True Vertical Depth Sub-Sea (TVDSS) calculation
    
    for dataframe in dataframes:
        if well.type == 'onshore':
            dataframe['TVDSS'] = dataframe.TVD - well.gl
        else:
            dataframe['TVDSS'] = dataframe.TVD

    # well logging data alignment

    well.df = df_alignment(dataframe=well.df)

    # setup formation top data after TVD calculation

    well.top = setuptop(dataframe=well.top)

    # setup for the rest of data after TVD calculation

    for dataframe in [well.pres, well.core, well.drill, well.mud]:
        dataframe.dropna(inplace = True)
        dataframe.reset_index(drop = True, inplace = True)

    # update las file

    descrs = ['True Vertical Depth', 'True Vertical Depth Sub-Sea',
                'Well Deviation in Azimuth', 'Well Deviation in Angle']
    cols = ['TVD', 'TVDSS', 'AZI', 'ANG']
    units = ['m', 'm', 'degree', 'degree']

    for i, col, unit, descr in zip(range(1,5), cols, units, descrs):
        well.las.insert_curve(i, col, well.df[col], unit=unit, descr=descr, value='')

    print('TVD and TVDSS are calculated for well %s' %well.name)

"""

Bad zone elimination

"""

# confidential interval factor for Bad hole flag cut-off

cif = 0.75 # changable (0.00-1.00, default = 0.75)

for well in wells:

    # construct bad hole flag

    well.df, well.las, well.other['ci'] = bhf_cal(dataframe=well.df, las=well.las, cif=cif)

    # replace nan for bad hole zone

    well.df = bhf_control(dataframe=well.df)

    # replace nan for low vp/vs ratio

    well.df = ratio_control(dataframe=well.df)

"""

Data normalization

"""

# prepare inputs for normalization

ref_high = np.mean([well.df.GR.quantile(0.95) for well in wells])
ref_low = np.mean([well.df.GR.quantile(0.05) for well in wells])

# GR normalization

for well in wells:
    well.df, well.las = norm_gr(dataframe=well.df, las=well.las, ref_high=ref_high, ref_low=ref_low)

"""

Data systhetic

"""

# create dataset for data training

dataset = set_data(dataframes=[well.df for well in wells])

# synthesize the data

for well in wells:
    print('The data of well %s are being synthesized.' %well.name)
    well.df, well.las = synthesis(dataframe=well.df, las=well.las, dataset=dataset)
print('Data synthesis is done.')

"""

3.Mechanical Stratigraphy

"""








# set directory to save files

save_folder = 'Saved files'
save_path = os.path.join(os.getcwd(), save_folder)

if not os.path.isdir(save_path):
    os.makedirs(save_path)

"""

TEST ZONE

"""

# Function for initial plot for first inspection

def inspection(dataframe, las):
    """
    inputs: dataframe = well logging in dataframe
            las = well logging in las file
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # create figure

    fig, axis = plt.subplots(nrows = 1, ncols = len(dataframe.columns), figsize = (30,20), sharey = True)
    
    units = [curve.unit for curve in las.curves]
    index_unit = units.pop(0)

    # plot setting for all axis

    top_depth = dataframe.index.min()
    bottom_depth = dataframe.index.max()

    axis[0].set_ylabel('MD[%s]' %index_unit, fontsize = 15)

    for ax, col, unit in zip(axis, dataframe.columns, units):
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black')
        ax.set_xlabel(col + '\n[%s]' %unit, fontsize = 15)

        if (col == 'RT') or (col == 'MSFL'):
            ax.plot(dataframe[col], dataframe.index, linewidth = '0.5')
            ax.set_xscale('log')

        elif col == 'BHF':

            dataframe['bhf'] = np.nan
            dataframe.loc[dataframe.BHF == 'BAD', 'bhf'] = 1
            ax.fill_betweenx(dataframe.index, 0, dataframe.bhf, color = 'red', capstyle = 'butt', linewidth = 1, label = 'BAD')
            dataframe.drop(columns = ['bhf'], inplace = True)

        elif col in ['TVD', 'TVDSS', 'AZI', 'ANG', 'CAL', 'BS']:
            ax.plot(dataframe[col], dataframe.index, linewidth = '1.0')
        
        else:
            ax.plot(dataframe[col], dataframe.index, linewidth = '0.5')

    fig.tight_layout()

    plt.show()

# Plot available curves

# test = wells[1].df.copy()
# test_las = wells[1].las

# inspection(dataframe=test, las=test_las)

# print(wells[0].df[['GR', 'GR_NORM']].describe())
# print(wells[1].df[['GR', 'GR_NORM']].describe())
# print(wells[2].df[['GR', 'GR_NORM']].describe())

print(dataset)
print(wells[0].las.curves)
print(wells[0].df)
print(wells[0].df.describe())
