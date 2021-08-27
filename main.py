# import downloaded modules

import os, glob
import lasio # well logging management
import pandas as pd
import numpy as np

# import devoloped modules

from mem.note import announce
from mem.well import *          # well class
from mem.audit import *         # data audit
from mem.mstati import *        # mechanical stratigraphy
from mem.obp import *           # overburden stress
from mem.pp import *            # pore pressure
from mem.rsep import *          # rock strength and elastic properties
from mem.stress import *        # minimum and maximum horizontal stresses
from mem.failana import *       # failure analysis
from mem.plots import *         # all model plots

"""

1.Data Audit & 2.Framework Model

"""

# define directories

files = {'las':['well logging file', ' (.las)'], 
         'dev':['deviation file', ' (.csv)'],
         'config':['configuration file', ' (.csv)'],
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

Temporary code

"""

def temppath(directory):
    return os.path.join(os.getcwd(), 'Sirikit field', directory)

dirpaths = {'las': temppath('Las files'), 
            'dev': temppath('Deviations'),
            'config': temppath('Configs'),
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

groupfiles = grouping(las=filepaths['las'], dev=filepaths['dev'], config=filepaths['config'],
                        top=filepaths['top'], pres=filepaths['pres'],core=filepaths['core'], 
                        drill=filepaths['drill'], mud=filepaths['mud'])

# import all files as borehole object

print('The number of wells is %d.' %len(groupfiles))

wells = []

for name, color in zip(groupfiles, default_colors(len(groupfiles))):
    well = borehole(las=lasio.read(groupfiles[name]['las']), dev=pd.read_csv(groupfiles[name]['dev']),
                    config=pd.read_csv(groupfiles[name]['config']), top=pd.read_csv(groupfiles[name]['top']),
                    pres=pd.read_csv(groupfiles[name]['pres']), core=pd.read_csv(groupfiles[name]['core']),
                    drill=pd.read_csv(groupfiles[name]['drill']), mud=pd.read_csv(groupfiles[name]['mud']),
                    color=color)
    if well.completion == True:
        wells.append(well)
        print('The data of well %s are imported' %well.name)
    else:
        print('The data of well %s are not imported' %well.name)

# setup well configuration

for well in wells:
    well.type = well.config.iloc[0].TYPE.lower()
    well.kb = well.config.iloc[0].KB
    well.ml = well.config.iloc[0].ML
    print('Well %s is in %s field' %(well.name, well.type))

    if well.type == 'onshore':
        well.gl = well.config.iloc[0].GL
        well.ag = well.kb - well.gl
        print('Ground level = %.2f, Mudline density = %.2f, Air gap = %.2f' %(well.gl, well.ml, well.ag))
    else:
        well.wl = well.config.iloc[0].WL
        well.ag = well.kb
        print('Water level = %.2f, Mudline density = %.2f, Air gap = %.2f' %(well.wl, well.ml, well.ag))

"""

Temporary code 2

"""

# for well in wells:
#     well.type = 'onshore'
#     well.kb = 62.91
#     well.gl = 53.62
#     well.ml = 1.32
#     well.ag = well.kb - well.gl
#     print('Well %s; Ground level = %.2f, Mudline density = %.2f, Air gap = %.2f' %(well.name, well.gl, well.ml, well.ag))

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

    well.df.reset_index(inplace=True) # well logging data
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

3.Mechanical stratigraphy

"""

# calculate volume of clay, effective porosity, and lithology

for well in wells:
    well.df, well.las = vcl_cal(dataframe=well.df, las=well.las)
    well.df, well.las = phie_cal(dataframe=well.df, las=well.las)
    well.df, well.las = litho_cal(dataframe=well.df, las=well.las)
    print('Volume of clay, effective porosity, and lithology are calculated for well %s' %well.name)

"""

4.Overburden stress

"""

# calculate extrapolation equation coefficient

mls = [well.ml for well in wells]
surfaces = [0 if well.type == 'onshore' else well.wl for well in wells]
dataframes = [well.df for well in wells]
coefs = extracoef(dataframes=dataframes, mls=mls, surfaces=surfaces)

# calculate overburden stress

for well, surface in zip(wells, surfaces):
    well.df, well.las = denextra(dataframe=well.df, las=well.las, ml=well.ml, surface=surface, coefs=coefs)
    
    if well.type == 'onshore':
        well.df, well.las = obp_cal(dataframe=well.df, las=well.las)
    else:
        well.df, well.las = obp_cal(dataframe=well.df, las=well.las, wl=well.wl)

    print('Overburden pressure and its gradient are calculated for well %s' %well.name)

"""

5.Pore presure

"""

# calculate pore pressure

for well in wells:
    well.df, well.las = pp_cal(dataframe=well.df, las=well.las)
    print('Pore pressure and its gradient are calculated for well %s' %well.name)

"""

6.Rock Strength & Elastic Properties

These equations are customizable through rsepeq.py:
1. Static Young's modulus
2. Unconfined compressive strength [UCS]

"""

# calculate YME, PR, UCS, FANG, and TSTR

for well in wells:
    well.df, well.las = yme_cal(dataframe=well.df, las=well.las)
    well.df, well.las = pr_cal(dataframe=well.df, las=well.las)
    well.df, well.las = ucs_cal(dataframe=well.df, las=well.las)
    well.df, well.las = fang_cal(dataframe=well.df, las=well.las)
    well.df, well.las = tstr_cal(dataframe=well.df, las=well.las)
    print('Rock strength and elastic properties are calculated for well %s' %well.name)

"""

8.Minimum horizontal stress & 9.Maximum horizontal stress
** P.S. 7.Horizontal Stress direction (skipped) **

"""

# calculate horizontal stresses

for well in wells:
    maxstrain, minstrain = strain_cal(dataframe=well.df, drill=well.drill)
    maxstrain, minstrain = 0.0002, 0.0001
    well.df, well.las = stress_cal(dataframe=well.df, las=well.las, maxstrain=maxstrain, minstrain=minstrain)
    print('Maximum and minimum horizontal stresses are calculated for well %s' %well.name)
    print('With maximum and minimum tectonic strains, %.5f and %.5f.' %(maxstrain, minstrain))

"""

10.Failure analysis

"""

# create mud window and breakout width

for well in wells:
    well.df, well.las = merge_mud(dataframe=well.df, las=well.las, mud=well.mud)
    well.df, well.las = mudwindow_cal(dataframe=well.df, las=well.las)
    well.df, well.las = wbo_cal(dataframe=well.df, las=well.las)
    print('Mud window components are calculated for well %s' %well.name)

"""

All model plots

"""

# set directory to save files

savefolder = 'Saved files'
savepath = os.path.join(os.getcwd(), savefolder)

if not os.path.isdir(savepath):
    os.makedirs(savepath)

for well in wells:
    print('Model is being created for well %s' %well.name)

    # mem plot

    mem_plot(dataframe=well.df, formtop=well.top, pres=well.pres, core=well.core,
                forms=allforms, toprange=well.range[0], botrange=well.range[1],
                wellname=well.name, savepath=savepath)

    # neutron-density crossplot

    nd_plot(dataframe=well.df, formtop=well.top, forms=allforms, toprange=well.range[0], 
                botrange=well.range[1], wellname=well.name, savepath=savepath)

    # overburden pressure plot

    obp_plot(dataframe=well.df, wellname=well.name, savepath=savepath)

# input for group plot

dataframes = [well.df for well in wells]
formtops = [well.top for well in wells]
wellids = {well.name:well.color for well in wells}

# histogram of gamma ray normalization

grnorm_plot(dataframes=dataframes, wellids=wellids, savepath=savepath)

# boxes plots for QC

for form in allforms:
    boxes_plot(dataframes=dataframes, formtops=formtops, wellids=wellids,
                    formation=form, savepath=savepath)

# all overburden stress plot

allobp_plot(dataframes=dataframes, wellids=wellids, savepath=savepath)

"""

Export data to .las and .csv files

"""

# export files

for well in wells:
    # well.df, well.las = textcol_drop(dataframe=well.df, las=well.las) # drop text data 
    well.export(savepath=savepath)
    print('Well %s data are export as .las and .csv' %well.name)
