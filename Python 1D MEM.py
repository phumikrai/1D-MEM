"""

1.Data Audit & 2.Framework Model

"""
# Support modules

import glob, os, re, random
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import scipy.stats as st
import lasio # Las file reader module
from datetime import datetime
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
from scipy import constants

# fix "SettingWithCopyWarning" of Pandas modules

pd.options.mode.chained_assignment = None  # default='warn'

"""

Import data

"""

note = """
1.) This is for modeling based on these assumption;
    - Isotropic homogeneous medium
    - Vertical well or lowly deviated well
    - No overpressure
    - Sand-shale or sedimentary basin
    - Oil reservoir
    - Using statistic decisions or machine learning techniques
    - Temperature criteria is ignored.

2.) The working directory should contain the data for modeling as sub-directory. 

3.) All data for modeling including;
    - well logging (.las)
    - Deviation (.csv)
    - Formation top (.csv)
    - Mud-weight log (.csv)
    - Pressure test (.csv)
    - Core test (.csv)
    - Drilling test (.csv)
    must be separated as sub-directory of the data directory. 

For example;
    - Working directory is "Drive:/Working/".
    - All data for modeling directory is "Drive:/Working/Data/".
    - Well logging file directory is "Drive:/Working/Data/Well logging/" as Sub-directory of the data directory.
    - Deviation file directory is "Drive:/Working/Data/Deviation/" as Sub-directory of the data directory.
    - Formation top file directory is "Drive:/Working/Data/Formation top/" as Sub-directory of the data directory.
    - Mud-weight log file directory is "Drive:/Working/Data/Mud weight/" as Sub-directory of the data directory.
    - Pressure test file directory is "Drive:/Working/Data/Pressure test/" as Sub-directory of the data directory.
    - Core test file directory is "Drive:/Working/Data/Core test/" as Sub-directory of the data directory.
    - Drilling test file directory is "Drive:/Working/Data/Drilling test/" as Sub-directory of the data directory.

4.) Well name should be set as prefix for each file. Its name will cause file ordering and file pairing for each file of that well.
    
Assuming that; 
well name is "Well-01" (Noted: No underscore ('_') be contained in well name) So this name should be set as prefix followed by underscore ('_') for each modeling input file like this "Well-01_(...Specific name for file type indication...)". 

For example;
    - well logging      is   "Well-01_las"    
    - Deviation         is   "Well-01_dev"
    - Formation top     is   "Well-01_top"
    - Mud-weight log    is   "Well-01_mw"
    - Pressure test     is   "Well-01_pp"
    - Core test         is   "Well-01_core"
    - Drilling test     is   "Well-01_test"

5.: Required data and file format;

- Well logging files including all necessary curves for 1D MEM such; 
    Measured depth                  (MD or DEPTH)   in meter unit [m] 
    Bitsize                         (BS)            in inch unit [in] 
    Caliper                         (CAL)           in inch unit [in] 
    Gamma ray                       (GR)            in American Petroleum Institute unit [API]
    Density                         (RHOB)          in grams per cubic centimetre unit [g/c3]
    Neutron porosity                (NPHI)          in fractional unit [V/V]
    Deep resistivity                (RT)            in ohm-meter unit [ohm-m]
    Shallow resistivity             (MSFL)          in ohm-meter unit [ohm-m]
    Compressional wave slowness     (DTC)           in microseconds per foot unit [us/ft]
    Shear wave slowness             (DTS)           in microseconds per foot unit [us/ft]

- Deviation files including; 
    Measured depth                  (MD)            in meter unit [m]           
    Azimuth                         (AZIMUTH)       in degree unit [degree]     
    Inclination or angle            (ANGLE)         in degree unit [degree]     

- Formation top files including;
    Formation name                  (FORMATIONS)                                                     
    Formation Top depth             (TOP)           in meter unit [m]           
    Formation Bottom depth          (BOTTOM)        in meter unit [m]

- Mud-weight log files including; 
    Measured depth                  (DEPTH)         in meter unit [m]           
    Mud weight                      (MUDWEIGHT)     in mud weight unit [ppg]

- Pressure test files including; 
    Measured depth                  (DEPTH)         in meter unit [m]           
    Pressure                        (PRESSURE)      in pound per square inch unit [psi]

- Core test files including; 
    Measured depth                  (DEPTH)         in meter unit [m]           
    Young's modulus                 (YME)           in pound per square inch unit [psi]
    Poisson's ratio                 (PR)            in a fractional number [unitless]
    Uniaxial compressive strength   (UCS)           in pound per square inch unit [psi]
    Friction angle                  (FANG)          in degree unit [degree]

- Drilling test files including; 
    Measured depth                  (DEPTH)         in meter unit [m]           
    Test type                       (TEST)          formation test such FIT, LOT, Minifrac, and etc.
    Result or value                 (RESULT)        in mud weight unit [ppg]
"""
print('Welcome to 1D Mechanical Earth Modeling by Python (Python 1D MEM).')
print('Please take note on this;')
print(note)

# Function for decision confirmation

def decision_confirm():
    """
    This function will ask a question to user for decision confirmation.
    The output of this function is either 'Yes' or 'No' only.
    """
    while True:
        confirm = input('Are you sure? [Yes/No]: ').strip()

        if confirm.lower() == 'yes':
            break
        
        elif confirm.lower() == 'no':
            break

        else:
            print('Please confirm again!')

    return confirm

# Function for input file path

def define_path(base_path, file, filetype):
    """
    This function is for getting file path from user input definition.
    base_path = path of base directory
    file = file or file name in folder
    filetype = type of file in folder
    """
    while True:

        print('Which one is your %s directory?' %file)
        folder = input('Please indicate your %s directory name%s: ' %(file, filetype)).strip()

        if folder == '':
            print('Please type the directory name!')

        else:
            folder_path = os.path.join(base_path, folder)

            if os.path.isdir(folder_path):
                break

            else:
                print('Please try again, your directory \'%s\' is not found!' %folder)

    return folder_path

# Setup data directory

paths = []

files = {'las':['well logging file', ' (.las)'], 
         'dev':['deviation file', ' (.csv)'],
         'top':['formation top file', ' (.csv)'],
         'mud':['mud-weight log file', ' (.csv)'],
         'pres':['pressure test file', ' (.csv)'],
         'core':['core test file', ' (.csv)'],
         'drill':['drilling test file', ' (.csv)']}

cwd_dir_list = ', '.join(os.listdir(os.getcwd()))
confirm = 'no'

print('According to your working directory,')
print(cwd_dir_list)

while confirm.lower() == 'no':

    data_path = define_path(os.getcwd(), 'data', '')
    
    data_dir_list = ', '.join(os.listdir(data_path))
    print(data_dir_list)
    print('These sub-directories are found')

    for name in files:
        path = define_path(data_path, files[name][0], files[name][1])
        paths.append(path)

    for path, name in zip(paths, files):
        print('Your %s directory is: %s.' %(files[name][0], path))

    print('Please confirm your input.')
    confirm = decision_confirm()

# Function for pairing the data and eliminating the incomplete

def pairing_files(las_files, sub_files):
    """
    This function is going to pairing the data (las files, dev files and top files) and disable the incompleted one.
    las_files = list of las files with paths
    sub_files = list of sub-files with paths such dev, top, mud, pres, core, and drill files.
    """

    paired_files = []

    # pairing files

    for las in las_files:
        for dev in sub_files[0]:
            for top in sub_files[1]:
                for mud in sub_files[2]:
                    for pres in sub_files[3]:
                        for core in sub_files[4]:
                            for drill in sub_files[5]:

                                pairing = [las, dev, top, mud, pres, core, drill]
                                pairing_name = [os.path.basename(file).split('_', 1)[0].lower() for file in pairing]

                                if all(element == pairing_name[0] for element in pairing_name):
                                    paired_files.append(pairing)

    return paired_files

# Generate file path

las_files = glob.glob(os.path.join(paths[0], '*.las'))
sub_files = []

for path in paths[1:]:
    sub_file = glob.glob(os.path.join(path, '*.csv'))
    sub_files.append(sub_file)

# Pairing files

paired_files = pairing_files(las_files, sub_files)

# Import files

lases, df_lases, devs, tops, muds, press, cores, drills = [list() for i in range(8)] # file storages

for files in paired_files:

    # Well logging data

    las = lasio.read(files[0])
    df = las.df() # convert to panda data frame
    df = df.rename_axis('MD')

    for store, file in zip([lases, df_lases], [las, df]):
        store.append(file)

    # deviation, formation top, mud-weight log, pressure test, core test, drilling test data

    for store, file in zip([devs, tops, muds, press, cores, drills], files[1:]):
        store.append(pd.read_csv(file)) 

    name = las.well['WELL'].value
    print('The data of well %s are imported' %name)

# Set directory to save files

sav_folder = 'Saved files'
sav_path = os.path.join(data_path, sav_folder)

if not os.path.isdir(sav_path):
    os.makedirs(sav_path)

# Function for random bright color list (color map)

def default_colors(n_color):
    """
    This function can generate the list of color code (hex code) following 25 defaults.
    n_color = a number of color
    """
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

# Setup well names with color identity

well_names = {}

for las, color in zip(lases, default_colors(len(lases))):
    well_names[las.well['WELL'].value] = color

print('The number of wells is %d.' %len(well_names))
print('Well names are %s.' %', '.join(well_names))

# Define standard curve alias for well process

alias = {
'BS' : ['BS', 'BIT'],
'CAL' : ['CAL', 'CALI', 'CALS', 'CLDC'],
'GR' : ['GR', 'GRGC', 'GAM'],
'RHOB' : ['RHOB', 'DEN', 'DENS'],
'NPHI' : ['NPHI', 'NPOR'],
'MSFL' : ['MSFL', 'R20T', 'RSHAL', 'RESS'],
'ILM' : ['ILM', 'R30T', 'R40T', 'R60T', 'RESM'],
'RT' : ['RT', 'R85T', 'LLD', 'RESD'],
'DTC' : ['DTC', 'DT35', 'DT'],
'DTS' : ['DTS', 'DTSM', 'DTSRM', 'DTSXX_COL', 'DTSYY_COL'],
'PEF' : ['PEF', 'PE', 'Pe', 'PDPE']
}

# Define curve name for modeling

mem_curves = ['CAL', 'BS', 'GR', 'RHOB', 'NPHI', 'RT', 'MSFL', 'DTC', 'DTS']

# Define based curve names

based_curves = ['TVD', 'TVDSS', 'AZIMUTH', 'ANGLE', 'BHF']

# Define non affected curves and affected curves for synthetic stage

non_affected = ['RT', 'MSFL', 'GR_NORM']
affected = ['NPHI', 'RHOB', 'DTC', 'DTS'] # element index refers to synthetic ordering

# Check available curves

data_ranges = []

for las, name in zip(lases, well_names):
    start = las.well['STRT'].value
    stop = las.well['STOP'].value
    data_ranges.append((start, stop))

    print('Well %s has logging data between %.2f and %.2f in measured depth (MD).' %(name, start, stop))
    print('Available curves are: \n%s' %', '.join([curve.mnemonic for curve in las.curves]))

    curves = [curve.mnemonic for curve in las.curves]
    extracted = []

    for curve in curves:
        for key, values in alias.items():
            if (curve.lower() in [value.lower() for value in values]) & (key in mem_curves):
                extracted.append(key)

    if set(extracted) == set(mem_curves):
        print('All necessary curves are completed')

    else:
        print('The data is incompleted.')
        
        if len(set(extracted).difference(set(mem_curves))) == 1:
            print('Curve %s is missing.' %', '.join([curve for curve in set(mem_curves) - set(extracted)]))

        else:
            print('Curves %s are missing.' %', '.join([curve for curve in set(mem_curves) - set(extracted)]))

# Function for ordering formations from all well data

def merge_sequences(seq1,seq2):
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

# Apply function to ordering all formations

forms = []

for top in tops:
    if forms == []:
        for form in top.dropna().FORMATIONS:
            forms.append(form)
    else:
        forms = merge_sequences(forms, list(top.dropna().FORMATIONS))

# Setup all formations with color identity

all_forms = {}

for form, color in zip(forms, default_colors(len(forms))):
    all_forms[form] = color

# Show all selectable formations

print('All formations in this field are: %s.' %', '.join(all_forms))

# Define selected formations in this project to focus

confirm = 'no'

while confirm.lower() == 'no':

    selected = []

    select = input('Which one is your interested formation? [Comma for multi-input]: ').strip()

    if select == '':
        print('Please type formation name.')
        continue

    else:
        for form in all_forms:
            if form.lower() in [name.strip().lower() for name in select.split(',')]:
                selected.append(form)
        
        if selected == []:
            print('Please try again, formation \"%s\" is not found.' %select)
            continue

        else:
            print('Now, only formations \"%s\" will be your selected formations' %', '.join(selected))

            confirm = decision_confirm()

# setup selected formations with color identity

selected_forms = {}

for form in selected:
    if form in all_forms:
        selected_forms[form] = all_forms[form]

# Define field parameters to adjust (remove air gap) the well logging by oil field type in the next step

confirm = 'no'

while confirm.lower() == 'no':

    ground_levels = [] # for onshore field only.
    water_levels = [] # for offshore field only.
    RHOmls = [] # mudline density or surface density for density extrapolation
    air_gaps = []

    field_type = input('What is this oil field type [Onshore/Offshore]: ').strip()

    if field_type.lower() == 'onshore':
        for name in well_names:
            print('Please type basic information for well %s' %name)

            kb = float(input('Kelly Bushing depth (KB level to sea level) [m]: ').strip())
            gl = float(input('Ground elevetion (ground level to sea level) [m]: ').strip())
            RHOml = float(input('Mudline density (density at ground level) [g/c3]: ').strip())
            ground_levels.append(gl)
            RHOmls.append(RHOml)
            air_gaps.append(kb - gl)

            water_levels.append(0)

        print('Please confirm these; Field type = %s' %(field_type))
        for name, gl, RHOml, ag in zip(well_names, ground_levels, RHOmls, air_gaps):
            print('Well %s; Ground level = %.2f, Mudline density = %.2f, Air gap = %.2f' %(name, gl, RHOml, ag))
        
        confirm = decision_confirm()
    
    elif field_type.lower() == 'offshore':
        for name in well_names:
            print('Please type basic information for well %s' %name)

            kb = float(input('Kelly Bushing depth (KB level to sea level) [m]: ').strip())
            wl = float(input('Water depth (sea level to seafloor level) [m]: ').strip())
            RHOml = float(input('Mudline density (density at sea level) [g/c3]: ').strip())
            water_levels.append(wl)
            RHOmls.append(RHOml)
            air_gaps.append(kb)

            ground_levels.append(0)

        print('Please confirm these; Field type = %s' %(field_type))
        for name, wl, RHOml, ag in zip(well_names, water_levels, RHOmls, air_gaps):
            print('Well %s; Water level = %.2f, Mudline density = %.2f, Air gap = %.2f' %(name, wl, RHOml, ag))
        
        confirm = decision_confirm()
    
    else:
        print('Please type only either \"Onshore\" or \"Offshore\"')
        continue

"""

Depth data conversion and manipulation

"""

# Function for depth extension

def exten_dep(las, df_las):
    """
    This function is for extending the logging data depth to staring point (measured depth = 0) for overburden stress computation.
    las = las file (.las) of the well data.
    df_las = las input in pandas data frame contains depth column in measured depth (MD) as dataframe index.
    """
    # input parameters

    start = las.well['STRT'].value
    step = las.well['STEP'].value

    # generate extended depth and merge with well logging data

    ex_depth = pd.DataFrame(np.arange(0, start, step), columns = ['MD'])

    df_las.reset_index(inplace = True)
    df_las = pd.concat([ex_depth, df_las]).sort_values(by = ['MD'])
    df_las.set_index('MD', inplace = True)

    return df_las

# Function for transfering the deviation data to well logging and formation top data

def merge_dev(df, dev):
    """
    This function is merging well deviation survey data both azimuth direction (AZIMUTH) and inclination angle (ANGLE) to well logging and formation top data.
    df = data in data frame which will be merged with deviation file
    dev = Deviation survey data in pandas data frame which contains:
        1. Measured depth (MD) in column name "MD"
        2. Azimuth direction (AZIMUTH) in column name "AZIMUTH"
        3. Inclination angle (ANGLE) in column name "ANGLE"
    """
    # merge deviation file

    df = pd.concat([dev, df]).sort_values(by = ['MD'])
    df = df.groupby('MD').max()

    # fill-in data using linear interpolation
    
    for col in ['AZIMUTH', 'ANGLE']:
        df[col].interpolate(method = 'linear', limit_area = 'inside', inplace = True)
    
    df.reset_index(inplace = True)

    return df

# Function for True Vertical Depth (TVD) computation by minimum curvature method

def tvd_cal(df, ag):
    """
    True Vertical Depth (TVD) computation function using minimum curvature survey calculation method
    df = data in data frame contains:
            1. Measured depth (MD) in column name "MD"
            2. Azimuth direction (AZIMUTH) in column name "AZIMUTH"
            3. Inclination angle (ANGLE) in column name "ANGLE"
    ag = defined air gap
    """
    # setup parameters
    
    md = df.MD
    prev_md = md.shift(periods = 1, fill_value = 0)
    diff_md = md - prev_md
    
    ang = df.ANGLE
    prev_ang = ang.shift(periods = 1, fill_value = 0)
    diff_ang = ang - prev_ang
    
    azi = df.AZIMUTH
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
    
    df['TVD'] = np.cumsum((diff_md / 2) * (np.cos(I1) + np.cos(I2) * rf))

    # remove air gab (ag)

    df.TVD -= ag

    return df

# Function for setting up both TVD and TVDSS depths for well logging and formation top data.

def setup_dep(las, df_las, dev, top, mud, pres, core, drill, field_type, gl, ag):
    """
    This function is going to setting up TVD and TVDSS depths for well logging and formation top data.
    las = las file (.las) of the well data.
    df_las = las input in pandas data frame contains depth column in measured depth (MD) as dataframe index.
    dev = Deviation survey data in pandas data frame which contains:
            1. Measured depth (MD)                      in column name "MD"
            2. Azimuth direction (AZIMUTH)              in column name "AZIMUTH"
            3. Inclination angle (ANGLE)                in column name "ANGLE"
    top = formation top data in pandas data frame which contains:
            1. Formation names                          in column name "Formations"
            2. Top depth boundary of the formation      in column name "Top_TVD"
            3. Bottom depth boundary of the formation   in column name "Bottom_TVD"
    mud = mud weight data in pandas data frame which contains:
            1. Measured depth (MD)                      in column name "DEPTH"
            2. Mud weight (MUD WEIGHT)                  in column name "MUDWEIGHT"
    pres = pressure test in pandas data frame which contains:
            1. Measured depth (MD)                      in column name "DEPTH"
            2. Pressure (PRESSURE)                      in column name "PRESSURE"
    core = core test data in panda data frame which contains:
            1. Measured depth (MD)                      in column name "DEPTH"
            2. Young's modulus (YME)                    in column name "YME"
            3. Poisson's ratio (PR)                     in column name "PR"
            4. Uniaxial compressive strength (UCS)      in column name "UCS"
            5. Friction angle (FANG)                    in column name "FANG"
    drill = drilling test data in panda data frame which contains:
            1. Measured depth (MD)                      in column name "DEPTH"
            2. Test type (TEST)                         in column name "TEST"
            3. Result or value (RESULT)                 in column name "RESULT"
    field_type = defined field type either 'onshore' or 'offshore'
    gl = defined ground level
    ag = defined air gap
    """
    # extend data depth to staring point (measured depth = 0)

    df_las = exten_dep(las, df_las)

    # setup MD column

    df_las.reset_index(inplace = True)
    top['MD'] = top.TOP

    for df in [mud, pres, core, drill]:
        df['MD'] = df.DEPTH

    # merge deviation data and calculate True Vertical Depth (TVD) using function tvd_mini_cuv

    dataframes = [df_las, top, mud, pres, core, drill]

    df_las, top, mud, pres, core, drill = [tvd_cal(merge_dev(df, dev), ag) for df in dataframes]

    # calculate True Vertical Depth Sub-Sea (TVDSS)

    for df in [df_las, top, mud, pres, core, drill]:
        if field_type.lower() == 'onshore':
            df['TVDSS'] = df.TVD - gl
        else:
            df['TVDSS'] = df.TVD

    # manage well logging data

    cols = df_las.columns.tolist()
    cols = cols[:1] + cols[-2:] + cols[1:-2]
    df_las = df_las[cols]
    df_las.set_index('MD', inplace = True)

    # manage formation top data
    
    last_tvd = top.TVD.max()
    last_tvdSS = top.TVDSS.max()

    top.dropna(inplace = True)
    top.reset_index(drop = True, inplace = True)

    top.rename(columns = {'TVD':'TVD_TOP', 'TVDSS':'TVDSS_TOP'}, inplace = True)

    top['TVD_BOTTOM'] = top['TVD_TOP'].shift(periods = -1)
    top['TVDSS_BOTTOM'] = top['TVDSS_TOP'].shift(periods = -1)

    top.fillna(value = {'TVD_BOTTOM': last_tvd, 'TVDSS_BOTTOM': last_tvdSS}, inplace = True)

    # manage the rest of data

    for df in [mud, pres, core, drill]:
        df.dropna(inplace = True)
        df.reset_index(drop = True, inplace = True)
    
    # update LAS file

    las.insert_curve(0, 'MD', df_las.index, unit = 'm', descr = 'Measured Depth', value = '')
    las.insert_curve(1, 'TVD', df_las.TVD, unit = 'm', descr = 'True Vertical Depth', value = '')
    las.insert_curve(2, 'TVDSS', df_las.TVDSS, unit = 'm', descr = 'True Vertical Depth Sub-Sea', value = '')
    las.insert_curve(3, 'AZIMUTH', df_las.AZIMUTH, unit = 'degree', descr = 'Well Deviation in Azimuth', value = '')
    las.insert_curve(4, 'ANGLE', df_las.ANGLE, unit = 'degree', descr = 'Well Deviation in Angle', value = '')
    del las.curves['DEPTH']

    return las, df_las, top, mud, pres, core, drill

# calculate and manipulate TVD and TVDSS

tvd_lases, tvd_tops, tvd_muds, tvd_press, tvd_cores, tvd_drills = [list() for i in range(6)]

for las, df_las, dev, top, mud, pres, core, drill, gl, ag in zip(lases, df_lases, devs, tops, muds, press, cores, drills, ground_levels, air_gaps):

    las, tvd_las, tvd_top, tvd_mud, tvd_pres, tvd_core, tvd_drill = setup_dep(las, df_las, dev, top, mud, pres, core, drill, field_type, gl, ag)
    tvd_files = [tvd_las, tvd_top, tvd_mud, tvd_pres, tvd_core, tvd_drill]

    for store, file in zip([tvd_lases, tvd_tops, tvd_muds, tvd_press, tvd_cores, tvd_drills], tvd_files):
        store.append(file)

    print('Well %s, True vertical depth (TVD) and True vertical depth Sub-Sea (TVDSS) are calculated' %las.well['WELL'].value)

"""

Bad Hole Flag (BHF)

"""

# Function for create Bad Hole flag

def bhf_cal(las, tvd_las, alias):
    """
    This function can compute Bad Hole Flag using confidential interval (ci) and update las file
    las = las file (.las) of the well data
    tvd_las = well logging data in data frame
    * Caliper and bitsize data are required.
    """
    # ci = confidential interval factor (0.00-1.00, default = 0.75)
    
    ci = 0.75 # changable

    for col in tvd_las.columns:
        if col in alias['CAL']:
            caliper = col
        elif col in alias['BS']:
            bitsize = col
    
    diff = tvd_las[caliper] - tvd_las[bitsize]
    interval = st.norm.interval(alpha = ci, loc = round(np.mean(diff), 2), scale = round(np.std(diff), 2))

    # apply confidential interval

    condition1 = (diff < interval[0]) | (diff > interval[1])
    condition2 = (diff >= interval[0]) & (diff <= interval[1])

    tvd_las['BHF'] = np.nan
    tvd_las['BHF'].loc[condition1] = 'BAD'
    tvd_las['BHF'].loc[condition2] = 'GOOD'

    # update LAS file

    las.append_curve('BHF', tvd_las['BHF'], unit = 'unitless', descr = 'Bad Hole Flag', value = '')

    return las, tvd_las, interval

# Create Bad Hole flag for each well

intervals = []

for las, tvd_las in zip(lases, tvd_lases):
    las, tvd_las, interval = bhf_cal(las, tvd_las, alias)
    intervals.append(interval)

# Functions for xtick decoration

def xtick_loc(axis):
    """
    This function is xtick decoration on each column
    axis = sub-axis column
    """
    xticks = axis.xaxis.get_major_ticks()

    xticks[0].label2.set_horizontalalignment('left')   # left align first tick 
    xticks[-1].label2.set_horizontalalignment('right') # right align last tick

# Bad Hole Flag plotting function for checking

def BHF_plot(las, tvd_las, data_range, interval, bhf_name):
    """
    This plotting is for calculated bad hole flag checking.
    las = las file (.las) of the well data
    tvd_las = well logging data in data frame
    data_range = recorded data range
    interval = confidence interval from function create_bhf
    bhf_name = saved figure name
    """
    # get caliper and bitsize data
    
    for col in tvd_las.columns:
        if col in alias['BS']:
            bitsize = col
        elif col in alias['CAL']:
            caliper = col
    
    diff = tvd_las[caliper] - tvd_las[bitsize]
    
    # create figure

    fig, axis = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 12), sharey = True)
    fig.suptitle('Bad Hole Flag of Well %s' %las.well['WELL'].value, fontsize = 14, y = 1.0)

    # general setting for all axis

    start = data_range[0]
    stop = data_range[1]

    condition = (tvd_las.index >= start) & (tvd_las.index <= stop)

    top_depth = tvd_las.loc[condition, 'TVDSS'].dropna().min()
    bottom_depth = tvd_las.loc[condition, 'TVDSS'].dropna().max()

    axis[0].set_ylabel('TVDSS[m]')
    
    for ax in axis:
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False)
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green', alpha = 0.5)
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black', alpha = 0.5) 

    # caliper - bitsize plot

    x = 1
    scale1 = np.arange(int(tvd_las[caliper].min()) - x, tvd_las[caliper].max() + x, x)
    if len(scale1) > 8:
        while len(scale1) > 8:
            x += 1
            scale1 = np.arange(int(tvd_las[caliper].min()) - x, tvd_las[caliper].max() + x, x)

    scalelist1 = [int(scale1[0])]
    for i in range(len(scale1) - 2):
        scalelist1.append('')
    scalelist1.append(int(scale1[-1]))
    
    ax11 = axis[0].twiny()
    ax11.set_xlim(scale1[0], scale1[-1])
    ax11.plot(tvd_las[bitsize], tvd_las.TVDSS, color = 'black', linewidth = '0.5')
    ax11.spines['top'].set_position(('axes', 1.02))
    ax11.set_xlabel('BS[in]',color = 'black')
    ax11.tick_params(axis = 'x', colors = 'black')
    ax11.set_xticks(scale1)
    ax11.set_xticklabels(scalelist1)
    xtick_loc(ax11)
    
    ax12 = axis[0].twiny()
    ax12.set_xlim(scale1[0], scale1[-1])
    ax12.plot(tvd_las[caliper], tvd_las.TVDSS, color = 'grey', linewidth = '0.5')
    ax12.spines['top'].set_position(('axes', 1.08))
    ax12.spines['top'].set_edgecolor('grey')
    ax12.set_xlabel('CAL[in]',color = 'grey')
    ax12.tick_params(axis = 'x', colors = 'grey')
    ax12.set_xticks(scale1)
    ax12.set_xticklabels(scalelist1)
    xtick_loc(ax12)

    ax12.grid(True)

    # caliper - bitsize difference plot

    y = 1
    scale2 = np.arange(int(diff.min()) - y, diff.max() + y, y)
    if len(scale2) > 8:
        while len(scale2) > 8:
            y += 1
            scale2 = np.arange(int(diff.min()) - y, diff.max() + y, y)
    
    scalelist2 = [int(scale2[0])]
    for i in range(len(scale2) - 2):
        scalelist2.append('')
    scalelist2.append(int(scale2[-1]))

    ax21 = axis[1].twiny()
    ax21.set_xlim(scale2[0], scale2[-1])
    ax21.plot(diff, tvd_las.TVDSS, color = 'black', linewidth = '0.5')
    left, right = ax21.get_xlim()
    ax21.set_xlim(left, right)
    ax21.spines['top'].set_position(('axes', 1.02))
    ax21.set_xlabel('DIFF[in]',color = 'black')
    ax21.tick_params(axis = 'x', colors = 'black')
    ax21.set_xticks(scale2)
    ax21.set_xticklabels(scalelist2)
    xtick_loc(ax21)
    
    # interval lines and highlight colors

    for xi in interval:
        ax21.axvline(x = xi, color = 'red')
    
    ax21.axvspan(left, interval[0], facecolor = 'red', alpha = 0.5)
    ax21.axvspan(interval[1], right, facecolor = 'red', alpha = 0.5)
    ax21.axvspan(interval[0], interval[1], facecolor = 'lightgreen', alpha = 0.8)

    ax21.grid(True)

    # Bad Hole Flag plot

    tvd_las['bhf'] = np.nan
    tvd_las.loc[tvd_las.BHF == 'BAD', 'bhf'] = 1
    
    ax31 = axis[2].twiny()
    ax31.set_xlim(0, 1)
    ax31.fill_betweenx(tvd_las.TVDSS, 0, tvd_las.bhf, color = 'red', capstyle = 'butt', linewidth = 1, label = 'BAD')
    ax31.spines['top'].set_position(('axes', 1.02))
    ax31.spines['top'].set_edgecolor('red')
    ax31.set_xlabel('BHF', color = 'red')
    ax31.tick_params(axis = 'x', colors = 'red')
    ax31.set_xticks([0, 1])
    ax31.set_xticklabels(['', ''])
    ax31.legend(loc = 'upper right')
    xtick_loc(ax31)
    
    tvd_las.drop(columns = ['bhf'], inplace = True)
    
    fig.tight_layout()

    # save files

    bhf_folder = 'LQC_BHF'
    bhf_path = os.path.join(sav_path, bhf_folder)

    if not os.path.isdir(bhf_path):
        os.makedirs(bhf_path)

    plt.savefig(os.path.join(bhf_path, bhf_name), dpi = 200, format = 'png', bbox_inches = "tight")
    
    plt.show()

# check created bad hole flag

for las, tvd_las, interval, data_range in zip(lases, tvd_lases, intervals, data_ranges):
    bhf_name = 'LQC_%s_BHF.png' %las.well['WELL'].value
    BHF_plot(las, tvd_las, data_range, interval, bhf_name)

"""

Quality Control 1

"""

# Function for initial plot for first inspection

def initial_inspection(las, tvd_las, based_curves, inspect_name):
    """
    For all curve initial inspection
    las = las file (.las) of the well data
    tvd_las = well logging data in data frame
    based_curves = list of based curve names
    inspect_name = name of saved figure
    """
    # create figure

    fig, axis = plt.subplots(nrows = 1, ncols = len(tvd_las.columns), figsize = (30,20), sharey = True)
    fig.suptitle('All curves of Well %s' %las.well['WELL'].value, fontsize = 30, y = 1.0)
    
    units = [curve.unit for curve in las.curves]
    index_unit = units.pop(0)

    base_cols = based_curves.copy()
    
    for bs in alias['BS']:
        base_cols.append(bs)

    # plot setting for all axis

    top_depth = tvd_las.index.min()
    bottom_depth = tvd_las.index.max()

    axis[0].set_ylabel('MD[%s]' %index_unit, fontsize = 15)

    for ax, col, unit in zip(axis, tvd_las.columns, units):
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black')
        ax.set_xlabel(col + '\n[%s]' %unit, fontsize = 15)

        if (col in alias['RT']) or (col in alias['ILM']) or (col in alias['MSFL']):
            ax.plot(tvd_las[col], tvd_las.index, linewidth = '0.5')
            ax.set_xscale('log')

        elif col == 'BHF':

            tvd_las['bhf'] = np.nan
            tvd_las.loc[tvd_las.BHF == 'BAD', 'bhf'] = 1
            ax.fill_betweenx(tvd_las.index, 0, tvd_las.bhf, color = 'red', capstyle = 'butt', linewidth = 1, label = 'BAD')
            tvd_las.drop(columns = ['bhf'], inplace = True)

        elif col in base_cols:
            ax.plot(tvd_las[col], tvd_las.index, linewidth = '1.0')
        
        else:
            ax.plot(tvd_las[col], tvd_las.index, linewidth = '0.5')

    fig.tight_layout()

    # save files

    inspect_folder = 'LQC_Inspection'
    inspect_path = os.path.join(sav_path, inspect_folder)

    if not os.path.isdir(inspect_path):
        os.makedirs(inspect_path)

    plt.savefig(os.path.join(inspect_path, inspect_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# Plot available curves

for las, tvd_las in zip(lases, tvd_lases):
    inspect_name = 'LQC_%s_Inspected.png' %las.well['WELL'].value
    initial_inspection(las, tvd_las, based_curves, inspect_name)

# Function for removing some zone.

def eliminate_zone(tvd_las, alias, top_zone, bottom_zone, based_curves):    
    """
    This function is for data elimination within the interested zone. 
    All curves within that zone will be removed.
    Except measured depth (MD), azimuth direction (AZIMUTH), angle or deviation (ANGLE), bitsize (BS), caliper (CAL) and bad hole flag (BHF).
    tvd_las = well logging data in pandas data frame.
    alias = curve alias or alterative name of the curve.
    top_zone = Top TVD depth of the zone you want to remove.
    bottom_zone = Bottom TVD depth of the zone you want to remove.
    based_curves = list of based curve names
    """
     # set data columns for elimination

    data_cols = tvd_las.columns.tolist()
    base_cols = based_curves.copy()
    edit_cols = []

    for bs in alias['BS']:
        base_cols.append(bs)

    for cal in alias['CAL']:
        base_cols.append(cal)

    for col in data_cols:
        if col not in base_cols:
            edit_cols.append(col)

    # eliminate the data within the assigned interval

    condition = (tvd_las.index >= float(top_zone)) & (tvd_las.index <= float(bottom_zone))

    tvd_las.loc[condition, edit_cols] = np.nan

    return tvd_las

# Define a top depth of the accepted logging data

for las, tvd_las, data_range in zip(lases, tvd_lases, data_ranges):
    start = data_range[0]
    stop = data_range[1]

    name = las.well['WELL'].value

    print('For well %s, it has logging data between measured depth (MD) %.2f to %.2f.' %(name, start, stop))

    while True:
        accepted_depth = float(input('Please indicate a top depth of the accepted logging data: ').strip())

        if (accepted_depth >= 0) & (accepted_depth < stop):
            tvd_las = eliminate_zone(tvd_las, alias, 0, accepted_depth, based_curves)
            inspect_edited_name = 'LQC_%s_Edited.png' %name
            initial_inspection(las, tvd_las, based_curves, inspect_edited_name)
            break

        else:
            print('Your input is out of range.')
            continue

# Define well and zone of messy data

while True:
    answer = input('Are there any depth dinterval want to be removed or deleted? [Yes/No]: ').strip()

    if answer.lower() == 'yes':

        while True:
            print('There are %d wells.' %len(well_names))
            print('%s, Which one want to be edited?' %', '.join(well_names))
            name = input('Please indicate the well name: ').strip()
            
            if name.lower() in [name.lower() for name in well_names]:
                i = [name.lower() for name in well_names].index(name.lower())

                while True:
                    depth_min = data_ranges[i][0]
                    depth_max = data_ranges[i][1]

                    print('This well has logging data between measured depth (MD) %.2f to %.2f.' %(depth_min, depth_max))
                    zone = input('Please indicate zone or interval using \"-\" as separator (Ex.100-200): ').strip()
                    zone = list(map(str.strip, zone.split('-')))

                    if len(zone) == 2:

                        zone = list(map(float, zone))

                        if (zone[0] >= 0) & (zone[0] < zone[1]) & (zone[1] <= depth_max):

                            while True:
                                name = lases[i].well['WELL'].value
                                print('Your data of well %s from %.2f to %.2f will be eliminated.' %(name, zone[0], zone[1]))
                                
                                confirm = decision_confirm()

                                if confirm.lower() == 'yes':
                                    tvd_lases[i] = eliminate_zone(tvd_lases[i], alias, zone[0], zone[1], based_curves)
                                    inspect_edited_name = 'LQC_%s_Edited.png' %name
                                    initial_inspection(lases[i], tvd_lases[i], based_curves, inspect_edited_name)

                                    print('The data has been eliminated.')
                                    break

                                else:
                                    break    
                            break

                        else:
                            print('Your input is out of range.')

                    else:
                        print('Please indicate again.')
                break

            else:
                print('Your well %s is not found., Please select again!' %name)

    elif answer.lower() == 'no':
        print('Got it!')
        break

    else:
        print('Please comfirm again!')

# Function for setting up the well logging data without the duplicate

def setup_curve(las, well, seen, dupes, mem_curves, based_curves):
    """
    This function will select and eliminate curve data for setting up modeling curve inputs.
    las = las file (.las) of the well data
    well = well logging data in pandas data frame with alias applied
    seen = dictionary contains the name and number of curve
    dupes = list of duplicated curve name
    mem_curves = necessary curve names for modeling
    based_curves = list of based curve names
    """
    # select modeling curves

    for col in well.columns:
        if col.split('_')[0] not in (based_curves + mem_curves):
            well = well.drop([col], axis=1)
            del las.curves[col]

    # manage duplicate curves

    new_col = {}
    choices = []

    for key, value in seen.items():
        if (key in mem_curves) & (key in dupes):
            print('%d curves of %s are found as duplicated curves for well %s' %(value, key, las.well['WELL'].value))

            for num in range(value):
                if num == 0:
                    curve = las.curves[key]
                    print('%s curve is %s' %(curve.mnemonic, curve.descr))
                    choices.append(key)
                else:
                    curve = las.curves[key + '_' + str(num+1)]
                    print('%s curve is %s' %(curve.mnemonic, curve.descr))
                    choices.append(key + '_' + str(num+1))

            while True:
                select = input('Please select a representative curve for %s: ' %key).strip()

                if select.lower() in [choice.lower() for choice in choices]:
                    index = [choice.lower() for choice in choices].index(select.lower())
                    new_col[choices[index]] = key
                    choices.pop(index)
                    break

                else:
                    print('Please type again!, your curve %s is not found.' %select)
                    continue
            
    # eliminate duplicates

    for col in choices:
        well = well.drop([col], axis=1)
        del las.curves[col]
            
    # set curve name
    
    well = well.rename(columns = new_col)

    for key, value in new_col.items():
        las.curves[key].mnemonic = value

    print('All curve data of well %s are setup already' %las.well['WELL'].value)

    return las, well

# Function for standardizing or renaming based on alias

def apply_alias(las, tvd_las, alias):
    """
    This function is going to rename curves based on alias. The duplicates will be named followed by the number.
    las = las file (.las) of the well data
    tvd_las = well logging data in pandas data frame.
    alias = curve alias or alterative name of the curve.
    """
    # get standard curve name from alias

    new_cols = {}
    seen = {}
    dupe = []

    for col in tvd_las.columns:
        for key, values in alias.items():
            
            if col in values:
                new_col = key

                if key not in seen:
                    seen[key] = 1
                
                else:
                    if seen[key] == 1:
                        dupe.append(key)

                    seen[key] += 1
                    new_col = "{}_{}".format(key, seen[key])

                new_cols[col] = new_col

    # apply to tvd_las

    tvd_las.rename(columns = new_cols, inplace = True)

    # apply to las

    for key, value in new_cols.items():
        las.curves[key].mnemonic = value

    print('All curve names of well %s are standardized' %las.well['WELL'].value)
    
    return las, tvd_las, seen, dupe

# Function for eliminating bad data using bad hole flag

def bhf_control(well, affected):
    """
    This function can eliminate the affected data within the bad zone including:
        1. density (RHOB)
        2. neutron porosity (NPHI)
        3. P-sonic (DTC)
        4. S-sonic (DTS)
    well = well logging data in pandas data frame with alias applied.
    affected = list of affected curve names.
    *Bad hole flag must be created using create_bhf function
    """
    # eliminate the data based on bad hole flag

    well.loc[well.BHF == 'BAD', affected] = np.nan
        
    return well

# Function for eliminating low Vp/Vs ratio data

def ratio_control(well):
    """
    This function can elimiate low Vp/Vs ratio (<1.6) led to negative poisson's ratio.
    well = well logging data in pandas data frame with alias applied.
    """
    # eliminate the data based on bad hole flag

    well.loc[well.DTS/well.DTC < 1.6, ['DTC', 'DTS']] = np.nan
    
    return well

# rename and setup curves for modeling

wells = []

print('The system is standardizing and setting up the curves.')

for las, tvd_las in zip(lases, tvd_lases):
    las, well, seen, dupes = apply_alias(las, tvd_las.copy(), alias)
    las, well = setup_curve(las, well, seen, dupes, mem_curves, based_curves)
    well = bhf_control(well, affected) # replace nan for bad hole zone
    well = ratio_control(well)
    wells.append(well)


# ## Data synthetic
# Function for normalizing gamma ray log for data synthetic

def norm_gr(las, well, ref_GR_high, ref_GR_low):
    """
    This function is used for "stretch and squeeze" normalization calculation (Shier, 2004) for well synthetic.
    las = las file (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    ref_GR_high = reference GR max value
    ref_GR_low = reference GR min value
    """
    # normolize gamma ray curve

    NORM = (well.GR - well.GR.quantile(0.05)) / (well.GR.quantile(0.95) - well.GR.quantile(0.05))

    well['GR_NORM'] = ref_GR_low + (ref_GR_high - ref_GR_low) * NORM

    # update las file

    las.append_curve('GR_NORM', well.GR_NORM, unit = 'API', descr = 'Normalized Gamma Ray', value = '')

    return las, well

# Get ref_GR_high and ref_GR_low from all wells mean value

GR_high = []
GR_low = []

for well in wells:
    GR_high.append(well.GR.quantile(0.95))
    GR_low.append(well.GR.quantile(0.05))

ref_GR_high = np.mean(GR_high)
ref_GR_low = np.mean(GR_low)

# Apply normalization function

for las, well in zip(lases, wells):
    las, well = norm_gr(las, well, ref_GR_high, ref_GR_low)

# Function for Checking normalized gamma ray

def hist_norm_gr(wells, well_names, norm_name):
    """
    This function can plot histogram of normalized gamma ray data from all well.
    wells = list of well logging data in pandas data frame with alias applied.
    well_names = list of well name in this field
    norm_name = name of saved figure
    *Normalized Gamma Ray must be calculated using norm_gr function.
    """
    # bins = a number of histogram bar (default = 150)

    bins = 150

    # create figure
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6), sharey = True)
    fig.suptitle('Histogram of Normalized Gamma Ray', fontsize= 15, y = 0.98)
    
    # plot histrogram
    for well, name in zip(wells, well_names):
        ax[0].hist(well.GR, bins = bins, histtype = 'step', label = name, color = well_names[name])
        ax[0].set_xlabel('Gamma Ray [API]')
        ax[0].set_ylabel('Frequency')
        ax[0].set_title('Before')
        ax[0].legend(loc='upper left')
        
        ax[1].hist(well.GR_NORM, bins = bins, histtype = 'step', label = name, color = well_names[name])
        ax[1].set_xlabel('Normalized Gamma Ray [API]')
        ax[1].set_title('After')
        ax[1].legend(loc='upper left')

    fig.tight_layout()

    # save files

    norm_folder = 'LQC_Nomalization'
    norm_path = os.path.join(sav_path, norm_folder)

    if not os.path.isdir(norm_path):
        os.makedirs(norm_path)

    plt.savefig(os.path.join(norm_path, norm_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()
        
# Check normalized gamma ray

norm_name = 'LQC_Norm_GR.png'
hist_norm_gr(wells, well_names, norm_name)

# Function for creating the data set of synthetic modeling

def set_data(wells, tvd_tops, forms, data_ranges, non_affected, affected):
    """
    This function can create the data set for model training and testing of synthetic function.
    wells = list of well logging data in pandas data frame with alias applied.
    tvd_tops = list for formation top data in pandas data frame which contains:
            1. Formation names in column name "Formations"
            2. Top depth boundary of the formation in column name "Top_TVD"
            3. Bottom depth boundary of the formation in column name "Bottom_TVD"
    forms = names of all formations in this field
    data_ranges = range of recorded data
    non_affected = list of non affected curve names for synthetic.
    affected =  list of affected curve names for synthetic.
    """
    # indicate curves for data set

    req_curves = non_affected.copy() + affected.copy()

    # extract the data for each formation from all wells
    
    data_forms = {} # by formation

    for formation in forms:

        data = pd.DataFrame() # build an empty data set

        for well, top in zip(wells, tvd_tops):
            
            if formation in top.FORMATIONS.tolist():

                # Set interval from each well for selected formation

                top_depth = float(top.loc[top.FORMATIONS == formation, 'TOP'])
                bottom_depth = float(top.loc[top.FORMATIONS == formation, 'BOTTOM'])

                # Select data from each well by interval

                condition = (well.index >= top_depth) & (well.index <= bottom_depth)
                
                data = pd.concat([data, well.loc[condition, req_curves]])

                # setup data

                data.reset_index(drop = True, inplace = True)

        # storing

        data_forms[formation] = data

    # none of the above formation, NOTA (assumed formation)

    above_data = pd.DataFrame() # build an empty data set

    for well, top, data_range in zip(wells, tvd_tops, data_ranges):

        start_point = data_range[0]
        above_depth = min([depth for depth in top.TOP])

        # Select data from each well by interval

        condition = (well.index >= start_point) & (well.index <= above_depth)

        above_data = pd.concat([above_data, well.loc[condition, req_curves]])

        # setup data

        above_data.reset_index(drop = True, inplace = True)

    # storing

    data_forms['NOTA'] = above_data

    # build an empty data set and collect the data from each well
    
    data_set = pd.DataFrame()
    
    for well in wells:
        data_set = pd.concat([data_set, well[req_curves]])
    
    # setup data

    data_set.dropna(inplace = True)
    data_set.reset_index(drop = True, inplace = True)
            
    return data_forms, data_set

# Generate the data set for synthetic stage

data_forms, data_set = set_data(wells, tvd_tops, forms, data_ranges, non_affected, affected)

# Function for curve synthetic

def well_syn(las, well, top, data_range, data_forms, data_set, non_affected, affected):
    """
    This function can synthesize bad data within bad zone indicated by bad hole flag.
    This function is going to fix or synthetic the curve for each formation one at the time until all curve are fixed.
    This function based on machine learning techniques (default = multilinear regression and random forest regression)
    
    RT, MSFL and GR_NORM are used as initial curves.
    NPHI, RHOB, DTC and DTS will be synthesized respectively.

    Neutron porosity synthesizing using;
    1.) Deep resistivity (RT)
    2.) Shallow resistivity (MSFL)
    3.) Normalized gamma ray (GR_NORM)
    
    Density synthesizing using;
    1.) Deep resistivity (RT)
    2.) Shallow resistivity (MSFL)
    3.) Normalized gamma ray (GR_NORM)
    4.) Neutron porosity (NPHI)
    
    P-Sonic synthesizing using;
    1.) Deep resistivity (RT)
    2.) Shallow resistivity (MSFL)
    3.) Normalized gamma ray (GR_NORM)
    4.) Neutron porosity (NPHI)
    5.) Density (RHOB)
    
    S-Sonic synthesizing using;
    1.) Deep resistivity (RT)
    2.) Shallow resistivity (MSFL)
    3.) Normalized gamma ray (GR_NORM)
    4.) Neutron porosity (NPHI)
    5.) Density (RHOB)
    6.) P-Sonic (DTC)
    
    las = las file (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    top = formation top data in pandas data frame which contains:
            1. Formation names in column name "Formations"
            2. Top depth boundary of the formation in column name "Top_TVD"
            3. Bottom depth boundary of the formation in column name "Bottom_TVD"
    data_range = range of recorded data
    data_forms = data set for model training and testing in pandas data frame separated by formation
    data_set = data set for model training and testing in pandas data frame
    *data_forms and data_set can be created using set_data.
    non_affected = list of non affected curve names for synthetic.
    affected =  list of affected curve names for synthetic. 
    *The element index of affected curve names will affect synthetic ordering.
    """
    # test_size = size of test data for modeling (0.00 - 1.00, default = 0.3)
    
    test_size = 0.3

    # n_tree = number of decision tree in random forest regression technique (default = 10)

    n_tree = 10

    # Set initial and synthesized data

    initial = non_affected.copy()
    syns = affected.copy()

    cols = initial.copy()

    print('System is synthesizing the data for well %s' %las.well['WELL'].value)

    # add none of the above row for formation top data

    top_data = top.copy()

    start_point = data_range[0]
    above_depth = min([depth for depth in top_data.TOP])

    nota_row = {'FORMATIONS':'NOTA', 'TOP':start_point, 'BOTTOM':above_depth}

    top_data = top_data.append(nota_row, ignore_index = True)

    # Synthesize data one at the time

    synthetic = pd.DataFrame().reindex(well.index)
    method = pd.DataFrame(columns = syns, index = top_data.FORMATIONS)
    score = pd.DataFrame(columns = syns, index = top_data.FORMATIONS)

    for syn in syns:
        
        synthetic[syn] = np.nan

        model_cols = initial.copy() + [syn]

        for form in top_data.FORMATIONS:

            r2 = {}
            
            print('Formation %s is being synthesized for %s.' %(form, syn))

            top_depth = float(top_data.loc[top_data.FORMATIONS == form, 'TOP'])
            bottom_depth = float(top_data.loc[top_data.FORMATIONS == form, 'BOTTOM'])

            # Select model data from each well by interval

            condition = (well.index >= top_depth) & (well.index <= bottom_depth)

            if data_forms[form][model_cols].dropna().empty:
                model_data = data_set
            else:
                model_data = data_forms[form][model_cols].dropna()
            
            # Split the data
            
            input_train = model_data[initial]
            output_train = model_data[syn]

            X_train, X_test, y_train, y_test = train_test_split(input_train, output_train, test_size = test_size, random_state = 0)

            # Setup synthesizing input

            syn_input = well.loc[condition, cols].dropna()
            
            # Multilinear regression modeling

            mlr = LinearRegression()
            mlr.fit(X_train, y_train)

            mlr_r = mlr.score(X_test, y_test)
            r2[mlr_r] = [mlr.predict(syn_input), 'Multilinear Regression', mlr_r]

            # Random forest regression modeling

            rfr = RandomForestRegressor(n_estimators = n_tree)
            rfr.fit(X_train, y_train)

            rfr_r = rfr.score(X_test, y_test)
            r2[rfr_r] = [rfr.predict(syn_input), 'Random Forest Regression', rfr_r]

            # Select the best regression

            syn_output = best_r2(r2)[0]

            method.loc[method.index == form, syn] = best_r2(r2)[1]
            score.loc[score.index == form, syn] = best_r2(r2)[2]

            print('%s is implemented with R-squared value %f' %(best_r2(r2)[1], best_r2(r2)[2]))

            # merge with the other formations

            synthetic[syn + '_' + form] = pd.DataFrame(syn_output, index = syn_input.index)
            synthetic[syn].fillna(synthetic[syn + '_' + form], inplace = True)

        well[syn + '_SYN'] = synthetic[syn]
            
        # Merge curve where synthetic curve replace bad hole sections, and good original curve data remains in place

        well[syn + '_MRG'] = well[syn].fillna(well[syn + '_SYN'])
                    
        # Iterate new syntheric curve with new initial curves
            
        initial.append(syn)
        cols.append(syn + '_MRG')
    
    # Update las file

    las.append_curve('NPHI_SYN', well['NPHI_SYN'], unit = 'V/V', descr = 'Synthetic neutron porosity', value = '')
    las.append_curve('NPHI_MRG', well['NPHI_MRG'], unit = 'V/V', descr = 'Merged neutron porosity', value = '')

    las.append_curve('RHOB_SYN', well['RHOB_SYN'], unit = 'g/c3', descr = 'Synthetic density', value = '')
    las.append_curve('RHOB_MRG', well['RHOB_MRG'], unit = 'g/c3', descr = 'Merged density', value = '')

    las.append_curve('DTC_SYN', well['DTC_SYN'], unit = 'us/ft', descr = 'Synthetic P-sonic', value = '')
    las.append_curve('DTC_MRG', well['DTC_MRG'], unit = 'us/ft', descr = 'Merged P-sonic', value = '')

    las.append_curve('DTS_SYN', well['DTS_SYN'], unit = 'us/ft', descr = 'Synthetic S-sonic', value = '')
    las.append_curve('DTS_MRG', well['DTS_MRG'], unit = 'us/ft', descr = 'Merged S-sonic', value = '')
    
    return las, well, method, score

# Function for selecting the best value

def best_r2(r2):
    """
    This function can select the best element from dictionary by the highest r-squared value.
    r2 = r-squared value with the elements in dictionary form.
    """
    max = list(r2.keys())[0]

    for x in r2: 
        if x > max: 
             max = x 
      
    return r2[max]

"""

Synthesize the data

"""

methods, scores = [], []

for las, well, top, data_range in zip(lases, wells, tvd_tops, data_ranges):
    las, well, method, score = well_syn(las, well, top, data_range, data_forms, data_set, non_affected, affected)
    methods.append(method)
    scores.append(score)

print('Data synthesizing is done.')

# Function for ploting comparision between before and after synthesizing

def syn_compare(las, well, data_range, syn_name):
    """
    This function shows ploting between before and after synthesizing.
    las = las file (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    data_range = recorded data range
    syn_name = name of saved figure
    """
    # Create figure
    
    fig, axis = plt.subplots(nrows = 1, ncols = 7, figsize = (14, 21), sharey = True)
    fig.suptitle('Data Synthesizing of Well %s' %las.well['WELL'].value, fontsize= 20, y = 1.01)
    
    #General setting for all axis

    axis[0].set_ylabel('TVDSS[m]')

    start = data_range[0]
    stop = data_range[1]

    condition = (well.index >= start) & (well.index <= stop)

    top_depth = well.loc[condition, 'TVDSS'].dropna().min()
    bottom_depth = well.loc[condition, 'TVDSS'].dropna().max()
    
    for ax in axis:
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False) 
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='blue')
    
    # Gamma ray plot
    
    ax01 = axis[0].twiny()
    ax01.plot(well.GR_NORM, well.TVDSS, color = 'green', linewidth = '0.5')
    ax01.spines['top'].set_position(('axes', 1.02))
    ax01.spines['top'].set_edgecolor('green') 
    ax01.set_xlim(0, 150)
    ax01.set_xlabel('GR_NORM[API]', color = 'green')    
    ax01.tick_params(axis = 'x', colors = 'green')
    ax01.set_xticks(np.arange(0, 151, 30))
    ax01.set_xticklabels(['0', '', '', '', '','150'])
    xtick_loc(ax01)
    
    ax01.grid(True)
    
    # Resisitivity plots
    
    ax11 = axis[1].twiny()
    ax11.set_xscale('log')
    ax11.plot(well.RT, well.TVDSS, color = 'red', linewidth = '0.5')
    ax11.spines['top'].set_position(('axes', 1.02))
    ax11.spines['top'].set_edgecolor('red')
    ax11.set_xlim(0.2, 2000)
    ax11.set_xlabel('RT[ohm-m]', color = 'red')    
    ax11.tick_params(axis = 'x', colors = 'red')
    ax11.set_xticks([0.2, 2, 20, 200, 2000])
    ax11.set_xticklabels(['0.2', '', '', '', '2000'])
    xtick_loc(ax11)
    
    ax11.grid(True)

    ax12 = axis[1].twiny()
    ax12.set_xscale('log')
    ax12.plot(well.MSFL, well.TVDSS, color = 'black', linewidth = '0.5')
    ax12.spines['top'].set_position(('axes', 1.05))
    ax12.spines['top'].set_edgecolor('black')
    ax12.set_xlim(0.2, 2000)
    ax12.set_xlabel('MSFL[ohm-m]', color = 'black')    
    ax12.tick_params(axis = 'x', colors = 'black')
    ax12.set_xticks([0.2, 2, 20, 200, 2000])
    ax12.set_xticklabels(['0.2', '', '', '', '2000'])
    xtick_loc(ax12)
    
    # Neutron porosity plot

    ax22 = axis[2].twiny()
    ax22.set_xlim(-0.15, 0.45)
    ax22.plot(well.NPHI_SYN, well.TVDSS, color = 'red', linewidth = '0.5')
    ax22.spines['top'].set_position(('axes', 1.05))
    ax22.spines['top'].set_edgecolor('red')
    ax22.set_xlabel('NPHI_SYN[V/V]', color = 'red')
    ax22.tick_params(axis = 'x', colors = 'red')
    ax22.set_xticks(np.arange(-0.15, 0.46, 0.12))
    ax22.set_xticklabels(['-0.15', '', '', '', '', '0.45'])
    xtick_loc(ax22)
    
    ax21 = axis[2].twiny()
    ax21.set_xlim(-0.15, 0.45)
    ax21.plot(well.NPHI, well.TVDSS, color = 'blue', linewidth = '0.8')
    ax21.spines['top'].set_position(('axes', 1.02))
    ax21.spines['top'].set_edgecolor('blue')
    ax21.set_xlabel('NPHI[V/V]', color = 'blue')
    ax21.tick_params(axis = 'x', colors = 'blue')
    ax21.set_xticks(np.arange(-0.15, 0.46, 0.12))
    ax21.set_xticklabels(['-0.15', '', '', '', '', '0.45'])
    xtick_loc(ax21)

    ax23 = axis[2].twiny()
    ax23.set_xlim(-0.15, 0.45)
    ax23.plot(well.NPHI_MRG, well.TVDSS, color = 'black', linewidth = '0.5', linestyle = '--')
    ax23.spines['top'].set_position(('axes', 1.08))
    ax23.spines['top'].set_edgecolor('black')
    ax23.set_xlabel('NPHI_MRG[V/V]', color = 'black')
    ax23.tick_params(axis = 'x', colors = 'black')
    ax23.set_xticks(np.arange(-0.15, 0.46, 0.12))
    ax23.set_xticklabels(['-0.15', '', '', '', '', '0.45'])
    xtick_loc(ax23)

    ax23.grid(True)

    ax23.text(0.5, 0.995, 'Correlation : %.2f' %well.NPHI.corr(well.NPHI_SYN), 
                    ha = 'center', va = 'top', transform = ax23.transAxes, color = 'red')
    
    # Density plot
    
    ax32 = axis[3].twiny()
    ax32.set_xlim(1.95, 2.95)
    ax32.plot(well.RHOB_SYN, well.TVDSS, color = 'red', linewidth = '0.5')
    ax32.spines['top'].set_position(('axes', 1.05))
    ax32.spines['top'].set_edgecolor('red')
    ax32.set_xlabel('RHOB_SYN[g/c3]', color = 'red')
    ax32.tick_params(axis = 'x', colors = 'red')
    ax32.set_xticks(np.arange(1.95, 2.96, 0.2))
    ax32.set_xticklabels(['1.95', '', '', '', '', '2.95'])
    xtick_loc(ax32)

    ax31 = axis[3].twiny()
    ax31.set_xlim(1.95, 2.95)
    ax31.plot(well.RHOB, well.TVDSS, color = 'blue', linewidth = '0.8')
    ax31.spines['top'].set_position(('axes', 1.02))
    ax31.spines['top'].set_edgecolor('blue')
    ax31.set_xlabel('RHOB[g/c3]', color = 'blue')
    ax31.tick_params(axis = 'x', colors = 'blue')
    ax31.set_xticks(np.arange(1.95, 2.96, 0.2))
    ax31.set_xticklabels(['1.95', '', '', '', '', '2.95'])
    xtick_loc(ax31)

    ax33 = axis[3].twiny()
    ax33.set_xlim(1.95, 2.95)
    ax33.plot(well.RHOB_MRG, well.TVDSS, color = 'black', linewidth = '0.5', linestyle = '--')
    ax33.spines['top'].set_position(('axes', 1.08))
    ax33.spines['top'].set_edgecolor('black')
    ax33.set_xlabel('RHOB_MRG[g/c3]', color = 'black')
    ax33.tick_params(axis = 'x', colors = 'black')
    ax33.set_xticks(np.arange(1.95, 2.96, 0.2))
    ax33.set_xticklabels(['1.95', '', '', '', '', '2.95'])
    xtick_loc(ax33)

    ax33.grid(True)

    ax33.text(0.5, 0.995, 'Correlation : %.2f' %well.RHOB.corr(well.RHOB_SYN), 
                    ha = 'center', va = 'top', transform = ax33.transAxes, color = 'red')

    # P-sonic plot

    ax42 = axis[4].twiny()
    ax42.set_xlim(40, 140)
    ax42.plot(well.DTC_SYN, well.TVDSS, color = 'red', linewidth = '0.5')
    ax42.spines['top'].set_position(('axes', 1.05))
    ax42.spines['top'].set_edgecolor('red')
    ax42.set_xlabel('DTC_SYN[us/ft]', color = 'red')
    ax42.tick_params(axis = 'x', colors = 'red')
    ax42.set_xticks(np.arange(40 , 141, 20))
    ax42.set_xticklabels(['40', '', '', '', '', '140'])
    xtick_loc(ax42)

    ax41 = axis[4].twiny()
    ax41.set_xlim(40, 140)
    ax41.plot(well.DTC, well.TVDSS, color = 'blue', linewidth = '0.8')
    ax41.spines['top'].set_position(('axes', 1.02))
    ax41.spines['top'].set_edgecolor('blue')
    ax41.set_xlabel('DTC[us/ft]', color = 'blue')
    ax41.tick_params(axis = 'x', colors = 'blue')
    ax41.set_xticks(np.arange(40 , 141, 20))
    ax41.set_xticklabels(['40', '', '', '', '', '140'])
    xtick_loc(ax41)

    ax43 = axis[4].twiny()
    ax43.set_xlim(40, 140)
    ax43.plot(well.DTC_MRG, well.TVDSS, color = 'black', linewidth = '0.5', linestyle = '--')
    ax43.spines['top'].set_position(('axes', 1.08))
    ax43.spines['top'].set_edgecolor('black')
    ax43.set_xlabel('DTC_MRG[us/ft]', color = 'black')
    ax43.tick_params(axis = 'x', colors = 'black')
    ax43.set_xticks(np.arange(40 , 141, 20))
    ax43.set_xticklabels(['40', '', '', '', '', '140'])
    xtick_loc(ax43)

    ax43.grid(True)

    ax43.text(0.5, 0.995, 'Correlation : %.2f' %well.DTC.corr(well.DTC_SYN), 
                    ha = 'center', va = 'top', transform = ax43.transAxes, color = 'red')
    
    # S-sonic plot
    
    ax52 = axis[5].twiny()
    ax52.set_xlim(40, 340)
    ax52.plot(well.DTS_SYN, well.TVDSS, color = 'red', linewidth = '0.5')
    ax52.spines['top'].set_position(('axes', 1.05))
    ax52.spines['top'].set_edgecolor('red')
    ax52.set_xlabel('DTS_SYN[us/ft]', color = 'red')
    ax52.tick_params(axis = 'x', colors = 'red')
    ax52.set_xticks(np.arange(40 , 341, 60))
    ax52.set_xticklabels(['40', '', '', '', '', '340'])
    xtick_loc(ax52)
    
    ax51 = axis[5].twiny()
    ax51.set_xlim(40, 340)
    ax51.plot(well.DTS, well.TVDSS, color = 'blue', linewidth = '0.8')
    ax51.spines['top'].set_position(('axes', 1.02))
    ax51.spines['top'].set_edgecolor('blue')
    ax51.set_xlabel('DTS[us/ft]', color = 'blue')
    ax51.tick_params(axis = 'x', colors = 'blue')
    ax51.set_xticks(np.arange(40 , 341, 60))
    ax51.set_xticklabels(['40', '', '', '', '', '340'])
    xtick_loc(ax51)

    ax53 = axis[5].twiny()
    ax53.set_xlim(40, 340)
    ax53.plot(well.DTS_MRG, well.TVDSS, color = 'black', linewidth = '0.5', linestyle = '--')
    ax53.spines['top'].set_position(('axes', 1.08))
    ax53.spines['top'].set_edgecolor('black')
    ax53.set_xlabel('DTS_MRG[us/ft]', color = 'black')
    ax53.tick_params(axis = 'x', colors = 'black')
    ax53.set_xticks(np.arange(40 , 341, 60))
    ax53.set_xticklabels(['40', '', '', '', '', '340'])
    xtick_loc(ax53)

    ax53.grid(True)

    ax53.text(0.5, 0.995, 'Correlation : %.2f' %well.DTS.corr(well.DTS_SYN), 
                    ha = 'center', va = 'top', transform = ax53.transAxes, color = 'red')

    # Bad Hole Flag plot

    well['bhf'] = np.nan
    well.loc[well.BHF == 'BAD', 'bhf'] = 1
    
    ax61 = axis[6].twiny()
    ax61.set_xlim(0, 1)
    ax61.fill_betweenx(well.TVDSS, 0, well.bhf, color = 'red', capstyle = 'butt', linewidth = 1, label = 'BAD')
    ax61.spines['top'].set_position(('axes', 1.02))
    ax61.spines['top'].set_edgecolor('red')
    ax61.set_xlabel('BHF', color = 'red')
    ax61.tick_params(axis = 'x', colors = 'red')
    ax61.set_xticks([0, 1])
    ax61.set_xticklabels(['', ''])
    ax61.legend(loc = 'upper right')
    
    well.drop(columns = ['bhf'], inplace = True)
        
    fig.tight_layout()

    # Save files

    synthetic_folder = 'LQC_Synthetic'
    synthetic_path = os.path.join(sav_path, synthetic_folder)

    if not os.path.isdir(synthetic_path):
        os.makedirs(synthetic_path)

    plt.savefig(os.path.join(synthetic_path, syn_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()
        
# Check the synthetic

for las, well, data_range in zip(lases, wells, data_ranges):
    syn_name = 'LQC_%s_Syn.png' %las.well['WELL'].value
    syn_compare(las, well, data_range, syn_name)

"""

Quality Control 2 by Boxplot

"""

# Function for check the quality of the input data for interested zone

def qc_data(wells, tvd_tops, form, well_names, qc_name):
    """
    This function will create the boxplot for checking the input data.
    wells = completed well data in pandas dataframe (Merged data with the synthetics)
    tvd_tops = list for formation top data in pandas data frame which contains:
            1. Formation names in column name "Formations"
            2. Top depth boundary of the formation in column name "Top_TVD"
            3. Bottom depth boundary of the formation in column name "Bottom_TVD"
    formation = input the name of the formation where the data can be compared
    well_names = list of well names with color code in dictionary format
    qc_name = name of saved figure
    """
    # Set data for specific interval
    
    GR_plot, RHOB_plot, NPHI_plot, DTC_plot, DTS_plot, well_labels = [list() for i in range(6)]
    
    data_plots = [GR_plot, RHOB_plot, NPHI_plot, DTC_plot, DTS_plot]
    
    selected_cols = ['GR_NORM', 'RHOB_MRG', 'NPHI_MRG', 'DTC_MRG', 'DTS_MRG']

    curve_labels = ['GR', 'RHOB', 'NPHI', 'DTC', 'DTS']

    unit_labels = ['API', 'g/c3', 'V/V', 'us/ft', 'us/ft']
    
    for well, top, name in zip(wells, tvd_tops, well_names):
        
        # Check available data for selected formation
        
        if form in list(top.FORMATIONS):
            
            # Set interval from each well for selected formation

            top_depth = float(top.loc[top.FORMATIONS == form, 'TOP'])
            bottom_depth = float(top.loc[top.FORMATIONS == form, 'BOTTOM'])

            well_labels.append(name)
            
            # Select data from each well by interval

            condition = (well.index >= top_depth) & (well.index <= bottom_depth)

            for store, col in zip(data_plots, selected_cols):
                store.append(well.loc[condition, col].dropna())

    # Setup well colors for plotting

    well_colors = []

    for name in well_labels:
        well_colors.append(well_names[name])

    well_colo = [item for sublist in [(c, c) for c in well_colors] for item in sublist]
    
    # Create figure
    
    fig, axis = plt.subplots(nrows = 1, ncols = 5, figsize = (12.5, 2.5), sharey = False)
    fig.suptitle('Box Plot Quality Control of formation ' + '\'' + form + '\'', fontsize = 20, y = 1.0)
    
    # Plot setting for all axis
        
    for data, label, unit, ax in zip(data_plots, curve_labels, unit_labels, axis):
        boxes = ax.boxplot(data, labels = well_labels, meanline = True, notch = True, showfliers = False, patch_artist = True)
        ax.set_ylabel('[%s]' %unit)
        
        # set decoration
        for patch, color in zip(boxes['boxes'], well_colors): 
            patch.set_facecolor(color) 
        
        for box_wk, box_cap, color in zip(boxes['whiskers'], boxes['caps'], well_colo):
            box_wk.set(color = color, linewidth = 1.5)
            box_cap.set(color = color, linewidth = 3)
        
        for median in boxes['medians']:
            median.set(color = 'black', linewidth = 3) 
            
        ax.set_title(label)
    
    fig.tight_layout()

    # Save files

    qc_folder = 'LQC_Boxplot'
    qc_path = os.path.join(sav_path, qc_folder)

    if not os.path.isdir(qc_path):
        os.makedirs(qc_path)

    plt.savefig(os.path.join(qc_path, qc_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# Plot boxplot for quality comtrol each selected formation

for form in selected_forms:
    qc_name = 'LQC_%s.png' %form
    qc_data(wells, tvd_tops, form, well_names, qc_name)

"""

Data visualization 1

"""

# Function for well data visualization in composite log plots

def composite_logs(las, well, tvd_top, data_range, all_forms, logs_name):
    """
    Plot the curves in composite logs
    las = las file (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    tvd_top = formation top data in pandas data frame.
    data_range = recorded data range
    all_forms = list of all formation names with color code in dictionary format
    logs_name = name of saved figure.
    """
    # Create figure and subplots
    
    fig, axis = plt.subplots(nrows = 1, ncols = 6, figsize = (15,20), sharey = True)
    fig.suptitle('Composite Log of Well %s' %las.well['WELL'].value, fontsize = 20, y = 1.0)
    
    # General setting for all axis

    axis[0].set_ylabel('TVDSS[m]')

    start = data_range[0]
    stop = data_range[1]

    condition = (well.index >= start) & (well.index <= stop)

    top_depth = well.loc[condition, 'TVDSS'].dropna().min()
    bottom_depth = well.loc[condition, 'TVDSS'].dropna().max()
    
    for ax in axis:
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False) 
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')
        
    # Plot formations

    for ax in axis[:-1]:
        for top, bottom, form in zip(tvd_top.TVDSS_TOP, tvd_top.TVDSS_BOTTOM, tvd_top.FORMATIONS):
            if (top >= top_depth) & (top <= bottom_depth):
                ax.axhline(y = top, linewidth = 1.5, color = all_forms[form], alpha = 0.8)

                if (bottom <= bottom_depth):
                    middle = top + (bottom - top)/2
                    ax.axhspan(top, bottom, facecolor = all_forms[form], alpha = 0.2)
                
                else:
                    middle = top + (bottom_depth - top)/2
                    ax.axhspan(top, bottom_depth, facecolor = all_forms[form], alpha = 0.2)
                    
                ax.text(0.01, middle , form, ha = 'left', va = 'center', color = all_forms[form], 
                        path_effects = [pe.withStroke(linewidth = 3, foreground = "white")], fontsize = 10, weight = 'bold')
    
    # Azimuth and angle plots
    
    ax11 = axis[0].twiny()
    ax11.plot(well.AZIMUTH, well.TVDSS, color = 'blue', linewidth = '0.8')
    ax11.spines['top'].set_position(('axes', 1.02))
    ax11.spines['top'].set_edgecolor('blue')
    ax11.set_xlim(0, 360)
    ax11.set_xlabel('AZIMUTH[degree]', color = 'blue')    
    ax11.tick_params(axis = 'x', colors = 'blue')
    ax11.set_xticks(np.arange(0, 361, 90))
    ax11.set_xticklabels(['0', '', '180', '', '360'])
    xtick_loc(ax11)
    
    ax11.grid(True)

    ax12 = axis[0].twiny()
    ax12.plot(well.ANGLE, well.TVDSS, color = 'red', linewidth = '0.8')
    ax12.spines['top'].set_position(('axes', 1.05))
    ax12.spines['top'].set_edgecolor('red')   
    ax12.set_xlim(0, 90)
    ax12.set_xlabel('ANGLE[degree]', color = 'red')    
    ax12.tick_params(axis = 'x', colors = 'red')
    ax12.set_xticks(np.arange(0, 91, 45))
    ax12.set_xticklabels(['0', '45', '90'])
    xtick_loc(ax12)
 
    # Gamma ray plot
    
    ax21 = axis[1].twiny()
    ax21.plot(well.GR_NORM, well.TVDSS, color = 'green', linewidth = '0.5')
    ax21.spines['top'].set_position(('axes', 1.02))
    ax21.spines['top'].set_edgecolor('green') 
    ax21.set_xlim(0, 150)
    ax21.set_xlabel('GR[API]', color = 'green')    
    ax21.tick_params(axis = 'x', colors = 'green')
    ax21.set_xticks(np.arange(0, 151, 30))
    ax21.set_xticklabels(['0', '', '', '', '','150'])
    xtick_loc(ax21)
    
    ax21.grid(True)
    
    # Resisitivity plots
    
    ax31 = axis[2].twiny()
    ax31.set_xscale('log')
    ax31.plot(well.RT, well.TVDSS, color = 'red', linewidth = '0.5')
    ax31.spines['top'].set_position(('axes', 1.02))
    ax31.spines['top'].set_edgecolor('red')
    ax31.set_xlim(0.2, 2000)
    ax31.set_xlabel('RT[ohm-m]', color = 'red')
    ax31.tick_params(axis = 'x', colors = 'red')
    ax31.set_xticks([0.2, 2, 20, 200, 2000])
    ax31.set_xticklabels(['0.2', '', '', '', '2000'])
    xtick_loc(ax31)
    
    ax31.grid(True)

    ax32 = axis[2].twiny()
    ax32.set_xscale('log')
    ax32.plot(well.MSFL, well.TVDSS, color = 'black', linewidth = '0.5')
    ax32.spines['top'].set_position(('axes', 1.05))
    ax32.spines['top'].set_edgecolor('black')
    ax32.set_xlim(0.2, 2000)
    ax32.set_xlabel('MSFL[ohm-m]', color = 'black')
    ax32.tick_params(axis = 'x', colors = 'black')
    ax32.set_xticks([0.2, 2, 20, 200, 2000])
    ax32.set_xticklabels(['0.2', '', '', '', '2000'])
    xtick_loc(ax32)
    
    # Density and neutron porosity plots
    
    ax41 = axis[3].twiny()
    ax41.plot(well.RHOB_MRG, well.TVDSS, color = 'red', linewidth = '0.5')
    ax41.spines['top'].set_position(('axes', 1.02))
    ax41.spines['top'].set_edgecolor('red')
    ax41.set_xlim(1.95, 2.95)
    ax41.set_xlabel('RHOB_MRG[g/c3]', color = 'red')    
    ax41.tick_params(axis = 'x', colors = 'red')
    ax41.set_xticks(np.arange(1.95, 2.96, 0.2))
    ax41.set_xticklabels(['1.95', '', '', '', '', '2.95'])
    xtick_loc(ax41)
    
    ax41.grid(True)

    ax42 = axis[3].twiny()
    ax42.plot(well.NPHI_MRG, well.TVDSS, color = 'blue', linewidth = '0.5')
    ax42.spines['top'].set_position(('axes', 1.05))
    ax42.spines['top'].set_edgecolor('blue')   
    ax42.set_xlim(0.45, -0.15)
    ax42.set_xlabel('NPHI_MRG[V/V]', color = 'blue')    
    ax42.tick_params(axis = 'x', colors = 'blue')
    ax42.set_xticks(np.arange(0.45, -0.16, -0.12))
    ax42.set_xticklabels(['0.45', '', '', '', '', '-0.15'])
    xtick_loc(ax42)
    
    # P_Sonic and S_Sonic plots
    
    ax51 = axis[4].twiny()
    ax51.plot(well.DTC_MRG, well.TVDSS, color = 'blue', linewidth = '0.5')
    ax51.spines['top'].set_position(('axes', 1.02))
    ax51.spines['top'].set_edgecolor('blue')
    ax51.set_xlim(140, 40)
    ax51.set_xlabel('DTC_MRG[us/ft]', color = 'blue')    
    ax51.tick_params(axis = 'x', colors = 'blue')
    ax51.set_xticks(np.arange(140, 39, -20))
    ax51.set_xticklabels(['140', '', '', '', '', '40'])
    xtick_loc(ax51)

    ax51.grid(True)

    ax52 = axis[4].twiny()
    ax52.plot(well.DTS_MRG, well.TVDSS, color = 'red', linewidth = '0.5')
    ax52.spines['top'].set_position(('axes', 1.05))
    ax52.spines['top'].set_edgecolor('red') 
    ax52.set_xlim(340, 40)
    ax52.set_xlabel('DTS_MRG[us/ft]', color = 'red')    
    ax52.tick_params(axis = 'x', colors = 'red')
    ax52.set_xticks(np.arange(340, 39, -60))
    ax52.set_xticklabels(['340', '', '', '', '', '40'])
    xtick_loc(ax52)

    # Bad hole flag plots

    well['bhf'] = np.nan
    well.loc[well.BHF == 'BAD', 'bhf'] = 1
    
    ax61 = axis[5].twiny()
    ax61.set_xlim(0, 1)
    ax61.fill_betweenx(well.TVDSS, 0, well.bhf, color = 'red', capstyle = 'butt', linewidth = 1, label = 'BAD')
    ax61.spines['top'].set_position(('axes', 1.02))
    ax61.spines['top'].set_edgecolor('red')
    ax61.set_xlabel('BHF', color = 'red')    
    ax61.tick_params(axis = 'x', colors = 'red')
    ax61.set_xticks([0, 1])
    ax61.set_xticklabels(['', ''])
    
    well.drop(columns = ['bhf'], inplace = True)
        
    fig.tight_layout()

    # Save files

    logs_folder = 'LQC_Composite_Logs'
    logs_path = os.path.join(sav_path, logs_folder)

    if not os.path.isdir(logs_path):
        os.makedirs(logs_path)

    plt.savefig(os.path.join(logs_path, logs_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# Plot available curves

for las, well, tvd_top, data_range in zip(lases, wells, tvd_tops, data_ranges):
    logs_name = 'LQC_%s_Logs.png' %las.well['WELL'].value
    composite_logs(las, well, tvd_top, data_range, all_forms, logs_name)

# Function for export the data to new las file (.las) and comma-separated values file (.csv)

def export_well(las, well, las_name, csv_name):
    """
    This function can export data to new las file (.las) and comma-separated values file (.csv)
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    """
    # create a new empty las file

    las_file = lasio.LASFile()

    # export well data (data frame format) to empty las file

    las_file.set_data(well)

    # update curve unit and description
    
    for curve_1, curve_2 in zip(las_file.curves, las.curves):
        if curve_1.mnemonic == curve_2.mnemonic:
            curve_1.unit = curve_2.unit
            curve_1.descr = curve_2.descr

    # update las file header

    las_file.well = las.well

    # update special note for las file
    las_file.other = 'This file was written by python code in %s' %datetime.today().strftime('%m-%d-%Y %H:%M:%S')

    # Save las files

    LQC_las_folder = 'LQC_LAS_files'
    LQC_las_path = os.path.join(sav_path, LQC_las_folder)

    if not os.path.isdir(LQC_las_path):
        os.makedirs(LQC_las_path)
    
    las_file.write(os.path.join(LQC_las_path, las_name), version = 2.0)

    # setup header for csv file

    headers = []

    for curve in las.curves:
        header = '%s[%s]' %(curve.mnemonic, curve.unit)
        headers.append(header)

    index = headers.pop(0)

    # Save csv files

    LQC_csv_folder = 'LQC_CSV_files'
    LQC_csv_path = os.path.join(sav_path, LQC_csv_folder)

    if not os.path.isdir(LQC_csv_path):
        os.makedirs(LQC_csv_path)

    well.rename_axis(index).to_csv(os.path.join(LQC_csv_path, csv_name), header = headers)

# Export las and csv files

for las, well in zip(lases, wells):
    las_name = 'LQC_%s_Input.las' %las.well['WELL'].value
    csv_name = 'LQC_%s_Input.csv' %las.well['WELL'].value
    export_well(las, well, las_name, csv_name)
    print('Well data of %s are exported to las and csv files.' %las.well['WELL'].value)

"""

3.Mechanical Stratigraphy

"""

# Function for calculating volume of clay

def vcl_cal(well):
    """
    This function is for calculating the volume of clay (VCL) using two methods;
    - Neutron-Density based on N-D crossplot for shaly sand formation excluding gas bearing formation (low density with low neutron porosity ).
    - Linear Gamma Ray for gas bearing formation (overestimation of Vcl is possible for high uranium shale)
    well = well logging data in pandas data frame with alias applied.
    """
    # output store

    vcl_df = pd.DataFrame().reindex(well.index)

    # input parameters

    RHOB = well.RHOB_MRG.dropna()
    NPHI = well.NPHI_MRG.dropna()

    # matrix and fluid parameters

    RHOBm, NPHIm = 2.65, 0
    RHOBf, NPHIf = 1.0, 1.0

    # shale parameters

    RHOBsh = RHOB.quantile(0.55)
    NPHIsh = NPHI.quantile(0.55)

    # volume of clay  from Neutron-Density crossplot equation (Bhuyan and Passey, 1994)

    term1 = (RHOBm-RHOBf)*(NPHI-NPHIf) - (RHOB-RHOBf)*(NPHIm-NPHIf)
    term2 = (RHOBm-RHOBf)*(NPHIsh-NPHIf) - (RHOBsh-RHOBf)*(NPHIm-NPHIf)
    vcl_df['VCLnd'] = term1/term2

    # volume of clay from GR

    GR = well.GR_NORM.dropna()

    vcl_df['VCLgr'] = (GR - GR.quantile(0.10)) / (GR.quantile(0.80) - GR.quantile(0.10))

    # iliminate volume of clay < 0 and volume of clay > 1

    vcl_df.loc[vcl_df.VCLnd < 0, 'VCLnd'] = np.nan

    # limit exceeding volume of clay (VCL > 1) to maximum value (VCL = 1)

    vcl_df.VCLnd.clip(0, 1, inplace = True)

    return vcl_df

# Vcl correlation plot

def vcl_plot(vcl_df, coef, intercept, vclcor_name):
    """
    This function is for plotting Vcl correlation plot
    vcl_df = well logging data in pandas data frame with calculated Vcl.
    coef = linear coefficient
    intercept = linear intercept
    """
    # Create figure

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5,5))
    fig.suptitle('Volume of clay correlation \n Slope = %f, Intercept = %f' %(coef, intercept), fontsize = 10, y = 0.97)

    # inputs

    VCLnd = vcl_df[['VCLnd', 'VCLgr']].dropna().VCLnd
    VCLgr = vcl_df[['VCLnd', 'VCLgr']].dropna().VCLgr

    # correlation plot

    ax.scatter(VCLgr, VCLnd, alpha = 0.5, marker = '.')

    start_line = (coef * vcl_df.VCLgr.min()) + intercept
    stop_line = (coef * vcl_df.VCLgr.max()) + intercept

    x_line = [vcl_df.VCLgr.min(), vcl_df.VCLgr.max()]
    y_line = [start_line, stop_line]

    line = mlines.Line2D(x_line, y_line, color = 'black')
    ax.add_line(line)

    ax.set_xlabel('VCL from gamma ray')
    ax.set_ylabel('VCL from neutron porosity and density')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    fig.tight_layout()

    # Save files

    ndplot_folder = 'Mechanical Stratigraphy'
    ndplot_path = os.path.join(sav_path, ndplot_folder)

    if not os.path.isdir(ndplot_path):
        os.makedirs(ndplot_path)

    plt.savefig(os.path.join(ndplot_path, vclcor_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# Function for Vcl correlation

def vcl_corr(las, well, vcl_df):
    """
    This function is for vcl correlation of VCLnd and VCLgr
    las = las files (.las) of the well data.
    well = well logging data in pandas data frame with alias applied.
    vcl_df = well logging data in pandas data frame with calculated Vcl.
    """
    # prepare data

    VCLnd = vcl_df[['VCLnd', 'VCLgr']].dropna().VCLnd.values.reshape(-1,1) # reshape for model prediction
    VCLgr = vcl_df[['VCLnd', 'VCLgr']].dropna().VCLgr.values.reshape(-1,1) # reshape for model prediction

    # curve fitting

    lr = LinearRegression()
    lr.fit(VCLgr, VCLnd)

    coef = lr.coef_[0][0]
    intercept = lr.intercept_[0]

    vcl_df['coef'] = coef
    vcl_df['intercept'] = intercept

    VCLcor = (vcl_df.VCLgr * vcl_df.coef) + vcl_df.intercept

    # plot correlation model

    vclcor_name = '%s_VCLcor.png' %las.well['WELL'].value
    vcl_plot(vcl_df, coef, intercept, vclcor_name)

    # get Vcl

    well['VCL'] = vcl_df.VCLnd
    well['VCL'].fillna(VCLcor, inplace = True)

    # update LAS file

    las.append_curve('VCL', well['VCL'], unit = 'V/V', descr = 'Volume of clay', value = '')

    return las, well

# Function for defining the lithology (sand/shale) using normalized gamma ray log

def litho_cal(las, well):
    """
    This function is for defining sand-shale lithology based on VCL
    las = las files (.las) of the well data
    well = well logging data in pandas data frame in TVD depth with alias applied.
    """
    # define sand-shale cut off from volume of clay (VCL)

    cutoff = 0.4

    well['LITHO'] = pd.cut(well.VCL, bins = [0, cutoff, 1], labels = ['SAND', 'SHALE'])

    # update LAS file

    las.append_curve('LITHO', well['LITHO'], unit = 'unitless', descr = 'Lithology', value = '')

    return las, well

# Function for calculating effective porosity 

def phie_cal(las, well):
    """
    This function is for calculating the effective porosity (PHIE) using neutron-density conbination;
    las = las files (.las) of the well data.
    well = well logging data in pandas data frame with alias applied.
    """
    # input parameters

    RHOB = well.RHOB_MRG.dropna()
    NPHI = well.NPHI_MRG.dropna()
    VCL = well.VCL.dropna()

    # matrix and water parameters

    RHOBm, RHOBw = 2.65, 1.0

    # shale parameters

    RHOBsh = RHOB.quantile(0.55)
    NPHIsh = NPHI.quantile(0.55)

    # density porosity computation with shale correction

    DPHI = (RHOBm - RHOB) / (RHOBm - RHOBw)
    DPHIsh = (RHOBm - RHOBsh) / (RHOBm - RHOBw)
    DPHIshcor = DPHI - (VCL * DPHIsh)

    # neutron porosity with shale correction

    NPHIshcor = NPHI - (VCL * NPHIsh)

    # total porosity

    POR = (NPHIshcor + DPHIshcor)/2

    # effective porosity

    well['PHIE'] = POR * (1 - VCL)

    # update LAS file

    las.append_curve('PHIE', well['PHIE'], unit = 'V/V', descr = 'Effective Porosity', value = '')

    return las, well

# calculate volume of clay, sand-shale lithology and effective porosity

for las, well in zip(lases, wells):
    vcl_df = vcl_cal(well)
    las, well = vcl_corr(las, well, vcl_df)
    las, well = litho_cal(las, well)
    las, well = phie_cal(las, well)
    print('VCL, LITHO and PHIE are calculated for well %s' %las.well['WELL'].value)

# Function for neutron-density crossplot

def ndplot(las, well, tvd_top, data_range, ndplot_name):
    """
    This function is able to built a crossplot of density and neutron porosity.
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    tvd_top = formation top data in pandas data frame.
    data_range = recorded data range
    ndplot_name = the name of saved figure
    """
    # Create figure

    fig = plt.figure(figsize=(16, 8))

    gs = gridspec.GridSpec(ncols = 4, nrows = 1, width_ratios = [5, 1, 1, 1])
    axis1 = fig.add_subplot(gs[0])
    axis2 = fig.add_subplot(gs[1])
    axis3 = fig.add_subplot(gs[2], sharey = axis2)
    axis4 = fig.add_subplot(gs[3], sharey = axis2)
    
    # plot neutron porosity and density

    q1 = well.GR_NORM.quantile(0.01)
    q99 = well.GR_NORM.quantile(0.99)

    condition = (well.GR_NORM >= q1) & (well.GR_NORM <= q99)

    NPHI = well.loc[condition, 'NPHI_MRG']
    RHOB = well.loc[condition, 'RHOB_MRG']
    GR = well.loc[condition, 'GR_NORM']

    cmap = mpl.cm.jet

    im = axis1.scatter(NPHI, RHOB, c = GR, marker = '.', cmap = cmap)

    RHOBsh = RHOB.quantile(0.55)
    NPHIsh = NPHI.quantile(0.55)

    x_line = [0.5, 0, NPHIsh]
    y_line = [1.8, 2.65, RHOBsh]

    line = mlines.Line2D(x_line, y_line, color = 'black')
    axis1.add_line(line)

    labels = ['Clean Sand Line', 'Shale Line']
    sandxy = (0.25, 2.225)
    shalexy = (NPHIsh/2, RHOBsh + ((2.65 - RHOBsh)/2))
    sandang = (np.degrees(np.arctan2(0.85/2, 0.5)))
    shaleang = (np.degrees(np.arctan2((2.65 - RHOBsh)/2, NPHIsh)))
    
    for label, xylabel, angle in zip(labels, [sandxy, shalexy], [sandang, shaleang]):
        axis1.annotate(label, xy = xylabel, ha = 'center', va = 'center', rotation = angle, color = 'black',
                        path_effects = [pe.withStroke(linewidth = 3, foreground = "white")], fontsize = 10, weight = 'bold')

    axis1.set_xlabel('NPHI[V/V]')
    axis1.set_ylabel('RHOB[g/c3]')
    axis1.set_xlim(-.05, .50)
    axis1.set_ylim(3, 1.8)
    axis1.grid(True)
    
    cbar = fig.colorbar(im, ax = axis1)
    cbar.set_label('NORMALIZED GAMMA RAY')

    axis1.set_title('Neutron-Density plot of Well %s' %las.well['WELL'].value, fontsize = 15, y = 1.06)

    # input parameters

    start = data_range[0]
    stop = data_range[1]

    condition = (well.index >= start) & (well.index <= stop)

    top_depth = well.loc[condition, 'TVDSS'].dropna().min()
    bottom_depth = well.loc[condition, 'TVDSS'].dropna().max()

    # general setting for log plot

    axis2.set_ylabel('TVDSS[m]')

    for ax in [axis2, axis3, axis4]:
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False) 

    for ax in [axis3, axis4]:
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')

    # plot formations

    ax21 = axis2.twiny()
    ax21.spines['top'].set_position(('axes', 1.02))
    ax21.spines['top'].set_edgecolor('black')
    ax21.set_xlim(0, 1)
    ax21.set_xlabel('FORMATIONS', color = 'black')    
    ax21.set_xticks([0, 1])
    ax21.set_xticklabels(['', ''])

    for top, bottom, form in zip(tvd_top.TVDSS_TOP, tvd_top.TVDSS_BOTTOM, tvd_top.FORMATIONS):
        if (top >= top_depth) & (top <= bottom_depth):
            ax21.axhline(y = top, linewidth = 1.5, color = all_forms[form], alpha = 0.5)

            if (bottom <= bottom_depth):
                middle = top + (bottom - top)/2
                ax21.axhspan(top, bottom, facecolor = all_forms[form], alpha = 0.2)
                
            else:
                middle = top + (bottom_depth - top)/2
                ax21.axhspan(top, bottom_depth, facecolor = all_forms[form], alpha = 0.2)

            ax21.text(0.5, middle, form, ha = 'center', va = 'center', color = all_forms[form],
                        path_effects = [pe.withStroke(linewidth = 3, foreground = "white")], fontsize = 10, weight = 'bold')
    
    # ax21.legend(loc = 'upper left')

    ax21.grid(False)

    # plot effective porosity, rock matrix, volume of clay

    ax31 = axis3.twiny()
    ax31.plot(well.VCL, well.TVDSS, color = 'SaddleBrown', linewidth = '0.5')
    ax31.spines['top'].set_position(('axes', 1.02))
    ax31.spines['top'].set_edgecolor('SaddleBrown')
    ax31.set_xlim(0, 1)
    ax31.set_xlabel('VCL[V/V]', color = 'SaddleBrown')    
    ax31.tick_params(axis = 'x', colors = 'SaddleBrown')
    ax31.set_xticks(np.arange(0, 1.1, 0.2))
    ax31.set_xticklabels(['0', '', '', '', '','1'])
    xtick_loc(ax31)

    ax32 = axis3.twiny()
    ax32.plot(well.PHIE, well.TVDSS, color = 'gray', linewidth = '0.5')
    ax32.spines['top'].set_position(('axes', 1.10))
    ax32.spines['top'].set_edgecolor('gray')
    ax32.set_xlim(1, 0)
    ax32.set_xlabel('PHIE[V/V]', color = 'gray')    
    ax32.tick_params(axis = 'x', colors = 'gray')
    ax32.set_xticks(np.arange(1.0, -0.1, -0.2))
    ax32.set_xticklabels(['1', '', '', '', '','0'])
    xtick_loc(ax32)

    ax33 = axis3.twiny()
    ax33.set_xlim(0, 1)
    ax33.fill_betweenx(well.TVDSS, 0, well.VCL, color='SaddleBrown', capstyle = 'butt', linewidth = 0.5, label = 'VCLAY')
    ax33.fill_betweenx(well.TVDSS, well.VCL, (1 - well.PHIE), color='yellow', capstyle = 'butt', linewidth = 0.5, label = 'MATRIX')
    ax33.fill_betweenx(well.TVDSS, (1 - well.PHIE), 1, color='gray', capstyle = 'butt', linewidth = 0.5, label = 'POROSITY')
    ax33.set_xticks([0, 1])
    ax33.set_xticklabels(['', ''])
    # ax33.legend(loc = 'upper left')

    ax33.grid(True)

    # plot sand-shale lithology

    well['liplot'] = np.nan
    well['liplot'].loc[well.LITHO == 'SAND'] = 1
    well['liplot'].loc[well.LITHO == 'SHALE'] = 0
        
    ax41 = axis4.twiny()
    ax41.fill_betweenx(well.TVDSS, well.liplot, 1, color = 'SaddleBrown', capstyle = 'butt', linewidth = 0.01, label = 'SHALE')
    ax41.fill_betweenx(well.TVDSS, 0, well.liplot, color = 'yellow', capstyle = 'butt', linewidth = 0.01, label = 'SAND')
    ax41.spines['top'].set_position(('axes', 1.02))
    ax41.spines['top'].set_edgecolor('gray')
    ax41.set_xlim(0, 1)
    ax41.set_xlabel('LITHOLOGY', color = 'gray')
    ax41.tick_params(axis = 'x', colors = 'gray')
    ax41.set_xticks([0, 1])
    ax41.set_xticklabels(['', ''])
    ax41.legend(loc = 'upper left')

    well.drop(columns = ['liplot'], inplace = True)

    fig.tight_layout()

    # Save files

    ndplot_folder = 'Mechanical Stratigraphy'
    ndplot_path = os.path.join(sav_path, ndplot_folder)

    if not os.path.isdir(ndplot_path):
        os.makedirs(ndplot_path)

    plt.savefig(os.path.join(ndplot_path, ndplot_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# generate neutron-density crossplot

for las, well, tvd_top, data_range in zip(lases, wells, tvd_tops, data_ranges):
    ndplot_name = '%s_NDplot.png' %las.well['WELL'].value
    ndplot(las, well, tvd_top, data_range, ndplot_name)

# Function for multiwell neutron-density crossplot

def multi_ndplot(lases, wells, well_names, multi_ndplot_name):
    """
    This function is able to built a crossplot of density and neutron porosity.
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    well_names = list of well name and its color identity
    multi_ndplot_name = the name of saved figure
    """
    # Create figure

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,8))
    fig.suptitle('Neutron-Density Plot', fontsize= 15, y = 0.98)
    
    # plot neutron porosity and density

    RHOB_q55, NPHI_q55 = [], []

    for well, name in zip(wells, well_names):
        ax.scatter(well.NPHI_MRG, well.RHOB_MRG, c = well_names[name], alpha = 0.5, marker = '.', label = name)
        NPHI_q55.append(well.NPHI_MRG.quantile(0.55))
        RHOB_q55.append(well.RHOB_MRG.quantile(0.55))

    RHOBsh = sum(RHOB_q55)/len(RHOB_q55)
    NPHIsh = sum(NPHI_q55)/len(NPHI_q55)

    x_line = [0.5, 0, NPHIsh]
    y_line = [1.8, 2.65, RHOBsh]

    line = mlines.Line2D(x_line, y_line, color = 'black')
    ax.add_line(line)

    labels = ['Clean Sand Line', 'Shale Line']
    sandxy = (0.25, 2.225)
    shalexy = (NPHIsh/2, RHOBsh + ((2.65 - RHOBsh)/2))
    sandang = (np.degrees(np.arctan2(0.85/2, 0.5)))
    shaleang = (np.degrees(np.arctan2((2.65 - RHOBsh)/2, NPHIsh)))
    
    for label, xylabel, angle in zip(labels, [sandxy, shalexy], [sandang, shaleang]):
        ax.annotate(label, xy = xylabel, ha = 'center', va = 'center', rotation = angle, color = 'black',
                    path_effects = [pe.withStroke(linewidth = 3, foreground = "white")], fontsize = 10, weight = 'bold')

    ax.set_xlabel('NPHI[V/V]')
    ax.set_ylabel('RHOB[g/c3]')
    ax.set_xlim(-.05, .50)
    ax.set_ylim(3, 1.8)
    ax.grid(True)
    ax.legend(loc = 'upper left')

    fig.tight_layout()

    # Save files

    ndplot_folder = 'Mechanical Stratigraphy'
    ndplot_path = os.path.join(sav_path, ndplot_folder)

    if not os.path.isdir(ndplot_path):
        os.makedirs(ndplot_path)

    plt.savefig(os.path.join(ndplot_path, multi_ndplot_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()
        
# generate neutron-density crossplot for all wells

multi_ndplot_name = 'NDplots.png'
multi_ndplot(lases, wells, well_names, multi_ndplot_name)

"""

4.Overburden Stress

"""

# Function for fitting a density extrapolation curve

def den_extra(las, well, tvd_top, RHOml, surface):
    """
    This function is able to fit a density extrapolation curve and create a extrapolated density.
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    tvd_top = formation top data in pandas data frame.
    RHOml = density at ground level or sea floor
    surface = position of ground level (onshore) or sea floor (offshore)
    """
    # input parameters

    A_depth = surface
    B_depth = well.loc[well.RHOB_MRG.notna(), 'TVD'].min()
    C_depth = well.loc[well.RHOB_MRG.notna(), 'TVD'].max()
    depth = well.loc[well.TVD >= surface, 'TVD']

    top_form1 = tvd_top.iloc[0].TOP
    bottom_form1 = tvd_top.iloc[0].BOTTOM

    top_form2 = tvd_top.iloc[-1].TOP
    bottom_form2 = tvd_top.iloc[-1].BOTTOM

    form1_mean = well.loc[(well.index > top_form1) & (well.index < bottom_form1), 'RHOB_MRG'].mean()
    form2_q90 = well.loc[(well.index > top_form2) & (well.index < bottom_form2), 'RHOB_MRG'].quantile(0.90)

    # density and position for each point

    A = (RHOml, A_depth)
    B = (form1_mean, B_depth)
    C = (form2_q90, C_depth)

    RHOBex = (A[0], B[0], C[0])
    TVDs = (A[1], B[1], C[1])
    RHOs = (RHOml, RHOml, RHOml)
    X = (RHOs, TVDs)

    popt, pcov = curve_fit(den_extra_eq, X, RHOBex)

    well['RHOB_EX'] = RHOml + (popt[0] * (depth**popt[1]))

    # update las file

    las.append_curve('RHOB_EX', well['RHOB_EX'], unit = las.curves['RHOB_MRG'].unit, descr = 'Extrapolated Density', value = '')

    return las, well

# Function for overburden stress calculation using integration

def obp_cal(las, well, tvd_top, surface, field_type):
    """
    This function is density integration along TVD depth for overburden pressure and its gradient (OBG).
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    tvd_top = formation top data in pandas data frame.
    surface = position of ground level (onshore) or sea floor (offshore)
    field_type = field type either onshore or offshore
    """
    # prepare density data (input)

    connect_depth = well.loc[well.RHOB_MRG.notna(), 'TVD'].min()
    well['obp_input'] = well.RHOB_MRG
    well.loc[well.TVD < connect_depth, 'obp_input'] = well.RHOB_EX

    well = fill_mean(well, tvd_top, 'obp_input')

    # set parameter

    condition = well.obp_input.notna()

    density = well.loc[condition, ['TVD', 'obp_input']].obp_input
    depth = well.loc[condition, ['TVD', 'obp_input']].TVD

    # calculate OBP and OBG using integration along TVD depth (output unit = MPa)

    well['OBP'] = np.nan
    
    if field_type == 'onshore':
        well.loc[condition, 'OBP'] = constants.g * cumtrapz(density, depth, initial = 0) * 1e3 * 0.000145038 # 0.000145038 for Pa to psi

    else:
        # water pressure gredient
        wg = 0.44 # psi/ft
        water_zone = wg * surface * (3.28084) # 3.28084 for m to ft
        well.loc[condition, 'OBP'] = water_zone + (constants.g * cumtrapz(density, depth, initial = 0) * 1e3 * 0.000145038) # 0.000145038 for Pa to psi

    well['OBG'] = (well.OBP / well.TVD) * (1/3.28084) * (1/0.052) # 3.28084 for m to ft, 1/0.052 for psi/ft to ppg

    well.drop(columns = ['obp_input'], inplace = True)

    # update LAS file

    las.append_curve('OBP', well['OBP'], unit = 'psi', descr = 'Overburden Pressure', value = '')
    las.append_curve('OBG', well['OBG'], unit = 'ppg', descr = 'Overburden Gradient', value = '')

    return las, well

# Function for replacing remained nan within bad zone by mean value of each formation

def fill_mean(well, tvd_top, col):
    """
    This function is replacing remained nan within bad zone by mean value of each formation in case of curve cann't be synthesized. I
    well = well logging data in pandas data frame with alias applied.
    tvd_top = formation top data in pandas data frame which contains:
            1. Formation names in column name "Formations"
            2. Top depth boundary of the formation in column name "Top_TVD"
            3. Bottom depth boundary of the formation in column name "Bottom_TVD"
    col = column names or curve names to be filled in with mean value.
    """
    # get all mean value from all cols for each for

    for form, top, bottom in zip(tvd_top.FORMATIONS, tvd_top.TOP, tvd_top.BOTTOM):

        condition1 = (well.index > top) & (well.index < bottom)
        condition2 = condition1 & (well.BHF == 'BAD') & (well[col].isna())

        mean = well.loc[condition1, col].mean()
        well.loc[condition2, col] = mean

    return well

# Function for density extrapolation

def den_extra_eq(X, A0, alpha):
    """
    This function is density extrapolation equation.
    RHOml = density at ground level or sea floor
    TVD = true vertival depth (air gap was removed)
    A0 = fitting parameter 1
    alpha = fitting parameter 2
    """
    # independent variables

    RHOml, TVD = X

    # density extrapolation equation

    RHOex = RHOml + (A0 * (TVD**alpha))

    return RHOex

# Function for getting plot scale of pressure plot

def plot_scale(max_scale, min_scale, increment):
    """
    This function is for getting plot scale of pressure or gradient plot.
    max_scale = maximum value
    min_scale = minimum value
    increment = increment value
    """
    x = increment
    start = np.ceil(float(min_scale)/increment) * increment
    stop = (np.ceil(float(max_scale)/increment) * increment) + (increment * 1.1)
    scale = np.arange(start, stop, increment)
    
    while len(scale) > 8:
        x += increment
        scale = np.arange(start, stop, x)

    scalelist = [str(int(start))]
    for i in range(len(scale) - 2):
        scalelist.append('')
    scalelist.append(str(int(scale[-1])))

    return scale, scalelist

# calculate overburden pressure

for las, well, tvd_top, RHOml, wl in zip(lases, wells, tvd_tops, RHOmls, water_levels):

    if field_type == 'onshore':
        surface = 0
    
    else:
        surface = wl

    las, well = den_extra(las, well, tvd_top, RHOml, surface)
    las, well = obp_cal(las, well, tvd_top, surface, field_type)

    print('Overburden pressure and its gradient are calculated for well %s' %las.well['WELL'].value)

# Function for overburden stress plot

def obpplot(las, well, obp_name):
    """
    This function is plotting the overburden stress with its input (density).
    las = las file (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    obp_name = name of saved figure.
    """
    # Create figure and subplots
    
    fig, axis = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 12), sharey = True)
    fig.suptitle('Overburden Pressure of Well %s' %las.well['WELL'].value, fontsize= 20, y = 1.0)
    
    # General setting for all axis

    axis[0].set_ylabel('TVD[m]')

    top_depth = 0
    bottom_depth = well.TVD.max()
    
    for ax in axis:
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False) 
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')
    
    # Density and extropolated density
    
    ax11 = axis[0].twiny()
    ax11.plot(well.RHOB_MRG, well.TVD, color = 'red', linewidth = '0.5')
    ax11.spines['top'].set_position(('axes', 1.02))
    ax11.spines['top'].set_edgecolor('red')
    ax11.set_xlim(0, 3)
    ax11.set_xlabel('RHOB[g/c3]', color = 'red')    
    ax11.tick_params(axis = 'x', colors = 'red')
    ax11.set_xticks(np.arange(0, 3.1, 0.5))
    ax11.set_xticklabels(['0', '', '', '', '', '', '3'])
    xtick_loc(ax11)
    
    ax11.grid(True)

    ax12 = axis[0].twiny()
    ax12.plot(well.RHOB_EX, well.TVD, color = 'black', linewidth = '1')
    ax12.spines['top'].set_position(('axes', 1.08))
    ax12.spines['top'].set_edgecolor('black')   
    ax12.set_xlim(0, 3)
    ax12.set_xlabel('RHOB_EX[g/c3]', color = 'black')    
    ax12.tick_params(axis = 'x', colors = 'black')
    ax12.set_xticks(np.arange(0, 3.1, 0.5))
    ax12.set_xticklabels(['0', '', '', '', '', '', '3'])
    xtick_loc(ax12)
    
    # Overdurden Pressure

    scale1, scalelist1 = plot_scale(well.OBP.max(), 0, 1000)
    
    ax21 = axis[1].twiny()
    ax21.plot(well.OBP, well.TVD, color = 'black', linewidth = '1')
    ax21.spines['top'].set_position(('axes', 1.02))
    ax21.spines['top'].set_edgecolor('black')
    ax21.set_xlim(scale1[0], scale1[-1])
    ax21.set_xlabel('OBP[psi]', color = 'black')    
    ax21.tick_params(axis = 'x', colors = 'black')
    ax21.set_xticks(scale1)
    ax21.set_xticklabels(scalelist1)
    xtick_loc(ax21)

    ax21.grid(True)

    # Overdurden Gradient

    scale2, scalelist2 = plot_scale(well.OBG.max(), 0, 2)

    ax31 = axis[2].twiny()
    ax31.plot(well.OBG, well.TVDSS, color = 'black', linewidth = '1')
    ax31.spines['top'].set_position(('axes', 1.02))
    ax31.spines['top'].set_edgecolor('black')
    ax31.set_xlim(scale2[0], scale2[-1])
    ax31.set_xlabel('OBG[ppg]', color = 'black')    
    ax31.tick_params(axis = 'x', colors = 'black')
    ax31.set_xticks(scale2)
    ax31.set_xticklabels(scalelist2)
    xtick_loc(ax31)

    ax31.grid(True)
        
    fig.tight_layout()

    # Save files

    obp_folder = 'Overburden Pressure'
    obp_path = os.path.join(sav_path, obp_folder)

    if not os.path.isdir(obp_path):
        os.makedirs(obp_path)

    plt.savefig(os.path.join(obp_path, obp_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# plotting overburden stress with its input

for las, well in zip(lases, wells):
    obp_name = 'LQC_%s_OBP.png' %las.well['WELL'].value
    obpplot(las, well, obp_name)

# Function for overburden stress plot

def multi_obpplot(lases, wells, well_names, multi_obp_name):
    """
    This function is plotting the overburden stress with the other.
    lasas = list of las file (.las) of the well data
    wells = list of well logging data in pandas data frame with alias applied.
    well_names = list of well name with its identity color
    multi_obp_name = name of saved figure.
    """
    # Create figure and subplots
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 10), sharey = True)
    fig.suptitle('Overburden Pressure', fontsize = 16, y = 1.0)
    
    # General setting for all axis

    top_depth = 0
    bottom_depth = max([well.TVD.max() for well in wells])

    max_scale = max([well.OBP.max() for well in wells])
    scale, scalelist = plot_scale(max_scale, 0, 1000)

    # overburden pressure

    for well, name in zip(wells, well_names):
        ax.plot(well.OBP, well.TVD, color = well_names[name], linewidth = '1', label = name)

    ax.set_ylim(top_depth, bottom_depth)
    ax.set_ylabel('TVD[m]')
    ax.invert_yaxis()
    ax.minorticks_on() #Scale axis
    ax.get_xaxis().set_visible(True)
    ax.set_xlim(scale[0], scale[-1]) 
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('OBP[psi]', color = 'black')
    ax.tick_params(axis = 'x', colors = 'black')
    ax.set_xticks(scale)
    ax.set_xticklabels(list(map(int, scale)))
    ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
    ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')
    ax.grid(True)
    ax.legend(loc = 'upper right')
        
    fig.tight_layout()

    # Save files

    obp_folder = 'Overburden Pressure'
    obp_path = os.path.join(sav_path, obp_folder)

    if not os.path.isdir(obp_path):
        os.makedirs(obp_path)

    plt.savefig(os.path.join(obp_path, multi_obp_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# plotting overburden stress with the other

multi_obp_name = 'LQC_multi_wells_OBP.png'
multi_obpplot(lases, wells, well_names, multi_obp_name)

"""

5.Hydrostatic Presure

"""

# Function for hydrostatic pressure calculation

def hydro_cal(las, well):
    """
    This function is for hydrostatic pressure and its gradient calculation.
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    """
    # hydrostatic pressure gredient
    
    wg = 0.44 # psi/ft

    # calculate using integration along TVD depth

    well['HYDRO'] = wg * well.TVD * (3.28084) # 3.28084 for m to ft
    well.loc[well.HYDRO < 0, 'HYDRO'] = np.nan

    well['WPG'] = (well.HYDRO / well.TVD) * (1/3.28084) * (1/0.052) # 3.28084 for m to ft, 1/0.052 for psi/ft to ppg

    # update LAS file

    las.append_curve('HYDRO', well['HYDRO'], unit = 'psi', descr = 'Hydrostatic Pressure', value = '')
    las.append_curve('WPG', well['WPG'], unit = 'ppg', descr = 'Water Pressure Gradient', value = '')

    return las, well

# calculate hydrostatic pressure

for las, well in zip(lases, wells):
    las, well = hydro_cal(las, well)
    print('Hydrostatic pressure and its gradient are calculated for well %s' %las.well['WELL'].value)

# Function for calculation of pressure in psi unit to its gradient in ppg unit using TVD

def get_gradient(df):
    """
    This function is for calculation of pressure in psi unit to its gradient in ppg unit using TVD.
    df = data frame contains:
        1. True vertical depth (TVD) in column name "TVD"
        2. Pressure (psi) in column name "PRESSURE"
    """
    # convert pressure (psia) to pressure (psig)

    df['GAUGE'] = df.PRESSURE - 14.7

    # gradient calculation

    df['GRADIENT'] = (df.GAUGE/df.TVD) * (1/3.28084) * (1/0.052) # 3.28084 for m to ft, 1/0.052 for psi/ft to ppg

    return df

# Function for calculation of its gradient in SG EMW unit to pressure in psi unit using TVD

def get_pressure(df):
    """
    This function is for calculation of its gradient in ppg unit to pressure in psi unit using TVD.
    df = data frame contains:
        1. True vertical depth (TVD) in column name "TVD"
        2. Gradient (SG EMW) in column name "PRESSURE"
    """
    # pressure calculation

    df['PRESSURE'] = (df.GRADIENT * 0.052) * (df.TVD * 3.28084)  # 0.052 for ppg to psi/ft, 3.28084 for m to ft

    return df

# calculate pressure and its gradient for mud weight and pressure test

for mud, pres in zip(tvd_muds, tvd_press):
    mud['GRADIENT'] = mud.MUDWEIGHT
    mud = get_pressure(mud)
    pres = get_gradient(pres)

# Function for overburden and hydrostatic presssures plot

def ppplot(las, well, tvd_top, tvd_mud, all_forms, tvd_press, well_names, pp_name):
    """
    This function is plotting the overburden pressure with hydrostatic presure.
    las = las file (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    tvd_top = formation top data in pandas data frame.
    tvd_mud = tvd_mud weight log in pandas data frame.
    all_forms = list of all formation names with color code in dictionary format
    tvd_press = lsit of pressure test in panda data frame
    well_names = list of well name and its color identity
    pp_name = name of saved figure.
    """
    # Create figure and subplots
    
    fig, axis = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 12), sharey = True)
    fig.suptitle('Overburden and Hydrostatic Pressures of Well %s' %las.well['WELL'].value, fontsize = 16, y = 1.0)
    
    # General setting for all axis

    axis[0].set_ylabel('TVD[m]')

    top_depth = 0
    bottom_depth = well.TVD.max()
    
    for ax in axis:
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False) 
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')
    
        # Plot formations

        for top, bottom, form in zip(tvd_top.TVD_TOP, tvd_top.TVD_BOTTOM, tvd_top.FORMATIONS):
            if (top >= top_depth) & (top <= bottom_depth):
                ax.axhline(y = top, linewidth = 1.5, color = all_forms[form], alpha = 0.8)

                if (bottom <= bottom_depth):
                    middle = top + (bottom - top)/2
                    ax.axhspan(top, bottom, facecolor = all_forms[form], alpha = 0.2)
                
                else:
                    middle = top + (bottom_depth - top)/2
                    ax.axhspan(top, bottom_depth, facecolor = all_forms[form], alpha = 0.2)
                    
                ax.text(0.01, middle , form, ha = 'left', va = 'center', color = all_forms[form], 
                        path_effects = [pe.withStroke(linewidth = 3, foreground = "white")], fontsize = 10, weight = 'bold')

    # Pressures
    max_pressure_test = max([pres.GAUGE.max() for pres in tvd_press])
    max_scale1 = max([well.OBP.max(), well.HYDRO.max(), tvd_mud.PRESSURE.max(), max_pressure_test])

    min_pressure_test = min([pres.GAUGE.min() for pres in tvd_press])
    min_scale1 = min([well.OBP.min(), well.HYDRO.min(), tvd_mud.PRESSURE.min(), min_pressure_test])

    scale1, scalelist1 = plot_scale(max_scale1, min_scale1, 1000)
    
    ax11 = axis[0].twiny()
    ax11.plot(well.OBP, well.TVD, color = 'black', linewidth = '1')
    ax11.spines['top'].set_position(('axes', 1.02))
    ax11.spines['top'].set_edgecolor('black')
    ax11.set_xlim(scale1[0], scale1[-1])
    ax11.set_xlabel('OBP[psi]', color = 'black')    
    ax11.tick_params(axis = 'x', colors = 'black')
    ax11.set_xticks(scale1)
    ax11.set_xticklabels(scalelist1)
    xtick_loc(ax11)

    ax11.grid(True)

    ax12 = axis[0].twiny()
    ax12.plot(well.HYDRO, well.TVD, color = 'lightskyblue', linewidth = '1')
    ax12.spines['top'].set_position(('axes', 1.08))
    ax12.spines['top'].set_edgecolor('lightskyblue')   
    ax12.set_xlim(scale1[0], scale1[-1])
    ax12.set_xlabel('HYDRO[psi]', color = 'lightskyblue')    
    ax12.tick_params(axis = 'x', colors = 'lightskyblue')
    ax12.set_xticks(scale1)
    ax12.set_xticklabels(scalelist1)
    xtick_loc(ax12)

    ax13 = axis[0].twiny()
    ax13.plot(tvd_mud.PRESSURE, tvd_mud.TVD, color = 'green', linewidth = '1')
    ax13.spines['top'].set_position(('axes', 1.14))
    ax13.spines['top'].set_edgecolor('green')   
    ax13.set_xlim(scale1[0], scale1[-1])
    ax13.set_xlabel('MW[psi]', color = 'green')    
    ax13.tick_params(axis = 'x', colors = 'green')
    ax13.set_xticks(scale1)
    ax13.set_xticklabels(scalelist1)
    xtick_loc(ax13)

    # pressure test data plot

    ax14 = axis[0].twiny() 

    for pres, name in zip(tvd_press, well_names):
        ax14.scatter(pres.GAUGE, pres.TVD, c = well_names[name], alpha = 0.5, marker = 'o', label = name)
    
    ax14.get_xaxis().set_visible(False) 
    ax14.set_xlim(scale1[0], scale1[-1])
    ax14.legend(loc = 'upper left')

    # Gradients

    max_gradient_test = max([pres.GRADIENT.max() for pres in tvd_press])
    max_scale2 = max([well.OBG.max(), well.WPG.max(), tvd_mud.GRADIENT.max(), max_gradient_test])

    min_gradient_test = min([pres.GRADIENT.min() for pres in tvd_press])
    min_scale2 = min([well.OBG.min(), well.WPG.min(), tvd_mud.GRADIENT.min(), min_gradient_test])

    scale2, scalelist2 = plot_scale(max_scale2, min_scale2, 2)

    ax21 = axis[1].twiny()
    ax21.plot(well.OBG, well.TVDSS, color = 'black', linewidth = '1')
    ax21.spines['top'].set_position(('axes', 1.02))
    ax21.spines['top'].set_edgecolor('black')
    ax21.set_xlim(scale2[0], scale2[-1])
    ax21.set_xlabel('OBG[ppg]', color = 'black')    
    ax21.tick_params(axis = 'x', colors = 'black')
    ax21.set_xticks(scale2) 
    ax21.set_xticklabels(scalelist2)
    xtick_loc(ax21)

    ax21.grid(True)

    ax22 = axis[1].twiny()
    ax22.plot(well.WPG, well.TVD, color = 'deepskyblue', linewidth = '1')
    ax22.spines['top'].set_position(('axes', 1.08))
    ax22.spines['top'].set_edgecolor('deepskyblue')   
    ax22.set_xlim(scale2[0], scale2[-1])
    ax22.set_xlabel('WPG[ppg]', color = 'deepskyblue')    
    ax22.tick_params(axis = 'x', colors = 'deepskyblue')
    ax22.set_xticks(scale2) 
    ax22.set_xticklabels(scalelist2)
    xtick_loc(ax22)

    ax23 = axis[1].twiny()
    ax23.plot(tvd_mud.GRADIENT, tvd_mud.TVD, color = 'green', linewidth = '1')
    ax23.spines['top'].set_position(('axes', 1.14))
    ax23.spines['top'].set_edgecolor('green')   
    ax23.set_xlim(scale2[0], scale2[-1])
    ax23.set_xlabel('MW[ppg]', color = 'green')    
    ax23.tick_params(axis = 'x', colors = 'green')
    ax23.set_xticks(scale2) 
    ax23.set_xticklabels(scalelist2)
    xtick_loc(ax23)

    # pressure test data plot

    ax24 = axis[1].twiny() 

    for pres, name in zip(tvd_press, well_names):
        ax24.scatter(pres.GRADIENT, pres.TVD, c = well_names[name], alpha = 0.5, marker = 'o', label = name)
    
    ax24.get_xaxis().set_visible(False) 
    ax24.set_xlim(scale2[0], scale2[-1])
    ax24.legend(loc = 'upper left')

    fig.tight_layout()

    # Save files

    pp_folder = 'Pore Pressure'
    pp_path = os.path.join(sav_path, pp_folder)

    if not os.path.isdir(pp_path):
        os.makedirs(pp_path)

    plt.savefig(os.path.join(pp_path, pp_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

for las, well, tvd_top, tvd_mud in zip(lases, wells, tvd_tops, tvd_muds):
    pp_name = 'LQC_%s_PP.png' %las.well['WELL'].value
    ppplot(las, well, tvd_top, tvd_mud, all_forms, tvd_press, well_names, pp_name)

"""

6.Rock Strength & Elastic Properties

"""

# Function for scientific notation plot

def sci_scale(axis, scalelist):
    """
    This function is for scientific notation of tick labels
    axis = plot axis or column
    scalelist = list of scale plot
    """
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))

    tick_func = mticker.FuncFormatter(g)
    axis.xaxis.set_major_formatter(tick_func)

    xticks = axis.xaxis.get_major_ticks()
    for i in range(len(scalelist) - 2):
        xticks[i+1].label2.set_visible(False)

# Function for calculated young's modulus plot

def yme_plot(las, syme_df, data_range, yme_name):
    """
    Plot the calculated young's modulus curves for checking.
    las = las file (.las) of the syme_df data
    syme_df = calculated young's modulus data in pandas data frame with alias applied.
    data_range = data range
    yme_name = name of saved figure.
    """
    # Create figure and subplots
    
    fig, axis = plt.subplots(nrows = 1, ncols = 8, figsize = (16,21), sharey = True)
    fig.suptitle('Static Young\'s modulus curves of %s' %las.well['WELL'].value, fontsize= 20, y = 1.0)
    
    # General setting for all axis

    axis[0].set_ylabel('MD[m]')

    top_depth = data_range[0]
    bottom_depth = data_range[1]
    
    for ax in axis:
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False) 
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')

    max_scale = syme_df.CYME.max() + syme_df.CYME.min()

    scale, scalelist = plot_scale(max_scale, 0, 10000)

    # DYME plots
    
    ax11 = axis[0].twiny()
    ax11.plot(syme_df.DYME, syme_df.index, color = 'chocolate', linewidth = '0.8')
    ax11.spines['top'].set_position(('axes', 1.02))
    ax11.spines['top'].set_edgecolor('chocolate')
    ax11.set_xlim(scale[0], scale[-1])
    ax11.set_xlabel('DYME[psi]', color = 'chocolate')    
    ax11.tick_params(axis = 'x', colors = 'chocolate')
    ax11.set_xticks(scale)
    sci_scale(ax11, scalelist)
    xtick_loc(ax11)
    
    ax11.grid(True)

    # SYME1 plots
    
    ax21 = axis[1].twiny()
    ax21.plot(syme_df.SYME1, syme_df.index, color = 'chocolate', linewidth = '0.8')
    ax21.spines['top'].set_position(('axes', 1.02))
    ax21.spines['top'].set_edgecolor('chocolate')
    ax21.set_xlim(scale[0], scale[-1])
    ax21.set_xlabel('SYME1[psi]', color = 'chocolate')    
    ax21.tick_params(axis = 'x', colors = 'chocolate')
    ax21.set_xticks(scale)
    sci_scale(ax21, scalelist)
    xtick_loc(ax21)

    ax21.scatter(syme_df.CYME, syme_df.index, color = 'black', label = 'core', marker = 'o')
    ax21.legend(loc = 'upper right')
    
    ax21.grid(True)
 
    # SYME2 plots
    
    ax31 = axis[2].twiny()
    ax31.plot(syme_df.SYME2, syme_df.index, color = 'chocolate', linewidth = '0.8')
    ax31.spines['top'].set_position(('axes', 1.02))
    ax31.spines['top'].set_edgecolor('chocolate')
    ax31.set_xlim(scale[0], scale[-1])
    ax31.set_xlabel('SYME2[psi]', color = 'chocolate')    
    ax31.tick_params(axis = 'x', colors = 'chocolate')
    ax31.set_xticks(scale)
    sci_scale(ax31, scalelist)
    xtick_loc(ax31)

    ax31.scatter(syme_df.CYME, syme_df.index, color = 'black', label = 'core', marker = 'o')
    ax31.legend(loc = 'upper right')
    
    ax31.grid(True)

    # SYME3 plots
    
    ax41 = axis[3].twiny()
    ax41.plot(syme_df.SYME3, syme_df.index, color = 'chocolate', linewidth = '0.8')
    ax41.spines['top'].set_position(('axes', 1.02))
    ax41.spines['top'].set_edgecolor('chocolate')
    ax41.set_xlim(scale[0], scale[-1])
    ax41.set_xlabel('SYME3[psi]', color = 'chocolate')    
    ax41.tick_params(axis = 'x', colors = 'chocolate')
    ax41.set_xticks(scale)
    sci_scale(ax41, scalelist)
    xtick_loc(ax41)

    ax41.scatter(syme_df.CYME, syme_df.index, color = 'black', label = 'core', marker = 'o')
    ax41.legend(loc = 'upper right')
    
    ax41.grid(True)

    # SYME4 plots
    
    ax51 = axis[4].twiny()
    ax51.plot(syme_df.SYME4, syme_df.index, color = 'chocolate', linewidth = '0.8')
    ax51.spines['top'].set_position(('axes', 1.02))
    ax51.spines['top'].set_edgecolor('chocolate')
    ax51.set_xlim(scale[0], scale[-1])
    ax51.set_xlabel('SYME4[degree]', color = 'chocolate')    
    ax51.tick_params(axis = 'x', colors = 'chocolate')
    ax51.set_xticks(scale)
    sci_scale(ax51, scalelist)
    xtick_loc(ax51)

    ax51.scatter(syme_df.CYME, syme_df.index, color = 'black', label = 'core', marker = 'o')
    ax51.legend(loc = 'upper right')
    
    ax51.grid(True)

    # SYME5 plots
    
    ax61 = axis[5].twiny()
    ax61.plot(syme_df.SYME5, syme_df.index, color = 'chocolate', linewidth = '0.8')
    ax61.spines['top'].set_position(('axes', 1.02))
    ax61.spines['top'].set_edgecolor('chocolate')
    ax61.set_xlim(scale[0], scale[-1])
    ax61.set_xlabel('SYME5[psi]', color = 'chocolate')    
    ax61.tick_params(axis = 'x', colors = 'chocolate')
    ax61.set_xticks(scale)
    sci_scale(ax61, scalelist)
    xtick_loc(ax61)
    
    ax61.scatter(syme_df.CYME, syme_df.index, color = 'black', label = 'core', marker = 'o')
    ax61.legend(loc = 'upper right')
    
    ax61.grid(True)

    # SYME6 plots
    
    ax71 = axis[6].twiny()
    ax71.plot(syme_df.SYME6, syme_df.index, color = 'chocolate', linewidth = '0.8')
    ax71.spines['top'].set_position(('axes', 1.02))
    ax71.spines['top'].set_edgecolor('chocolate')
    ax71.set_xlim(scale[0], scale[-1])
    ax71.set_xlabel('SYME6[psi]', color = 'chocolate')    
    ax71.tick_params(axis = 'x', colors = 'chocolate')
    ax71.set_xticks(scale)
    sci_scale(ax71, scalelist)
    xtick_loc(ax71)

    ax71.scatter(syme_df.CYME, syme_df.index, color = 'black', label = 'core', marker = 'o')
    ax71.legend(loc = 'upper right')
    
    ax71.grid(True)

    # SYME7 plots
    
    ax81 = axis[7].twiny()
    ax81.plot(syme_df.SYME7, syme_df.index, color = 'chocolate', linewidth = '0.8')
    ax81.spines['top'].set_position(('axes', 1.02))
    ax81.spines['top'].set_edgecolor('chocolate')
    ax81.set_xlim(scale[0], scale[-1])
    ax81.set_xlabel('SYME7[psi]', color = 'chocolate')    
    ax81.tick_params(axis = 'x', colors = 'chocolate')
    ax81.set_xticks(scale)
    sci_scale(ax81, scalelist)
    xtick_loc(ax81)

    ax81.scatter(syme_df.CYME, syme_df.index, color = 'black', label = 'core', marker = 'o')
    ax81.legend(loc = 'upper right')
    
    ax81.grid(True)
        
    fig.tight_layout()

    # Save files

    prop_folder = 'Properties'
    prop_path = os.path.join(sav_path, prop_folder)

    if not os.path.isdir(prop_path):
        os.makedirs(prop_path)

    plt.savefig(os.path.join(prop_path, yme_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# Function for convert dynamic young's modulus to static young's modulus

def yme_cor(las, well, core, data_range):
    """
    This function is for static young's modulus correlation from dynamic value using empirical equations (Najibi et al. 2015).
    Sedimentary equations:
        - Eissa and Kazi 1 (1988)
        - Eissa and Kazi 2 (1988) using logarithm
        - Lacy (1997)
    Shale equations:
        - Ohen (2003)
        - Horsrud (2001)
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    core = core data in pandas data frame with calculated true vertical depth
    data_range = data range
    """
    # prepare input parameters

    dyme_sand = well.loc[well.LITHO == 'SAND', 'DYME'].dropna()
    rhob_sand = well.loc[well.LITHO == 'SAND', 'RHOB_MRG'].dropna()
    
    dyme_shale = well.loc[well.LITHO == 'SHALE', 'DYME'].dropna()
    dtc_shale = well.loc[well.LITHO == 'SHALE', 'DTC_MRG'].dropna()
    Vp_shale = (1/dtc_shale)

    # convert psi unit to GPa unit for empirical equations

    dyme_sand *= 6.8948e-6 # 6.8948e-6 for psi unit to GPa unit
    
    dyme_shale *= 6.8948e-6
    Vp_shale *= 304.800 # 304.800 for ft/us to km/s
 
    # calculate static value for sand in GPa unit

    syme_sand1 = (0.74 * dyme_sand) - 0.82                                    # Eissa and Kazi 1 (1988)
    syme_sand2 = 10 ** (0.02 + (0.7 * (np.log10(rhob_sand * dyme_sand))))       # Eissa and Kazi 2 (1988)
    syme_sand3 = (0.018 * (dyme_sand**2)) + (0.422 * dyme_sand)                 # Lacy (1997)

    # calculate static value for shale in GPa unit

    syme_shale1 = 0.0158 * (dyme_shale**2.74)                                   # Ohen (2003)
    syme_shale2 = 0.076 * (Vp_shale**3.23)                                      # Horsrud (2001)

    # convert GPa unit to psi unit

    syme_sand1 *= 145038 # 145038 for GPa unit to psi unit
    syme_sand2 *= 145038
    syme_sand3 *= 145038
    
    syme_shale1 *= 145038
    syme_shale2 *= 145038

    """
    Data Merging
    """
    # set equation names and columns

    equations = {
    'SYME1': 'Eissa and Kazi 1 (1988) and Ohen (2003) for sandstone and shale respectively',
    'SYME2': 'Eissa and Kazi 1 (1988) and Horsrud (2001) for sandstone and shale respectively',
    'SYME3': 'Eissa and Kazi 2 (1988) and Ohen (2003) for sandstone and shale respectively',
    'SYME4': 'Eissa and Kazi 2 (1988) and Horsrud (2001) for sandstone and shale respectively',
    'SYME5': 'Lacy (1997) and Ohen (2003) for sandstone and shale respectively',
    'SYME6': 'Lacy (1997) and Horsrud (2001) for sandstone and shale respectively',
    'SYME7': 'user (customizable)'
    }

    # set static Young's modulus for sand

    syme_df = pd.DataFrame().reindex(well.index)
    syme_df['DYME'] = well.DYME
    syme_df['CYME'] = core.set_index('MD').YME

    for eq in [eq for eq in equations][:2]:
        syme_df[eq] = syme_sand1

    for eq in [eq for eq in equations][2:4]:
        syme_df[eq] = syme_sand2

    for eq in [eq for eq in equations][4:6]:
        syme_df[eq] = syme_sand3

    # merge with static Young's modulus for shale

    for eq in [eq for eq in equations][:5:2]:
        syme_df[eq].fillna(syme_shale1, inplace = True)

    for eq in [eq for eq in equations][1:6:2]:
        syme_df[eq].fillna(syme_shale2, inplace = True)

    # customizable equation
    
    syme_df['SYME7'] = syme_df.DYME * 0.8

    # Check calculated young's modulus curves

    yme_name = 'LQC_%s_YME.png' %las.well['WELL'].value
    yme_plot(las, syme_df, data_range, yme_name)

    while True:
    
        equa_list = [eq for eq in equations]

        select_eq = input('Please select the best calculated young\'s modulus: ').strip()

        if select_eq.lower() in [eq.lower() for eq in equa_list]:
            i = [eq.lower() for eq in equa_list].index(select_eq.lower())
            equation = equa_list[i]
            break
        else:
            print('Your %s is not matched.' %select_eq)

    print('Static Young\'s modulus is calculated using equations of %s.' %equations[equation])

    return syme_df[equation]

# Function for young's modulus computation

def yme_cal(las, well, core, data_range):
    """
    This function is for dynamic Young's modulus computation.
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    core = core data in pandas data frame with calculated true vertical depth
    data_range = data range
    """
    # prepare input parameters

    RHOB = well.RHOB_MRG # density
    DTC = well.DTC_MRG # p-slowness
    DTS = well.DTS_MRG # s-slowness

    # convert slowness (us/ft) to velosity (m/s)

    Vp = (1/DTC) * 304800 # 304800 for ft/us to m/s
    Vs = (1/DTS) * 304800

    # compute dynamic Young's modulus (psi)

    term1 = (3*(Vp**2)) - (4*(Vs**2))
    term2 = (Vp**2) - (Vs**2)

    well['DYME'] = RHOB * (Vs**2) * (term1/term2) * 0.145038 # 0.145038 for conversion factor

    print('Dynamic Young\'s modulus is calculated')

    # convert to static Young's modulus

    well['YME'] = yme_cor(las, well, core, data_range)

    # update LAS file

    las.append_curve('DYME', well['DYME'], unit = 'psi', descr = 'Dynamic Young\'s modulus', value = '')
    las.append_curve('YME', well['YME'], unit = 'psi', descr = 'Static Young\'s modulus', value = '')

    return las, well

# Function for Poisson's ratio computation

def pr_cal(las, well):
    """
    This function is for dynamic Poisson's ratio computation.
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    """
    # prepare input parameters

    DTC = well.DTC_MRG # p-slowness
    DTS = well.DTS_MRG # s-slowness

    # convert slowness (us/ft) to velosity (m/s)

    Vp = (1/DTC) * 304800 # 304800 for ft/us to m/s
    Vs = (1/DTS) * 304800

    # compute dynamic Poisson's ratio

    term1 = (Vp**2) - (2*(Vs**2))
    term2 = (Vp**2) - (Vs**2)

    well['DPR'] = 0.5 * (term1/term2)

    print('Dynamic Poisson\'s ratio is calculated')

    # convert to static Poisson's ratio

    factor = 1.0

    well['PR'] = factor * well.DPR

    print('Static Poisson\'s ratio is calculated')

    # update LAS file

    las.append_curve('DPR', well['DPR'], unit = 'unitless', descr = 'Dynamic Poisson\'s ratio', value = '')
    las.append_curve('PR', well['PR'], unit = 'unitless', descr = 'Static Poisson\'s ratio', value = '')

    return las, well

# Function for calculated young's modulus plot

def ucs_plot(las, ucs_df, data_range, ucs_name):
    """
    Plot the calculated young's modulus curves for checking.
    las = las file (.las) of the ucs_df data
    ucs_df = calculated unconfined compressive strength data in pandas data frame with alias applied.
    data_range = recorded data range
    ucs_name = name of saved figure.
    """
    # Create figure and subplots
    
    fig, axis = plt.subplots(nrows = 1, ncols = 7, figsize = (14,21), sharey = True)
    fig.suptitle('Unconfined compressive strength curves of %s' %las.well['WELL'].value, fontsize= 20, y = 1.0)
    
    # General setting for all axis

    axis[0].set_ylabel('MD[m]')

    top_depth = data_range[0]
    bottom_depth = data_range[1]
    
    for ax in axis:
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False) 
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')

    max_scale = ucs_df.CUCS.max() + ucs_df.CUCS.min()

    scale, scalelist = plot_scale(max_scale, 0, 2000)
    
    # UCS1 plots
    
    ax11 = axis[0].twiny()
    ax11.plot(ucs_df.UCS1, ucs_df.index, color = 'red', linewidth = '0.8')
    ax11.spines['top'].set_position(('axes', 1.02))
    ax11.spines['top'].set_edgecolor('red')
    ax11.set_xlim(scale[0], scale[-1])
    ax11.set_xlabel('UCS1[psi]', color = 'red')    
    ax11.tick_params(axis = 'x', colors = 'red')
    ax11.set_xticks(scale)
    ax11.set_xticklabels(scalelist)
    xtick_loc(ax11)

    ax11.scatter(ucs_df.CUCS, ucs_df.index, color = 'black', label = 'core', marker = 'o')
    ax11.legend(loc = 'upper right')
    
    ax11.grid(True)
 
    # UCS2 plots
    
    ax21 = axis[1].twiny()
    ax21.plot(ucs_df.UCS2, ucs_df.index, color = 'red', linewidth = '0.8')
    ax21.spines['top'].set_position(('axes', 1.02))
    ax21.spines['top'].set_edgecolor('red')
    ax21.set_xlim(scale[0], scale[-1])
    ax21.set_xlabel('UCS2[psi]', color = 'red')    
    ax21.tick_params(axis = 'x', colors = 'red')
    ax21.set_xticks(scale)
    ax21.set_xticklabels(scalelist)
    xtick_loc(ax21)

    ax21.scatter(ucs_df.CUCS, ucs_df.index, color = 'black', label = 'core', marker = 'o')
    ax21.legend(loc = 'upper right')
    
    ax21.grid(True)

    # UCS3 plots
    
    ax31 = axis[2].twiny()
    ax31.plot(ucs_df.UCS3, ucs_df.index, color = 'red', linewidth = '0.8')
    ax31.spines['top'].set_position(('axes', 1.02))
    ax31.spines['top'].set_edgecolor('red')
    ax31.set_xlim(scale[0], scale[-1])
    ax31.set_xlabel('UCS3[psi]', color = 'red')    
    ax31.tick_params(axis = 'x', colors = 'red')
    ax31.set_xticks(scale)
    ax31.set_xticklabels(scalelist)
    xtick_loc(ax31)

    ax31.scatter(ucs_df.CUCS, ucs_df.index, color = 'black', label = 'core', marker = 'o')
    ax31.legend(loc = 'upper right')
    
    ax31.grid(True)

    # UCS4 plots
    
    ax41 = axis[3].twiny()
    ax41.plot(ucs_df.UCS4, ucs_df.index, color = 'red', linewidth = '0.8')
    ax41.spines['top'].set_position(('axes', 1.02))
    ax41.spines['top'].set_edgecolor('red')
    ax41.set_xlim(scale[0], scale[-1])
    ax41.set_xlabel('UCS4[degree]', color = 'red')    
    ax41.tick_params(axis = 'x', colors = 'red')
    ax41.set_xticks(scale)
    ax41.set_xticklabels(scalelist)
    xtick_loc(ax41)

    ax41.scatter(ucs_df.CUCS, ucs_df.index, color = 'black', label = 'core', marker = 'o')
    ax41.legend(loc = 'upper right')
    
    ax41.grid(True)

    # UCS5 plots
    
    ax51 = axis[4].twiny()
    ax51.plot(ucs_df.UCS5, ucs_df.index, color = 'red', linewidth = '0.8')
    ax51.spines['top'].set_position(('axes', 1.02))
    ax51.spines['top'].set_edgecolor('red')
    ax51.set_xlim(scale[0], scale[-1])
    ax51.set_xlabel('UCS5[psi]', color = 'red')    
    ax51.tick_params(axis = 'x', colors = 'red')
    ax51.set_xticks(scale)
    ax51.set_xticklabels(scalelist)
    xtick_loc(ax51)
    
    ax51.scatter(ucs_df.CUCS, ucs_df.index, color = 'black', label = 'core', marker = 'o')
    ax51.legend(loc = 'upper right')
    
    ax51.grid(True)

    # UCS6 plots
    
    ax61 = axis[5].twiny()
    ax61.plot(ucs_df.UCS6, ucs_df.index, color = 'red', linewidth = '0.8')
    ax61.spines['top'].set_position(('axes', 1.02))
    ax61.spines['top'].set_edgecolor('red')
    ax61.set_xlim(scale[0], scale[-1])
    ax61.set_xlabel('UCS6[psi]', color = 'red')    
    ax61.tick_params(axis = 'x', colors = 'red')
    ax61.set_xticks(scale)
    ax61.set_xticklabels(scalelist)
    xtick_loc(ax61)

    ax61.scatter(ucs_df.CUCS, ucs_df.index, color = 'black', label = 'core', marker = 'o')
    ax61.legend(loc = 'upper right')
    
    ax61.grid(True)

    # UCS7 plots
    
    ax71 = axis[6].twiny()
    ax71.plot(ucs_df.UCS7, ucs_df.index, color = 'red', linewidth = '0.8')
    ax71.spines['top'].set_position(('axes', 1.02))
    ax71.spines['top'].set_edgecolor('red')
    ax71.set_xlim(scale[0], scale[-1])
    ax71.set_xlabel('UCS7[psi]', color = 'red')    
    ax71.tick_params(axis = 'x', colors = 'red')
    ax71.set_xticks(scale)
    ax71.set_xticklabels(scalelist)
    xtick_loc(ax71)

    ax71.scatter(ucs_df.CUCS, ucs_df.index, color = 'black', label = 'core', marker = 'o')
    ax71.legend(loc = 'upper right')
    
    ax71.grid(True)
        
    fig.tight_layout()

    # Save files

    prop_folder = 'Properties'
    prop_path = os.path.join(sav_path, prop_folder)

    if not os.path.isdir(prop_path):
        os.makedirs(prop_path)

    plt.savefig(os.path.join(prop_path, ucs_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# Function for Unconfined Compressive Strength (UCS) computation

def ucs_cal(las, well, core, data_range):
    """
    This function is for Unconfined Compressive Strength (UCS) computation using empirical equations (Chang et al. 2006).
    Sedimentary equations:
        - Bradford (1998)
        - Vernik (1993)
        - Lacy (1997)
    Shale equations:
        - Horsrud 1 (2001) using compressional wave slowness
        - Horsrud 2 (2001) using static Young's modulus
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    core = core data in pandas data frame with calculated true vertical depth
    data_range = data range
    """
    # prepare input parameters

    yme_sand = well.loc[well.LITHO == 'SAND', 'YME'].dropna()
    phie_sand = well.loc[well.LITHO == 'SAND', 'PHIE'].dropna()

    yme_shale = well.loc[well.LITHO == 'SHALE', 'YME'].dropna()
    dtc_shale = well.loc[well.LITHO == 'SHALE', 'DTC_MRG'].dropna()
    Vp_shale = (1/dtc_shale)

    # convert psi unit to GPa unit for empirical equations

    yme_sand *= 6.8948e-6 # 6.8948e-6 for psi unit to GPa unit
    
    yme_shale *= 6.8948e-6
    Vp_shale *= 304.800 # 304.800 for ft/us to km/s

    # calculate unconfined compressive strength for sand in MPa unit

    ucs_sand1 = 2.28 + (4.0189 * yme_sand)                      # Bradford (1998)
    ucs_sand2 = 254 * ((1 - (2.7 * phie_sand))**2)              # Vernik (1993)
    ucs_sand3 = (0.278 * (yme_sand**2)) + (2.458 * yme_sand)    # Lacy (1997)

    # calculate unconfined compressive strength for sand in MPa unit

    ucs_shale1 = 1.35 * (Vp_shale**2.6)                         # Horsrud 1 (2001)
    ucs_shale2 = 7.22 * (yme_shale**0.712)                      # Horsrud 2 (2001)

    # convert MPa unit to psi unit

    ucs_sand1 *= 145.038 # 145.038 for GPa unit to psi unit
    ucs_sand2 *= 145.038
    ucs_sand3 *= 145.038
    
    ucs_shale1 *= 145.038
    ucs_shale2 *= 145.038

    """
    Data Merging
    """
    # set equation names and columns

    equations = {
    'UCS1': 'Bradford (1998) and Horsrud 1 (2001) for sandstone and shale respectively',
    'UCS2': 'Bradford (1998) and Horsrud 2 (2001) for sandstone and shale respectively',
    'UCS3': 'Vernik (1993) and Horsrud 1 (2001) for sandstone and shale respectively',
    'UCS4': 'Vernik (1993) and Horsrud 2 (2001) for sandstone and shale respectively',
    'UCS5': 'Lacy (1997) and Horsrud 1 (2001) for sandstone and shale respectively',
    'UCS6': 'Lacy (1997) and Horsrud 2 (2001) for sandstone and shale respectively',
    'UCS7': 'user (customizable)'
    }

    # set unconfined compressive strength for sand

    ucs_df = pd.DataFrame().reindex(well.index)
    ucs_df['CUCS'] = core.set_index('MD').UCS

    for eq in [eq for eq in equations][:2]:
        ucs_df[eq] = ucs_sand1

    for eq in [eq for eq in equations][2:4]:
        ucs_df[eq] = ucs_sand2

    for eq in [eq for eq in equations][4:6]:
        ucs_df[eq] = ucs_sand3

    # merge with unconfined compressive strength for shale

    for eq in [eq for eq in equations][:5:2]:
        ucs_df[eq].fillna(ucs_shale1, inplace = True)

    for eq in [eq for eq in equations][1:6:2]:
        ucs_df[eq].fillna(ucs_shale2, inplace = True)

    # customizable equation

    ucs_df['UCS7'] = ucs_df.UCS2 * 0.7

    # Check calculated unconfined compressive strength curves

    ucs_name = 'LQC_%s_UCS.png' %las.well['WELL'].value
    ucs_plot(las, ucs_df, data_range, ucs_name)

    while True:
    
        equa_list = [eq for eq in equations]

        select_eq = input('Please select the best calculated Unconfined Compressive Strength: ').strip()

        if select_eq.lower() in [eq.lower() for eq in equa_list]:
            i = [eq.lower() for eq in equa_list].index(select_eq.lower())
            equation = equa_list[i]
            break
        else:
            print('Your %s is not matched.' %select_eq)

    well['UCS'] = ucs_df[equation]

    print('Unconfined Compressive Strength is calculated using equations of %s.' %equations[equation])

    # update LAS file

    las.append_curve('UCS', well['UCS'], unit = 'psi', descr = 'Unconfined compressive strength', value = '')

    return las, well

# Function for friction angle computation

def fang_cal(las, well):
    """
    This function is for friction angle computation using linear correlation method with volume of clay.
    When 100% shale relate to 20 degree friction angle
    And  100% sand relate to 40 degree friction angle
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    """
    # prepare input parameters

    vcl = well.VCL

    # calculate friction angle using volume of clay

    well['FANG'] = ((1 - vcl) + 1) * 20

    print('Angle of internal friction is calculated')

    # update LAS file

    las.append_curve('FANG', well['FANG'], unit = 'degree', descr = 'Angle of internal friction', value = '')

    return las, well

# Function for tensile strength computation

def tstr_cal(las, well):
    """
    This function is for tensile strength computation.
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    """
    # prepare input parameters

    ucs_sand = well.loc[well.LITHO == 'SAND', 'UCS'].dropna()
    ucs_shale = well.loc[well.LITHO == 'SHALE', 'UCS'].dropna()

    # calculate tensile strength

    tstr_sand = ucs_sand * 0.1
    tstr_shale = ucs_shale * 0.05

    well['TSTR'] = tstr_sand
    well['TSTR'].fillna(tstr_shale, inplace = True)

    print('Tensile formation strength is calculated')

    # update LAS file

    las.append_curve('TSTR', well['TSTR'], unit = 'psi', descr = 'Tensile formation strength', value = '')

    return las, well

# calculate young's modulas, poisson's ratio and unconfined compressive strength

for las, well, core, data_range in zip(lases, wells, tvd_cores, data_ranges):
    
    print('Well %s is being implemented.' %las.well['WELL'].value)
    
    las, well = yme_cal(las, well, core, data_range)
    las, well = pr_cal(las, well)
    las, well = ucs_cal(las, well, core, data_range)
    las, well = fang_cal(las, well)
    las, well = tstr_cal(las, well)

"""

Quality Control 3 by Boxplot

"""

# Function for check the quality of the input data for interested zone

def qc_data2(wells, tvd_tops, formation, well_names, qc_name):
    """
    This function will create the boxplot for checking the input data.
    wells = completed well data in pandas dataframe (Merged data with the synthetics)
    tvd_tops = list for formation top data in pandas data frame which contains:
            1. Formation names in column name "Formations"
            2. Top depth boundary of the formation in column name "Top_TVD"
            3. Bottom depth boundary of the formation in column name "Bottom_TVD"
    formation = input the name of the formation where the data can be compared
    well_names = list of well names with color code in dictionary format
    qc_name = name of saved file
    """
    # Set data for specific interval
    
    GR_plot, RHOB_plot, NPHI_plot, DTC_plot, DTS_plot = [list() for i in range(5)]
    YME_plot, PR_plot, UCS_plot, FANG_plot, TSTR_plot = [list() for i in range(5)]

    data_plots = [GR_plot, RHOB_plot, NPHI_plot, DTC_plot, DTS_plot,
                  YME_plot, PR_plot, UCS_plot, FANG_plot, TSTR_plot]
    
    selected_cols = ['GR_NORM', 'RHOB_MRG', 'NPHI_MRG', 'DTC_MRG', 
                     'DTS_MRG','YME', 'PR', 'UCS', 'FANG', 'TSTR']
    
    curve_labels = ['GR', 'RHOB', 'NPHI', 'DTC', 'DTS',
                    'YME', 'PR', 'UCS', 'FANG', 'TSTR']

    unit_labels = ['API', 'g/c3', 'V/V', 'us/ft', 'us/ft',
                    'psi', 'unitless', 'psi', 'degree', 'psi']

    well_labels = []
    
    for well, top, name in zip(wells, tops, well_names):
        
        # Check available data for selected formation
        
        if formation in list(top.FORMATIONS):
            
            # Set interval from each well for selected formation

            top_depth = float(top.loc[top.FORMATIONS == formation, 'TOP'])
            bottom_depth = float(top.loc[top.FORMATIONS == formation, 'BOTTOM'])

            well_labels.append(name)
            
            # Select data from each well by interval

            condition = (well.index >= top_depth) & (well.index <= bottom_depth)

            for store, col in zip(data_plots, selected_cols):
                store.append(well.loc[condition, col].dropna())

    # Setup well colors for plotting

    well_colors = []

    for name in well_labels:
        well_colors.append(well_names[name])

    well_colo = [item for sublist in [(c, c) for c in well_colors] for item in sublist]
    
    # Create figure
    
    fig, axis = plt.subplots(nrows = 2, ncols = 5, figsize = (12.5, 5), sharey = False)
    fig.suptitle('Box Plot Quality Control of formation ' + '\'' + formation + '\'', fontsize = 20, y = 1.0)
    axis = axis.flatten()
    
    # Plot setting for all axis
    
    for data, label, unit, ax in zip(data_plots, curve_labels, unit_labels, axis):
        boxes = ax.boxplot(data, labels = well_labels, meanline = True, notch = True, showfliers = False, patch_artist = True)
        ax.set_ylabel('[%s]' %unit)

        # scientific notation for YME

        if label == 'YME':
            formatter = mticker.ScalarFormatter(useMathText = True)
            formatter.set_scientific(True) 
            formatter.set_powerlimits((-5,5))
            ax.yaxis.set_major_formatter(formatter)
        
        # set decoration
        for patch, color in zip(boxes['boxes'], well_colors): 
            patch.set_facecolor(color) 
        
        for box_wk, box_cap, color in zip(boxes['whiskers'], boxes['caps'], well_colo):
            box_wk.set(color = color, linewidth = 1.5)
            box_cap.set(color = color, linewidth = 3)
        
        for median in boxes['medians']:
            median.set(color = 'black', linewidth = 3) 
            
        ax.set_title(label)
    
    fig.tight_layout()

    # Save files

    qc_folder = 'LQC_Boxplot'
    qc_path = os.path.join(sav_path, qc_folder)

    if not os.path.isdir(qc_path):
        os.makedirs(qc_path)

    plt.savefig(os.path.join(qc_path, qc_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# Plot boxplot for quality comtrol each selected formation

for form in selected_forms:
    qc_name = 'LQC_%s.png' %form
    qc_data2(wells, tops, form, well_names, qc_name)

"""

Data visualization 2

"""

# Function for well data visualization in composite log plots

def composite_logs2(las, well, tvd_top, tvd_core, data_range, all_forms, logs_name2):
    """
    Plot the curves in composite logs
    las = las file (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    tvd_top = formation top data in pandas data frame.
    data_range = recorded data range
    all_forms = list of all formation names with color code in dictionary format
    logs_name2 = name of saved figure.
    """
    # Create figure and subplots
    
    fig, axis = plt.subplots(nrows = 1, ncols = 10, figsize = (22,21), sharey = True)
    fig.suptitle('Composite Log of Well %s' %las.well['WELL'].value, fontsize= 20, y = 1.0)
    
    # General setting for all axis

    axis[0].set_ylabel('TVDSS[m]')

    start = data_range[0]
    stop = data_range[1]

    condition = (well.index >= start) & (well.index <= stop)

    top_depth = well.loc[condition, 'TVDSS'].dropna().min()
    bottom_depth = well.loc[condition, 'TVDSS'].dropna().max()
    
    for ax in axis:
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False) 
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')
        
    # Plot formations

    for ax in axis[:-1]:
        for top, bottom, form in zip(tvd_top.TVDSS_TOP, tvd_top.TVDSS_BOTTOM, tvd_top.FORMATIONS):
            if (top >= top_depth) & (top <= bottom_depth):
                ax.axhline(y = top, linewidth = 1.5, color = all_forms[form], alpha = 0.8)

                if (bottom <= bottom_depth):
                    middle = top + (bottom - top)/2
                    ax.axhspan(top, bottom, facecolor = all_forms[form], alpha = 0.2)
                
                else:
                    middle = top + (bottom_depth - top)/2
                    ax.axhspan(top, bottom_depth, facecolor = all_forms[form], alpha = 0.2)
                    
                ax.text(0.01, middle , form, ha = 'left', va = 'center', color = all_forms[form], 
                        path_effects = [pe.withStroke(linewidth = 3, foreground = "white")], fontsize = 10, weight = 'bold')
    
    # Azimuth and angle plots
    
    ax11 = axis[0].twiny()
    ax11.plot(well.AZIMUTH, well.TVDSS, color = 'blue', linewidth = '0.8')
    ax11.spines['top'].set_position(('axes', 1.02))
    ax11.spines['top'].set_edgecolor('blue')
    ax11.set_xlim(0, 360)
    ax11.set_xlabel('AZIMUTH[degree]', color = 'blue')    
    ax11.tick_params(axis = 'x', colors = 'blue')
    ax11.set_xticks(np.arange(0, 361, 90))
    ax11.set_xticklabels(['0', '', '180', '', '360'])
    xtick_loc(ax11)
    
    ax11.grid(True)

    ax12 = axis[0].twiny()
    ax12.plot(well.ANGLE, well.TVDSS, color = 'red', linewidth = '0.8')
    ax12.spines['top'].set_position(('axes', 1.05))
    ax12.spines['top'].set_edgecolor('red')   
    ax12.set_xlim(0, 90)
    ax12.set_xlabel('ANGLE[degree]', color = 'red')    
    ax12.tick_params(axis = 'x', colors = 'red')
    ax12.set_xticks(np.arange(0, 91, 45))
    ax12.set_xticklabels(['0', '45', '90'])
    xtick_loc(ax12)
 
    # Gamma ray plot
    
    ax21 = axis[1].twiny()
    ax21.plot(well.GR_NORM, well.TVDSS, color = 'green', linewidth = '0.5')
    ax21.spines['top'].set_position(('axes', 1.02))
    ax21.spines['top'].set_edgecolor('green') 
    ax21.set_xlim(0, 150)
    ax21.set_xlabel('GR[API]', color = 'green')    
    ax21.tick_params(axis = 'x', colors = 'green')
    ax21.set_xticks(np.arange(0, 151, 30))
    ax21.set_xticklabels(['0', '', '', '', '','150'])
    xtick_loc(ax21)
    
    ax21.grid(True)
    
    # Resisitivity plots
    
    ax31 = axis[2].twiny()
    ax31.set_xscale('log')
    ax31.plot(well.RT, well.TVDSS, color = 'red', linewidth = '0.5')
    ax31.spines['top'].set_position(('axes', 1.02))
    ax31.spines['top'].set_edgecolor('red')
    ax31.set_xlim(0.2, 2000)
    ax31.set_xlabel('RT[ohm-m]', color = 'red')
    ax31.tick_params(axis = 'x', colors = 'red')
    ax31.set_xticks([0.2, 2, 20, 200, 2000])
    ax31.set_xticklabels(['0.2', '', '', '', '2000'])
    xtick_loc(ax31)
    
    ax31.grid(True)

    ax32 = axis[2].twiny()
    ax32.set_xscale('log')
    ax32.plot(well.MSFL, well.TVDSS, color = 'black', linewidth = '0.5')
    ax32.spines['top'].set_position(('axes', 1.05))
    ax32.spines['top'].set_edgecolor('black')
    ax32.set_xlim(0.2, 2000)
    ax32.set_xlabel('MSFL[ohm-m]', color = 'black')
    ax32.tick_params(axis = 'x', colors = 'black')
    ax32.set_xticks([0.2, 2, 20, 200, 2000])
    ax32.set_xticklabels(['0.2', '', '', '', '2000'])
    xtick_loc(ax32)
    
    # Density and neutron porosity plots
    
    ax41 = axis[3].twiny()
    ax41.plot(well.RHOB_MRG, well.TVDSS, color = 'red', linewidth = '0.5')
    ax41.spines['top'].set_position(('axes', 1.02))
    ax41.spines['top'].set_edgecolor('red')
    ax41.set_xlim(1.95, 2.95)
    ax41.set_xlabel('RHOB_MRG[g/c3]', color = 'red')    
    ax41.tick_params(axis = 'x', colors = 'red')
    ax41.set_xticks(np.arange(1.95, 2.96, 0.2))
    ax41.set_xticklabels(['1.95', '', '', '', '', '2.95'])
    xtick_loc(ax41)
    
    ax41.grid(True)

    ax42 = axis[3].twiny()
    ax42.plot(well.NPHI_MRG, well.TVDSS, color = 'blue', linewidth = '0.5')
    ax42.spines['top'].set_position(('axes', 1.05))
    ax42.spines['top'].set_edgecolor('blue')   
    ax42.set_xlim(0.45, -0.15)
    ax42.set_xlabel('NPHI_MRG[V/V]', color = 'blue')    
    ax42.tick_params(axis = 'x', colors = 'blue')
    ax42.set_xticks(np.arange(0.45, -0.16, -0.12))
    ax42.set_xticklabels(['0.45', '', '', '', '', '-0.15'])
    xtick_loc(ax42)
    
    # P_Sonic and S_Sonic plots
    
    ax51 = axis[4].twiny()
    ax51.plot(well.DTC_MRG, well.TVDSS, color = 'blue', linewidth = '0.5')
    ax51.spines['top'].set_position(('axes', 1.02))
    ax51.spines['top'].set_edgecolor('blue')
    ax51.set_xlim(140, 40)
    ax51.set_xlabel('DTC_MRG[us/ft]', color = 'blue')    
    ax51.tick_params(axis = 'x', colors = 'blue')
    ax51.set_xticks(np.arange(140, 39, -20))
    ax51.set_xticklabels(['140', '', '', '', '', '40'])
    xtick_loc(ax51)

    ax51.grid(True)

    ax52 = axis[4].twiny()
    ax52.plot(well.DTS_MRG, well.TVDSS, color = 'red', linewidth = '0.5')
    ax52.spines['top'].set_position(('axes', 1.05))
    ax52.spines['top'].set_edgecolor('red') 
    ax52.set_xlim(340, 40)
    ax52.set_xlabel('DTS_MRG[us/ft]', color = 'red')    
    ax52.tick_params(axis = 'x', colors = 'red')
    ax52.set_xticks(np.arange(340, 39, -60))
    ax52.set_xticklabels(['340', '', '', '', '', '40'])
    xtick_loc(ax52)

    # Young's modulus plot

    scale1, scalelist1 = plot_scale(well.YME.max(), 0, 10000)
    
    ax61 = axis[5].twiny()
    ax61.plot(well.YME, well.TVDSS, color = 'chocolate', linewidth = '0.5')
    ax61.spines['top'].set_position(('axes', 1.02))
    ax61.spines['top'].set_edgecolor('chocolate') 
    ax61.set_xlim(scale1[0], scale1[-1])
    ax61.set_xlabel('YME[psi]', color = 'chocolate')    
    ax61.tick_params(axis = 'x', colors = 'chocolate')
    ax61.set_xticks(scale1)
    sci_scale(ax61, scalelist1)
    xtick_loc(ax61)

    ax61.grid(True)

    ax62 = axis[5].twiny()
    ax62.scatter(tvd_core.YME, tvd_core.TVDSS, c = 'black', alpha = 1, marker = 'o')
    ax62.get_xaxis().set_visible(False) 
    ax62.set_xlim(scale1[0], scale1[-1])
    
    # Poisson's ratio plot
    
    ax71 = axis[6].twiny()
    ax71.plot(well.PR, well.TVDSS, color = 'salmon', linewidth = '0.5')
    ax71.spines['top'].set_position(('axes', 1.02))
    ax71.spines['top'].set_edgecolor('salmon') 
    ax71.set_xlim(0, 0.5)
    ax71.set_xlabel('PR[unitless]', color = 'salmon')    
    ax71.tick_params(axis = 'x', colors = 'salmon')
    ax71.set_xticks(np.arange(0, 0.51, 0.1))
    ax71.set_xticklabels(['0', '', '', '', '','0.5'])
    xtick_loc(ax71)

    ax71.grid(True)
    
    # unconfined compressive strength and Tensile formation strength plots

    max_scale = max([well.UCS.max(), well.TSTR.max()])

    scale2, scalelist2 = plot_scale(max_scale, 0, 2000)
    
    ax81 = axis[7].twiny()
    ax81.plot(well.UCS, well.TVDSS, color = 'red', linewidth = '0.5')
    ax81.spines['top'].set_position(('axes', 1.02))
    ax81.spines['top'].set_edgecolor('red')
    ax81.set_xlim(scale2[0], scale2[-1])
    ax81.set_xlabel('UCS[psi]', color = 'red')    
    ax81.tick_params(axis = 'x', colors = 'red')
    ax81.set_xticks(scale2)
    ax81.set_xticklabels(scalelist2)
    xtick_loc(ax81)

    ax81.grid(True)

    ax82 = axis[7].twiny()
    ax82.plot(well.TSTR, well.TVDSS, color = 'darkorange', linewidth = '0.5')
    ax82.spines['top'].set_position(('axes', 1.05))
    ax82.spines['top'].set_edgecolor('darkorange') 
    ax82.set_xlim(scale2[0], scale2[-1])
    ax82.set_xlabel('TSTR[psi]', color = 'darkorange')    
    ax82.tick_params(axis = 'x', colors = 'darkorange')
    ax82.set_xticks(scale2)
    ax82.set_xticklabels(scalelist2)
    xtick_loc(ax82)

    ax83 = axis[7].twiny() 
    ax83.scatter(tvd_core.UCS, tvd_core.TVDSS, c = 'black', alpha = 1, marker = 'o')
    ax83.get_xaxis().set_visible(False) 
    ax83.set_xlim(scale2[0], scale2[-1])
    
    # angle of internal friction plot
    
    ax91 = axis[8].twiny()
    ax91.plot(well.FANG, well.TVDSS, color = 'green', linewidth = '0.5')
    ax91.spines['top'].set_position(('axes', 1.02))
    ax91.spines['top'].set_edgecolor('green') 
    ax91.set_xlim(0, 50)
    ax91.set_xlabel('FANG[degree]', color = 'green')    
    ax91.tick_params(axis = 'x', colors = 'green')
    ax91.set_xticks(np.arange(0, 51, 10))
    ax91.set_xticklabels(['0', '', '', '', '','50'])
    xtick_loc(ax91)
    
    ax91.grid(True)
    
    # Bad hole flag plots

    well['bhf'] = np.nan
    well.loc[well.BHF == 'BAD', 'bhf'] = 1
    
    ax101 = axis[9].twiny()
    ax101.set_xlim(0, 1)
    ax101.fill_betweenx(well.TVDSS, 0, well.bhf, color = 'red', capstyle = 'butt', linewidth = 1, label = 'BAD')
    ax101.spines['top'].set_position(('axes', 1.02))
    ax101.spines['top'].set_edgecolor('red')
    ax101.set_xlabel('BHF', color = 'red')    
    ax101.tick_params(axis = 'x', colors = 'red')
    ax101.set_xticks([0, 1])
    ax101.set_xticklabels(['', ''])
    
    well.drop(columns = ['bhf'], inplace = True)
        
    fig.tight_layout()

    # Save files

    prop_folder = 'Properties'
    prop_path = os.path.join(sav_path, prop_folder)

    if not os.path.isdir(prop_path):
        os.makedirs(prop_path)

    plt.savefig(os.path.join(prop_path, logs_name2), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# Plot available curves

for las, well, tvd_top, tvd_core, data_range in zip(lases, wells, tvd_tops, tvd_cores, data_ranges):
    logs_name2 = '%s_Prop.png' %las.well['WELL'].value
    composite_logs2(las, well, tvd_top, tvd_core, data_range, all_forms, logs_name2)


"""

8.Minimum Stress & 9.Maximum Stress
P.S. 7.Horizontal Stress direction (skipped)

"""

# Function for predict stress value

def stress_predict(well, epsilon_min, epsilon_max, depth):
    """
    This function is for stress value prediction of mising eqaution inputs
    well = well logging data in pandas data frame with alias applied.
    epsilon_min = minimum tectonic strain
    epsilon_max = maximum tectonic strain
    depth = depth point of core data
    """
    # prepare input parameters

    cols = ['OBP', 'HYDRO', 'YME', 'PR']
    n_data = int(round(len(well[cols].dropna().index) * 0.1))
    data = well[cols].dropna().head(n_data)
    
    OBP = data.OBP # overburden pressure
    PP = data.HYDRO # pore pressure
    YME = data.YME # young's modulus
    PR = data.PR # poisson's ratio
    alpha = 1 # Biot's constant

    # calculate stress with zero maximum and maximum tectonic strains

    data['stress'] = hstress_eq(OBP, PP, YME, PR, alpha, epsilon_min, epsilon_max)

    # linear regression for core data depth point

    X_data = data.index.values.reshape(-1,1) # reshape for model prediction
    Y_data = data['stress'].values.reshape(-1,1) # reshape for model prediction

    Y_pred = np.array([depth]).reshape((-1, 1))

    lr = LinearRegression()
    lr.fit(X_data, Y_data)

    lr_r = lr.score(X_data, Y_data)
    stress = lr.predict(Y_pred)

    print('Now, predicted minimum stress at core depth is %f with R-squared value %f.' %(stress.item(0), lr_r))

    return stress.item(0)

# Function for maximum and minimum tectonic strain (epsilon_max, epsilon_min) prediction using iteration process.

def tect_strain(well, drill):
    """
    This function is for maximum and minimum tectonic strain (epsilon_max, epsilon_min) prediction using iteration process.
    well = well logging data in pandas data frame with alias applied.
    drill = drilling test data such FIT or LOT
    """
    # iterative parameters

    epsilon_min = 0 # minimum tectonic strain
    epsilon_max = 0 # maximum tectonic strain

    lr = 0.0001 # increment rate
    
    # prepare input parameters

    drill['PRESSURE'] = (drill.RESULT * 0.052) * (drill.TVD * 3.28084)  # 0.052 for ppg to psi/ft, 3.28084 for m to ft

    for row in drill.iterrows():

        depth = row[1].MD
        tvd = row[1].TVD
        Shmin = row[1].PRESSURE
        test = row[1].TEST

        OBP = well.OBP.loc[well.index == depth].values[0]
        PP = well.HYDRO.loc[well.index == depth].values[0]
        YME = well.YME.loc[well.index == depth].values[0]
        PR = well.PR.loc[well.index == depth].values[0]
        alpha = 1 # Biot's constant

        print('Minimum horizontal stress from %s at core depth %f (TVD) is %f (psi).' %(test, tvd, Shmin))

        # leak off or fracture pressure

        if row[1].TEST == 'FIT':

            stress = hstress_eq(OBP, PP, YME, PR, alpha, epsilon_min, epsilon_max)

            if np.isnan(stress): 

               # in case of no iput data available 
                
                stress = stress_predict(well, epsilon_min, epsilon_max, depth)                

                while (stress <= Shmin):
                    epsilon_min += lr
                    epsilon_max += lr
                    stress = stress_predict(well, epsilon_min, epsilon_max, depth)

                epsilon_min -= lr
                epsilon_max -= lr
                stress = stress_predict(well, epsilon_min, epsilon_max, depth)

                while (stress <= Shmin):
                    epsilon_max += lr
                    stress = stress_predict(well, epsilon_min, epsilon_max, depth)

            else:
                while (stress <= Shmin):
                    epsilon_min += lr
                    epsilon_max += lr
                    stress = hstress_eq(OBP, PP, YME, PR, alpha, epsilon_min, epsilon_max)
                    print('Now, calculated minimum stress at core depth is %f.' %stress)
            
                epsilon_min -= lr
                epsilon_max -= lr
                stress = hstress_eq(OBP, PP, YME, PR, alpha, epsilon_min, epsilon_max)
                print('Now, calculated minimum stress at core depth is %f.' %stress)

                while (stress <= Shmin):
                    epsilon_max += lr
                    stress = hstress_eq(OBP, PP, YME, PR, alpha, epsilon_min, epsilon_max)
                    print('Now, calculated minimum stress at core depth is %f.' %stress)
        
        else:

            stress = hstress_eq(OBP, PP, YME, PR, alpha, epsilon_min, epsilon_max)

            if np.isnan(stress):

                # in case of no iput data available 
                
                stress = stress_predict(well, epsilon_min, epsilon_max, depth)                

                while (stress < Shmin):
                    epsilon_min += lr
                    epsilon_max += lr
                    stress = stress_predict(well, epsilon_min, epsilon_max, depth)

                epsilon_min -= lr
                epsilon_max -= lr
                stress = stress_predict(well, epsilon_min, epsilon_max, depth)

                while (stress < Shmin):
                    epsilon_max += lr
                    stress = stress_predict(well, epsilon_min, epsilon_max, depth)

            else:
                while (stress <= Shmin):
                    epsilon_min += lr
                    epsilon_max += lr
                    stress = hstress_eq(OBP, PP, YME, PR, alpha, epsilon_min, epsilon_max)
                    print('Now, calculated minimum stress at core depth is %f.' %stress)
            
                epsilon_min -= lr
                epsilon_max -= lr
                stress = hstress_eq(OBP, PP, YME, PR, alpha, epsilon_min, epsilon_max)
                print('Now, calculated minimum stress at core depth is %f.' %stress)

                while (stress <= Shmin):
                    epsilon_max += lr
                    stress = hstress_eq(OBP, PP, YME, PR, alpha, epsilon_min, epsilon_max)
                    print('Now, calculated minimum stress at core depth is %f.' %stress)

    if epsilon_min < 0:
        epsilon_min = 0

    if epsilon_max < 0:
        epsilon_max = 0
    
    return epsilon_max, epsilon_min

# Function for maximum and minimum horizontal stresses calculation

def stresses_cal(las, well, drill):
    """
    This function is for maximum and minimum horizontal stresses calculation.
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    drill = drilling test data such FIT or LOT
    """
    # prepare input parameters

    OBP = well.OBP # overburden pressure
    PP = well.HYDRO # pore pressure
    YME = well.YME # young's modulus
    PR = well.PR # poisson's ratio
    alpha = 1 # Biot's constant

    # calculate maximum and minimum tectonic strain

    epsilon_max, epsilon_min = tect_strain(well, drill)

    # calculate minimum and maximum horizontal stresses

    well['Shmin'] = hstress_eq(OBP, PP, YME, PR, alpha, epsilon_min, epsilon_max)
    well['SHmax'] = hstress_eq(OBP, PP, YME, PR, alpha, epsilon_max, epsilon_min)

    print('Maximum and minimum horizontal stresses are calculated with maximum tectonic strain %.4f and minimum tectonic strain %.4f.' %(epsilon_max, epsilon_min))

    # update LAS file

    las.append_curve('SHmax', well['SHmax'], unit = 'psi', descr = 'Maximum horizontal stress', value = '')
    las.append_curve('Shmin', well['Shmin'], unit = 'psi', descr = 'Minimum horizontal stress', value = '')

    return las, well

# Function for horizontal stress calculation

def hstress_eq(OBP, PP, YME, PR, alpha, epsilon1, epsilon2):
    """
    This function is horizontal stress calculation equation.
    OBP = overburden pressure or vertical stress
    PP = Pore pressure or hydrostatic pressure
    YME = Young's modulus
    PR = Poisson's ratio
    alpha = Biot's constant (Normally, equal to 1)
    epsilon1 = minimum tectonic strain for minimum horizontal stress calculation
               / maximum tectonic strain for maximum horizontal stress calculation
    epsilon2 = maximum tectonic strain for minimum horizontal stress calculation 
               / minimum tectonic strain for maximum horizontal stress calculation
    """
    tectonic = (YME / (1 - (PR**2))) * (epsilon1 + (PR * epsilon2))
    stress = ((PR / (1 - PR)) * (OBP - (alpha * PP))) + (alpha * PP) + tectonic

    return stress

# calculate maximum and minimum horizontal stresses

for las, well, drill in zip(lases, wells, tvd_drills):
    
    print('Well %s is being implemented.' %las.well['WELL'].value)
    
    las, well = stresses_cal(las, well, drill)

# Function for all stresses plot

def stresses_plot(las, well, tvd_top, tvd_mud, all_forms, tvd_press, well_names, tvd_drill, stress_name):
    """
    This function is plotting all principal stresses.
    las = las file (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    tvd_top = formation top data in pandas data frame.
    tvd_mud = tvd_mud weight log in pandas data frame.
    all_forms = list of all formation names with color code in dictionary format
    tvd_press = lsit of pressure test in panda data frame
    well_names = list of well name and its color identity
    tvd_drill = drilling test data such FIT or LOT
    stress_name = name of saved figure.
    """
    # Create figure and subplots
    
    fig, axis = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 12), sharey = True)
    fig.suptitle('All principal stresses of Well %s' %las.well['WELL'].value, fontsize = 20, y = 0.98)
    
    # General setting for all axis

    top_depth = 0
    bottom_depth = well.TVD.max()
    
    # Plot formations

    axis.set_ylim(top_depth, bottom_depth)
    axis.set_ylabel('TVD[m]')
    axis.invert_yaxis()
    axis.minorticks_on() #Scale axis
    axis.get_xaxis().set_visible(False)
    axis.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
    axis.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')

    for top, bottom, form in zip(tvd_top.TVD_TOP, tvd_top.TVD_BOTTOM, tvd_top.FORMATIONS):
        if (top >= top_depth) & (top <= bottom_depth):
            axis.axhline(y = top, linewidth = 1.5, color = all_forms[form], alpha = 0.8)

            if (bottom <= bottom_depth):
                middle = top + (bottom - top)/2
                axis.axhspan(top, bottom, facecolor = all_forms[form], alpha = 0.2)
            
            else:
                middle = top + (bottom_depth - top)/2
                axis.axhspan(top, bottom_depth, facecolor = all_forms[form], alpha = 0.2)
                
            axis.text(0.01, middle , form, ha = 'left', va = 'center', color = all_forms[form], 
                        path_effects = [pe.withStroke(linewidth = 3, foreground = "white")], fontsize = 10, weight = 'bold')

    # Pressures

    max_pressure_test = max([pres.GAUGE.max() for pres in tvd_press])
    max_scale = max([well.OBP.max(), well.HYDRO.max(), well.SHmax.max(), well.Shmin.max(), tvd_mud.PRESSURE.max(), max_pressure_test])

    min_pressure_test = min([pres.GAUGE.min() for pres in tvd_press])
    min_scale = min([well.OBP.min(), well.HYDRO.min(), well.SHmax.min(), well.Shmin.min(), tvd_mud.PRESSURE.min(), min_pressure_test])

    scale, scalelist = plot_scale(max_scale, min_scale, 1000)

    ax11 = axis.twiny()
    ax11.plot(well.SHmax, well.TVD, color = 'blue', linewidth = '0.5')
    ax11.spines['top'].set_position(('axes', 1.02))
    ax11.spines['top'].set_edgecolor('blue')
    ax11.set_xlim(scale[0], scale[-1])
    ax11.set_xlabel('SHmax[psi]', color = 'blue')    
    ax11.tick_params(axis = 'x', colors = 'blue')
    ax11.set_xticks(scale)
    ax11.set_xticklabels(scalelist)

    ax11.grid(True)
    
    ax12 = axis.twiny()
    ax12.plot(well.Shmin, well.TVD, color = 'lime', linewidth = '0.5')
    ax12.spines['top'].set_position(('axes', 1.08))
    ax12.spines['top'].set_edgecolor('lime')
    ax12.set_xlim(scale[0], scale[-1])
    ax12.set_xlabel('Shmin[psi]', color = 'lime')    
    ax12.tick_params(axis = 'x', colors = 'lime')
    ax12.set_xticks(scale)
    ax12.set_xticklabels(scalelist)

    ax13 = axis.twiny()
    ax13.plot(well.OBP, well.TVD, color = 'black', linewidth = '1')
    ax13.spines['top'].set_position(('axes', 1.14))
    ax13.spines['top'].set_edgecolor('black')
    ax13.set_xlim(scale[0], scale[-1])
    ax13.set_xlabel('OBP[psi]', color = 'black')    
    ax13.tick_params(axis = 'x', colors = 'black')
    ax13.set_xticks(scale)
    ax13.set_xticklabels(scalelist)

    ax14 = axis.twiny()
    ax14.plot(well.HYDRO, well.TVD, color = 'deepskyblue', linewidth = '1')
    ax14.spines['top'].set_position(('axes', 1.20))
    ax14.spines['top'].set_edgecolor('deepskyblue')   
    ax14.set_xlim(scale[0], scale[-1])
    ax14.set_xlabel('HYDRO[psi]', color = 'deepskyblue')    
    ax14.tick_params(axis = 'x', colors = 'deepskyblue')
    ax14.set_xticks(scale)
    ax14.set_xticklabels(scalelist)

    ax15 = axis.twiny()
    ax15.plot(tvd_mud.PRESSURE, tvd_mud.TVD, color = 'green', linewidth = '1')
    ax15.spines['top'].set_position(('axes', 1.26))
    ax15.spines['top'].set_edgecolor('green')   
    ax15.set_xlim(scale[0], scale[-1])
    ax15.set_xlabel('MW[psi]', color = 'green')    
    ax15.tick_params(axis = 'x', colors = 'green')
    ax15.set_xticks(scale)
    ax15.set_xticklabels(scalelist)

    # pressure test and drilling test data plots

    tvd_drill['PRESSURE'] = (tvd_drill.RESULT * 0.052) * (tvd_drill.TVD * 3.28084)  # 0.052 for ppg to psi/ft, 3.28084 for m to ft

    ax16 = axis.twiny() 

    for pres, name in zip(tvd_press, well_names):
        ax16.scatter(pres.GAUGE, pres.TVD, c = well_names[name], alpha = 0.5, marker = 'o', label = name)
    
    ax16.scatter(tvd_drill.PRESSURE, tvd_drill.TVD, c = 'red', alpha = 1, marker = 'X', label = tvd_drill.TEST.values[0])
    ax16.get_xaxis().set_visible(False) 
    ax16.set_xlim(scale[0], scale[-1])
    ax16.legend(loc = 'upper right')

    fig.tight_layout()

    # Save files

    stress_folder = 'Principal Stresses'
    stress_path = os.path.join(sav_path, stress_folder)

    if not os.path.isdir(stress_path):
        os.makedirs(stress_path)

    plt.savefig(os.path.join(stress_path, stress_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

for las, well, tvd_top, tvd_mud, tvd_drill in zip(lases, wells, tvd_tops, tvd_muds, tvd_drills):
    stress_name = 'LQC_%s_stresses.png' %las.well['WELL'].value
    stresses_plot(las, well, tvd_top, tvd_mud, all_forms, tvd_press, well_names, tvd_drill, stress_name)

"""

10.Failure Analysis

"""

# Function for transfering the mud weight data to well logging data

def merge_mud(las, well, mud):
    """
    This function is merging mud weight data (MUDWEIGHT) to well logging data for another computation step.
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    mud = mud weight log in pandas data frame.
    """
    # merge deviation file

    df = pd.DataFrame().reindex(well.index)
    df['MW'] = mud.set_index('MD').MUDWEIGHT
    df.interpolate(method = 'linear', limit_area = 'inside', inplace = True)

    well['MW'] = df.MW
    well['PM'] = (well.MW * 0.052) * (well.TVD * 3.28084)  # 0.052 for ppg to psi/ft, 3.28084 for m to ft

    # update LAS file

    las.append_curve('MW', well['MW'], unit = 'ppg', descr = 'mud weight', value = '')
    las.append_curve('PM', well['PM'], unit = 'psi', descr = 'mud pressure', value = '')

    return las, well

# Function for hoop stress calculation

def tensile_failure(las, well):
    """
    This function is calculating minimum hoop stress for tensile failure checking.
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    """
    # prepare input parameters

    SHmax = well.SHmax # maximum horizontal stress
    Shmin = well.Shmin # minimum horizontal stress
    pp = well.HYDRO # pore pressure
    pm = well.PM # mud weight in psi
    TVD = well.TVD # true vetical depth

    # minimum hoop stress calculation for tensile failure

    well['TSF'] = (3 * Shmin) - SHmax - pp - pm

    print('Minimum hoop stress is calculated.')

    # update LAS file

    las.append_curve('TSF', well['TSF'], unit = 'psi', descr = 'Minimum hoop stress for tensile failure', value = '')

    return las, well

# Function for lower and upper mud weight boundary

def mud_window(las, well):
    """
    This function is for mud window components calculation.
    Including: 
        1. Kick mud weight
        2. Breakout mud weight or minimum mud weight
        3. Loss mud weight
        4. Breakdown mud weight or maximum mud weight
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    """
    # prepare input parameters

    SHmax = well.SHmax # maximum horizontal stress
    Shmin = well.Shmin # minimum horizontal stress
    pp = well.HYDRO # pore pressure
    UCS = well.UCS # unconfined compressive strength
    TSTR = well.TSTR # tensile strength
    FANG = well.FANG # friction angle
    TVD = well.TVD # true vetical depth

    # Kick mud weight calculation

    well['CMW_KICK'] = well.WPG

    # Breakout or minimum mud weight caluculation

    min_mw = (3 * SHmax) - Shmin - pp - UCS

    well['CMW_MIN_MC'] = (min_mw/TVD) * (1/3.28084) * (1/0.052) # 3.28084 for m to ft, 1/0.052 for psi/ft to ppg

    # Loss mud weight calculation

    well['CMW_LOSS'] = (Shmin/TVD) * (1/3.28084) * (1/0.052) # 3.28084 for m to ft, 1/0.052 for psi/ft to ppg

    # Breakdown or maximum mud weight caluculation

    max_mw = (3 * Shmin) - SHmax - pp + TSTR

    well['CMW_MAX_MTS'] = (max_mw/TVD) * (1/3.28084) * (1/0.052) # 3.28084 for m to ft, 1/0.052 for psi/ft to ppg

    print('Mud-window components for kick, breakout, loss and breakdown are calculated.')

    # update LAS file

    las.append_curve('CMW_KICK', well['CMW_KICK'], unit = 'ppg', descr = 'Calculated kick mud weight', value = '')
    las.append_curve('CMW_MIN_MC', well['CMW_MIN_MC'], unit = 'ppg', descr = 'Calculated breakout or minimum mud weight', value = '')
    las.append_curve('CMW_LOSS', well['CMW_LOSS'], unit = 'ppg', descr = 'Calculated loss mud weight', value = '')
    las.append_curve('CMW_MAX_MTS', well['CMW_MAX_MTS'], unit = 'ppg', descr = 'Calculated breakdown or maximum mud weight', value = '')

    return las, well

# Function for break out width calculation

def breakout_width(las, well):
    """
    This function is for minimum and maximum mud window calculation.
    las = las files (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    """
    # prepare input parameters

    SHmax = well.SHmax # maximum horizontal stress
    Shmin = well.Shmin # minimum horizontal stress
    pp = well.HYDRO # pore pressure
    UCS = well.UCS # unconfined compressive strength
    pm = well.PM # mud weight in psi
    TVD = well.TVD # true vetical depth

    # breakout width caluculation

    term1 = SHmax + Shmin - pm - pp - UCS
    term2 = 2 * (SHmax - Shmin)

    cos_2 = term1/term2

    well['WBO'] = np.rad2deg(np.pi - np.arccos(cos_2))

    print('Breakout width is calculated.')

    # update LAS file

    las.append_curve('WBO', well['WBO'], unit = 'degree', descr = 'Breakout width', value = '')

    return las, well

# calculate hoop stress, mud window and breakout width

for las, well, mud in zip(lases, wells, tvd_muds):
    
    print('Well %s is being implemented.' %las.well['WELL'].value)

    las, well = merge_mud(las, well, mud)
    las, well = tensile_failure(las, well)
    las, well = mud_window(las, well)
    las, well = breakout_width(las, well)

# Function for 1D MEM plot

def mem_plot(las, well, tvd_top, tvd_pres, tvd_core, data_range, all_forms, mem_name):
    """
    This function is constructing 1D mechanical earth model using the data from previous step.
    las = las file (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    tvd_top = formation top data in pandas data frame.
    tvd_pres = pressure test in panda data frame
    tvd_core = core test in panda data frame
    data_range = recorded data range
    all_forms = list of all formation names with color code in dictionary format
    mem_name = name of saved figure.
    """
    # Create figure

    fig = plt.figure(figsize=(30, 15))
    fig.suptitle('1D MEM of well %s' %las.well['WELL'].value, fontsize = 20, y = 1.07)

    w_ratio = [1] + [2 for i in range(9)] + [4, 2, 6] + [2 for i in range(4)]

    gs = gridspec.GridSpec(ncols = 17, nrows = 1, width_ratios = w_ratio)

    axis0 = fig.add_subplot(gs[0])
    axis = [axis0] + [fig.add_subplot(gs[i+1], sharey = axis0) for i in range(16)]

    fig.subplots_adjust(wspace = 0.05)

    # General setting for all axis

    axis[0].set_ylabel('TVDSS[m]')

    start = data_range[0]
    stop = data_range[1]

    condition = (well.index >= start) & (well.index <= stop)

    top_depth = tvd_top.loc[tvd_top.FORMATIONS == 'MS', 'TVDSS_TOP'].values
    bottom_depth = well.loc[condition, 'TVDSS'].dropna().max()
    
    for ax in axis:
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False) 
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')

    for ax in axis[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    # formations plot

    ax11 = axis[0].twiny()
    ax11.set_xlim(0, 1)
    ax11.set_xticks([0, 1])
    ax11.set_xticklabels(['', ''])

    for top, bottom, form in zip(tvd_top.TVDSS_TOP, tvd_top.TVDSS_BOTTOM, tvd_top.FORMATIONS):
        if (top >= top_depth) & (top <= bottom_depth):
            ax11.axhline(y = top, linewidth = 1.5, color = all_forms[form], alpha = 0.5)

            if (bottom <= bottom_depth):
                middle = top + (bottom - top)/2
                ax11.axhspan(top, bottom, facecolor = all_forms[form], alpha = 0.2)
                
            else:
                middle = top + (bottom_depth - top)/2
                ax11.axhspan(top, bottom_depth, facecolor = all_forms[form], alpha = 0.2)

            ax11.text(0.5, middle, form, ha = 'center', va = 'center', color = all_forms[form],
                        path_effects = [pe.withStroke(linewidth = 3, foreground = "white")], fontsize = 10, weight = 'bold')

    ax11.grid(False)

    # formation plots for other columns

    for ax in axis[1:-2]:
        for top, bottom, form in zip(tvd_top.TVDSS_TOP, tvd_top.TVDSS_BOTTOM, tvd_top.FORMATIONS):
            if (top >= top_depth) & (top <= bottom_depth):
                ax.axhline(y = top, linewidth = 1.5, color = all_forms[form], alpha = 0.5)
                ax.axhspan(top, bottom, facecolor = all_forms[form], alpha = 0.2)

    # Azimuth and angle plots
    
    ax21 = axis[1].twiny()
    ax21.plot(well.AZIMUTH, well.TVDSS, color = 'blue', linewidth = '0.8')
    ax21.spines['top'].set_position(('axes', 1.02))
    ax21.spines['top'].set_edgecolor('blue')
    ax21.set_xlim(0, 360)
    ax21.set_xlabel('AZIMUTH[degree]', color = 'blue')    
    ax21.tick_params(axis = 'x', colors = 'blue')
    ax21.set_xticks(np.arange(0, 361, 90))
    ax21.set_xticklabels(['0', '', '180', '', '360'])
    xtick_loc(ax21)
    
    ax21.grid(True)

    ax22 = axis[1].twiny()
    ax22.plot(well.ANGLE, well.TVDSS, color = 'red', linewidth = '0.8')
    ax22.spines['top'].set_position(('axes', 1.06))
    ax22.spines['top'].set_edgecolor('red')   
    ax22.set_xlim(0, 90)
    ax22.set_xlabel('ANGLE[degree]', color = 'red')    
    ax22.tick_params(axis = 'x', colors = 'red')
    ax22.set_xticks(np.arange(0, 91, 45))
    ax22.set_xticklabels(['0', '45', '90'])
    xtick_loc(ax22)

    # Gamma ray plot
    
    ax31 = axis[2].twiny()
    ax31.plot(well.GR_NORM, well.TVDSS, color = 'green', linewidth = '0.5')
    ax31.spines['top'].set_position(('axes', 1.02))
    ax31.spines['top'].set_edgecolor('green') 
    ax31.set_xlim(0, 150)
    ax31.set_xlabel('GR[API]', color = 'green')    
    ax31.tick_params(axis = 'x', colors = 'green')
    ax31.set_xticks(np.arange(0, 151, 30))
    ax31.set_xticklabels(['0', '', '', '', '','150'])
    xtick_loc(ax31)
    
    ax31.grid(True)
    
    # Resisitivity plots
    
    ax41 = axis[3].twiny()
    ax41.set_xscale('log')
    ax41.plot(well.RT, well.TVDSS, color = 'red', linewidth = '0.5')
    ax41.spines['top'].set_position(('axes', 1.02))
    ax41.spines['top'].set_edgecolor('red')
    ax41.set_xlim(0.2, 2000)
    ax41.set_xlabel('RT[ohm-m]', color = 'red')
    ax41.tick_params(axis = 'x', colors = 'red')
    ax41.set_xticks([0.2, 2, 20, 200, 2000])
    ax41.set_xticklabels(['0.2', '', '', '', '2000'])
    xtick_loc(ax41)
    
    ax41.grid(True)

    ax42 = axis[3].twiny()
    ax42.set_xscale('log')
    ax42.plot(well.MSFL, well.TVDSS, color = 'black', linewidth = '0.5')
    ax42.spines['top'].set_position(('axes', 1.06))
    ax42.spines['top'].set_edgecolor('black')
    ax42.set_xlim(0.2, 2000)
    ax42.set_xlabel('MSFL[ohm-m]', color = 'black')
    ax42.tick_params(axis = 'x', colors = 'black')
    ax42.set_xticks([0.2, 2, 20, 200, 2000])
    ax42.set_xticklabels(['0.2', '', '', '', '2000'])
    xtick_loc(ax42)
    
    # Density and neutron porosity plots
    
    ax51 = axis[4].twiny()
    ax51.plot(well.RHOB_MRG, well.TVDSS, color = 'red', linewidth = '0.5')
    ax51.spines['top'].set_position(('axes', 1.02))
    ax51.spines['top'].set_edgecolor('red')
    ax51.set_xlim(1.95, 2.95)
    ax51.set_xlabel('RHOB_MRG[g/c3]', color = 'red')    
    ax51.tick_params(axis = 'x', colors = 'red')
    ax51.set_xticks(np.arange(1.95, 2.96, 0.2))
    ax51.set_xticklabels(['1.95', '', '', '', '', '2.95'])
    xtick_loc(ax51)
    
    ax51.grid(True)

    ax52 = axis[4].twiny()
    ax52.plot(well.NPHI_MRG, well.TVDSS, color = 'blue', linewidth = '0.5')
    ax52.spines['top'].set_position(('axes', 1.06))
    ax52.spines['top'].set_edgecolor('blue')   
    ax52.set_xlim(0.45, -0.15)
    ax52.set_xlabel('NPHI_MRG[V/V]', color = 'blue')    
    ax52.tick_params(axis = 'x', colors = 'blue')
    ax52.set_xticks(np.arange(0.45, -0.16, -0.12))
    ax52.set_xticklabels(['0.45', '', '', '', '', '-0.15'])
    xtick_loc(ax52)
    
    # P_Sonic and S_Sonic plots
    
    ax61 = axis[5].twiny()
    ax61.plot(well.DTC_MRG, well.TVDSS, color = 'blue', linewidth = '0.5')
    ax61.spines['top'].set_position(('axes', 1.02))
    ax61.spines['top'].set_edgecolor('blue')
    ax61.set_xlim(140, 40)
    ax61.set_xlabel('DTC_MRG[us/ft]', color = 'blue')    
    ax61.tick_params(axis = 'x', colors = 'blue')
    ax61.set_xticks(np.arange(140, 39, -20))
    ax61.set_xticklabels(['140', '', '', '', '', '40'])
    xtick_loc(ax61)

    ax61.grid(True)

    ax62 = axis[5].twiny()
    ax62.plot(well.DTS_MRG, well.TVDSS, color = 'red', linewidth = '0.5')
    ax62.spines['top'].set_position(('axes', 1.06))
    ax62.spines['top'].set_edgecolor('red') 
    ax62.set_xlim(340, 40)
    ax62.set_xlabel('DTS_MRG[us/ft]', color = 'red')    
    ax62.tick_params(axis = 'x', colors = 'red')
    ax62.set_xticks(np.arange(340, 39, -60))
    ax62.set_xticklabels(['340', '', '', '', '', '40'])
    xtick_loc(ax62)

    # Young's modulus plot

    scale1, scalelist1 = plot_scale(well.YME.max(), 0, 10000)
    
    ax71 = axis[6].twiny()
    ax71.plot(well.YME, well.TVDSS, color = 'chocolate', linewidth = '0.5')
    ax71.spines['top'].set_position(('axes', 1.02))
    ax71.spines['top'].set_edgecolor('chocolate') 
    ax71.set_xlim(scale1[0], scale1[-1])
    ax71.set_xlabel('YME[psi]', color = 'chocolate')    
    ax71.tick_params(axis = 'x', colors = 'chocolate')
    ax71.set_xticks(scale1)
    sci_scale(ax71, scalelist1)
    xtick_loc(ax71)

    ax71.grid(True)

    ax72 = axis[6].twiny()
    ax72.scatter(tvd_core.YME, tvd_core.TVDSS, c = 'black', alpha = 1, marker = 'o')
    ax72.get_xaxis().set_visible(False) 
    ax72.set_xlim(scale1[0], scale1[-1])
    
    # Poisson's ratio plot
    
    ax81 = axis[7].twiny()
    ax81.plot(well.PR, well.TVDSS, color = 'salmon', linewidth = '0.5')
    ax81.spines['top'].set_position(('axes', 1.02))
    ax81.spines['top'].set_edgecolor('salmon') 
    ax81.set_xlim(0, 0.5)
    ax81.set_xlabel('PR[unitless]', color = 'salmon')    
    ax81.tick_params(axis = 'x', colors = 'salmon')
    ax81.set_xticks(np.arange(0, 0.51, 0.1))
    ax81.set_xticklabels(['0', '', '', '', '','0.5'])
    xtick_loc(ax81)

    ax81.grid(True)

    # unconfined compressive strength and Tensile formation strength plots

    max_scale2 = max([well.UCS.max(), well.TSTR.max()])
    scale2, scalelist2 = plot_scale(max_scale2, 0, 2000)
    
    ax91 = axis[8].twiny()
    ax91.plot(well.UCS, well.TVDSS, color = 'red', linewidth = '0.5')
    ax91.spines['top'].set_position(('axes', 1.02))
    ax91.spines['top'].set_edgecolor('red')
    ax91.set_xlim(scale2[0], scale2[-1])
    ax91.set_xlabel('UCS[psi]', color = 'red')    
    ax91.tick_params(axis = 'x', colors = 'red')
    ax91.set_xticks(scale2)
    ax91.set_xticklabels(scalelist2)
    xtick_loc(ax91)

    ax91.grid(True)

    ax92 = axis[8].twiny()
    ax92.plot(well.TSTR, well.TVDSS, color = 'darkorange', linewidth = '0.5')
    ax92.spines['top'].set_position(('axes', 1.06))
    ax92.spines['top'].set_edgecolor('darkorange') 
    ax92.set_xlim(scale2[0], scale2[-1])
    ax92.set_xlabel('TSTR[psi]', color = 'darkorange')    
    ax92.tick_params(axis = 'x', colors = 'darkorange')
    ax92.set_xticks(scale2)
    ax92.set_xticklabels(scalelist2)
    xtick_loc(ax92)

    ax93 = axis[8].twiny() 
    ax93.scatter(tvd_core.UCS, tvd_core.TVDSS, c = 'black', alpha = 1, marker = 'o')
    ax93.get_xaxis().set_visible(False) 
    ax93.set_xlim(scale2[0], scale2[-1])
    
    # angle of internal friction plot
    
    ax101 = axis[9].twiny()
    ax101.plot(well.FANG, well.TVDSS, color = 'green', linewidth = '0.5')
    ax101.spines['top'].set_position(('axes', 1.02))
    ax101.spines['top'].set_edgecolor('green') 
    ax101.set_xlim(0, 50)
    ax101.set_xlabel('FANG[degree]', color = 'green')    
    ax101.tick_params(axis = 'x', colors = 'green')
    ax101.set_xticks(np.arange(0, 51, 10))
    ax101.set_xticklabels(['0', '', '', '', '','50'])
    xtick_loc(ax101)
    
    ax101.grid(True)
    
    # principle stresses plots

    max_scale3 = max([well.OBP.max(), well.HYDRO.max(), well.SHmax.max(), well.Shmin.max(), tvd_pres.GAUGE.max(), well.PM.max()])
    scale3, scalelist3 = plot_scale(max_scale3, 0, 1000)

    ax111 = axis[10].twiny()
    ax111.plot(well.SHmax, well.TVDSS, color = 'blue', linewidth = '0.5')
    ax111.spines['top'].set_position(('axes', 1.02))
    ax111.spines['top'].set_edgecolor('blue')
    ax111.set_xlim(scale3[0], scale3[-1])
    ax111.set_xlabel('SHmax[psi]', color = 'blue')    
    ax111.tick_params(axis = 'x', colors = 'blue')
    ax111.set_xticks(scale3)
    ax111.set_xticklabels(scalelist3)
    xtick_loc(ax111)
    
    ax112 = axis[10].twiny()
    ax112.plot(well.Shmin, well.TVDSS, color = 'lime', linewidth = '0.5')
    ax112.spines['top'].set_position(('axes', 1.06))
    ax112.spines['top'].set_edgecolor('lime')
    ax112.set_xlim(scale3[0], scale3[-1])
    ax112.set_xlabel('Shmin[psi]', color = 'lime')    
    ax112.tick_params(axis = 'x', colors = 'lime')
    ax112.set_xticks(scale3)
    ax112.set_xticklabels(scalelist3)
    xtick_loc(ax112)

    ax113 = axis[10].twiny()
    ax113.plot(well.OBP, well.TVDSS, color = 'black', linewidth = '1')
    ax113.spines['top'].set_position(('axes', 1.10))
    ax113.spines['top'].set_edgecolor('black')
    ax113.set_xlim(scale3[0], scale3[-1])
    ax113.set_xlabel('OBP[psi]', color = 'black')    
    ax113.tick_params(axis = 'x', colors = 'black')
    ax113.set_xticks(scale3)
    ax113.set_xticklabels(scalelist3)
    xtick_loc(ax113)

    ax114 = axis[10].twiny()
    ax114.plot(well.HYDRO, well.TVDSS, color = 'deepskyblue', linewidth = '1')
    ax114.spines['top'].set_position(('axes', 1.14))
    ax114.spines['top'].set_edgecolor('deepskyblue')   
    ax114.set_xlim(scale3[0], scale3[-1])
    ax114.set_xlabel('HYDRO[psi]', color = 'deepskyblue')    
    ax114.tick_params(axis = 'x', colors = 'deepskyblue')
    ax114.set_xticks(scale3)
    ax114.set_xticklabels(scalelist3)
    xtick_loc(ax114)

    ax115 = axis[10].twiny()
    ax115.plot(well.PM, well.TVDSS, color = 'green', linewidth = '1', linestyle = '--')
    ax115.spines['top'].set_position(('axes', 1.18))
    ax115.spines['top'].set_edgecolor('green')   
    ax115.set_xlim(scale3[0], scale3[-1])
    ax115.set_xlabel('MW[psi]', color = 'green')    
    ax115.tick_params(axis = 'x', colors = 'green')
    ax115.set_xticks(scale3) 
    ax115.set_xticklabels(scalelist3)
    xtick_loc(ax115)

    ax116 = axis[10].twiny() 
    ax116.scatter(tvd_pres.GAUGE, tvd_pres.TVDSS, c = 'black', alpha = 1, marker = 'o')
    ax116.get_xaxis().set_visible(False) 
    ax116.set_xlim(scale3[0], scale3[-1])
    
    # minimum hoop stress and Tensile formation strength plots

    negative_TSTR = well.TSTR * (-1)

    incre = 1000

    max_scale4 = max([well.TSF.max(), negative_TSTR.max()])
    min_scale4 = min([well.TSF.min(), negative_TSTR.min()])

    scale4, scalelist4 = plot_scale(max_scale4, min_scale4, incre)

    ax121 = axis[11].twiny()
    ax121.plot(well.TSF, well.TVDSS, color = 'red', linewidth = '0.5')
    ax121.spines['top'].set_position(('axes', 1.02))
    ax121.spines['top'].set_edgecolor('red') 
    ax121.set_xlim(scale4[0], scale4[-1])
    ax121.set_xlabel('TSF[psi]', color = 'red')
    ax121.tick_params(axis = 'x', colors = 'red')
    ax121.set_xticks(scale4)
    ax121.set_xticklabels(scalelist4)
    xtick_loc(ax121)

    ax121.grid(True)

    ax122 = axis[11].twiny()
    ax122.plot(negative_TSTR, well.TVDSS, color = 'darkorange', linewidth = '0.5')
    ax122.spines['top'].set_position(('axes', 1.06))
    ax122.spines['top'].set_edgecolor('darkorange') 
    ax122.set_xlim(scale4[0], scale4[-1])
    ax122.set_xlabel('TSTR[psi]', color = 'darkorange')    
    ax122.tick_params(axis = 'x', colors = 'darkorange')
    ax122.set_xticks(scale4)
    ax122.set_xticklabels(scalelist4)
    xtick_loc(ax122)
    
    # mud window plots

    ax131 = axis[12].twiny()
    ax131.plot(well.CMW_KICK, well.TVDSS, color = 'gray', linewidth = '0.5')
    ax131.spines['top'].set_position(('axes', 1.02))
    ax131.spines['top'].set_edgecolor('gray')
    ax131.set_xlim(8, 18)
    ax131.set_xlabel('CMW_KICK[ppg]', color = 'gray')    
    ax131.tick_params(axis = 'x', colors = 'gray')
    ax131.set_xticks(np.arange(8, 19, 2))
    ax131.set_xticklabels(['8', '', '', '', '', '18'])
    xtick_loc(ax131)

    ax131.grid(True)
    
    ax132 = axis[12].twiny()
    ax132.plot(well.CMW_MIN_MC, well.TVDSS, color = 'red', linewidth = '0.5')
    ax132.spines['top'].set_position(('axes', 1.06))
    ax132.spines['top'].set_edgecolor('red')
    ax132.set_xlim(8, 18)
    ax132.set_xlabel('CMW_MIN_MC[ppg]', color = 'red')    
    ax132.tick_params(axis = 'x', colors = 'red')
    ax132.set_xticks(np.arange(8, 19, 2))
    ax132.set_xticklabels(['8', '', '', '', '', '18'])
    xtick_loc(ax132)

    ax133 = axis[12].twiny()
    ax133.plot(well.CMW_LOSS, well.TVDSS, color = 'indigo', linewidth = '0.5')
    ax133.spines['top'].set_position(('axes', 1.10))
    ax133.spines['top'].set_edgecolor('indigo')
    ax133.set_xlim(8, 18)
    ax133.set_xlabel('CMW_LOSS[ppg]', color = 'indigo')    
    ax133.tick_params(axis = 'x', colors = 'indigo')
    ax133.set_xticks(np.arange(8, 19, 2))
    ax133.set_xticklabels(['8', '', '', '', '', '18'])
    xtick_loc(ax133)

    ax134 = axis[12].twiny()
    ax134.plot(well.CMW_MAX_MTS, well.TVDSS, color = 'darkslateblue', linewidth = '0.5')
    ax134.spines['top'].set_position(('axes', 1.14))
    ax134.spines['top'].set_edgecolor('darkslateblue')   
    ax134.set_xlim(8, 18)
    ax134.set_xlabel('CMW_MAX_MTS[ppg]', color = 'darkslateblue')    
    ax134.tick_params(axis = 'x', colors = 'darkslateblue')
    ax134.set_xticks(np.arange(8, 19, 2))
    ax134.set_xticklabels(['8', '', '', '', '', '18'])
    xtick_loc(ax134)
    
    ax135 = axis[12].twiny()
    ax135.plot(well.MW, well.TVDSS, color = 'green', linewidth = '1', linestyle = '--')
    ax135.spines['top'].set_position(('axes', 1.18))
    ax135.spines['top'].set_edgecolor('green')   
    ax135.set_xlim(8, 18)
    ax135.set_xlabel('MW[ppg]', color = 'green')    
    ax135.tick_params(axis = 'x', colors = 'green')
    ax135.set_xticks(np.arange(8, 19, 2)) 
    ax135.set_xticklabels(['8', '', '', '', '', '18'])
    xtick_loc(ax135)

    loc1 = well.CMW_KICK > 8
    loc2 = well.CMW_MIN_MC > well.CMW_KICK
    loc3 = well.CMW_MAX_MTS > well.CMW_LOSS
    loc4 = 18 > well.CMW_MAX_MTS
    
    ax136 = axis[12].twiny()
    ax136.set_xlim(8, 18)
    ax136.fill_betweenx(well.TVDSS, 8, well.CMW_KICK, where = loc1, color='silver', capstyle = 'butt', linewidth = 0.5, alpha = 0.5, label = 'KICK')
    ax136.fill_betweenx(well.TVDSS, well.CMW_KICK, well.CMW_MIN_MC, where = loc2, color='yellow', capstyle = 'butt', linewidth = 0.5, alpha = 0.5, label = 'BREAKOUT')
    ax136.fill_betweenx(well.TVDSS, well.CMW_LOSS, well.CMW_MAX_MTS, where = loc3, color='slateblue', capstyle = 'butt', linewidth = 0.5, alpha = 0.5, label = 'LOSS')
    ax136.fill_betweenx(well.TVDSS, well.CMW_MAX_MTS, 18, where = loc4, color='darkslateblue', capstyle = 'butt', linewidth = 0.5, alpha = 0.5, label = 'BREAKDOWN')
    ax136.set_xticks(np.arange(8, 19, 2))
    ax136.set_xticklabels(['', '', '', '', '', ''])
    ax136.legend(loc = 'upper left')
    
    # breakout width plot

    wbo1 = well.WBO / 2
    wbo2 = (well.WBO / 2) * (-1)

    ax141 = axis[13].twiny()
    ax141.fill_betweenx(well.TVDSS, wbo2, wbo1, color = 'red', capstyle = 'butt', linewidth = 1, label = 'BAD')
    ax141.spines['top'].set_position(('axes', 1.02))
    ax141.spines['top'].set_edgecolor('red')   
    ax141.set_xlim(-90, 90)
    ax141.set_xlabel('WBO[degree]', color = 'red')    
    ax141.tick_params(axis = 'x', colors = 'red')
    ax141.set_xticks(np.arange(-90, 91, 45))
    ax141.set_xticklabels(['-90', '', '', '', '90'])
    xtick_loc(ax141)

    ax141.grid(True)

    # caliper and bitsize plots
    
    ax151 = axis[14].twiny()
    ax151.set_xlim(6, 11)
    ax151.plot(well.BS, well.TVDSS, color = 'black', linewidth = '0.5')
    ax151.spines['top'].set_position(('axes', 1.02))
    ax151.set_xlabel('BS[in]',color = 'black')
    ax151.tick_params(axis = 'x', colors = 'black')
    ax151.set_xticks(np.arange(6, 12, 1))
    ax151.set_xticklabels(['6', '', '', '', '', '11'])
    xtick_loc(ax151)

    ax151.grid(True)
    
    ax152 = axis[14].twiny()
    ax152.set_xlim(6, 11)
    ax152.plot(well.CAL, well.TVDSS, color = 'grey', linewidth = '0.5')
    ax152.spines['top'].set_position(('axes', 1.06))
    ax152.spines['top'].set_edgecolor('grey')
    ax152.set_xlabel('CAL[in]',color = 'grey')
    ax152.tick_params(axis = 'x', colors = 'grey')
    ax152.set_xticks(np.arange(6, 12, 1))
    ax152.set_xticklabels(['6', '', '', '', '', '11'])
    xtick_loc(ax152)

    loc5 = well.BS > well.CAL
    loc6 = well.CAL > well.BS

    ax153 = axis[14].twiny()
    ax153.set_xlim(6, 11)
    ax153.fill_betweenx(well.TVDSS, well.CAL, well.BS, where = loc5, color='yellow', capstyle = 'butt', linewidth = 0.5, alpha = 0.5, label = 'SWELLING')
    ax153.fill_betweenx(well.TVDSS, well.BS, well.CAL, where = loc6, color='red', capstyle = 'butt', linewidth = 0.5, alpha = 0.5, label = 'CAVING')
    ax153.set_xticks(np.arange(6, 12, 1))
    ax153.set_xticklabels(['', '', '', '', '', ''])
    # ax153.legend(loc = 'upper left')

    # effective porosity, rock matrix, volume of clay plots

    ax161 = axis[15].twiny()
    ax161.plot(well.VCL, well.TVDSS, color = 'SaddleBrown', linewidth = '0.5')
    ax161.spines['top'].set_position(('axes', 1.02))
    ax161.spines['top'].set_edgecolor('SaddleBrown')
    ax161.set_xlim(0, 1)
    ax161.set_xlabel('VCL[V/V]', color = 'SaddleBrown')    
    ax161.tick_params(axis = 'x', colors = 'SaddleBrown')
    ax161.set_xticks(np.arange(0, 1.1, 0.2))
    ax161.set_xticklabels(['0', '', '', '', '','1'])
    xtick_loc(ax161)

    ax162 = axis[15].twiny()
    ax162.plot(well.PHIE, well.TVDSS, color = 'gray', linewidth = '0.5')
    ax162.spines['top'].set_position(('axes', 1.06))
    ax162.spines['top'].set_edgecolor('gray')
    ax162.set_xlim(1, 0)
    ax162.set_xlabel('PHIE[V/V]', color = 'gray')    
    ax162.tick_params(axis = 'x', colors = 'gray')
    ax162.set_xticks(np.arange(1.0, -0.1, -0.2))
    ax162.set_xticklabels(['1', '', '', '', '','0'])
    xtick_loc(ax161)

    ax163 = axis[15].twiny()
    ax163.set_xlim(0, 1)
    ax163.fill_betweenx(well.TVDSS, 0, well.VCL, color='SaddleBrown', capstyle = 'butt', linewidth = 0.5, label = 'VCLAY')
    ax163.fill_betweenx(well.TVDSS, well.VCL, (1 - well.PHIE), color='yellow', capstyle = 'butt', linewidth = 0.5, label = 'MATRIX')
    ax163.fill_betweenx(well.TVDSS, (1 - well.PHIE), 1, color='gray', capstyle = 'butt', linewidth = 0.5, label = 'POROSITY')
    ax163.set_xticks([0, 1])
    ax163.set_xticklabels(['', ''])
    # ax163.legend(loc = 'upper left')

    ax163.grid(True)

    # plot sand-shale lithology

    well['liplot'] = np.nan
    well['liplot'].loc[well.LITHO == 'SAND'] = 1
    well['liplot'].loc[well.LITHO == 'SHALE'] = 0
        
    ax171 = axis[16].twiny()
    ax171.fill_betweenx(well.TVDSS, well.liplot, 1, color = 'SaddleBrown', capstyle = 'butt', linewidth = 0.01, label = 'SHALE')
    ax171.fill_betweenx(well.TVDSS, 0, well.liplot, color = 'yellow', capstyle = 'butt', linewidth = 0.01, label = 'SAND')
    ax171.spines['top'].set_position(('axes', 1.02))
    ax171.spines['top'].set_edgecolor('gray')
    ax171.set_xlim(0, 1)
    ax171.set_xlabel('LITHOLOGY', color = 'gray')
    ax171.tick_params(axis = 'x', colors = 'gray')
    ax171.set_xticks([0, 1])
    ax171.set_xticklabels(['', ''])
    ax171.legend(loc = 'upper left')

    well.drop(columns = ['liplot'], inplace = True)

    # Save files

    mem_folder = 'MEM'
    mem_path = os.path.join(sav_path, mem_folder)

    if not os.path.isdir(mem_path):
        os.makedirs(mem_path)

    plt.savefig(os.path.join(mem_path, mem_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# construct 1D MEM

for las, well, tvd_top, tvd_pres, tvd_core, data_range in zip(lases, wells, tvd_tops, tvd_press, tvd_cores, data_ranges):
    mem_name = '%s_MEM.png' %las.well['WELL'].value
    mem_plot(las, well, tvd_top, tvd_pres, tvd_core, data_range, all_forms, mem_name)

# Function for 1D MEM plot for specific formation

def form_mem_plot(las, well, tvd_top, tvd_pres, tvd_core, data_range, all_forms, formation, form_mem_name):
    """
    This function is constructing 1D mechanical earth model using the data from previous step.
    las = las file (.las) of the well data
    well = well logging data in pandas data frame with alias applied.
    tvd_top = formation top data in pandas data frame.
    tvd_pres = pressure test in panda data frame
    tvd_core = core test in panda data frame
    data_range = recorded data range
    all_forms = list of all formation names with color code in dictionary format
    formation = input the name of the formation where the data can be compared
    mem_name = name of saved figure.
    """
    # Create figure

    fig = plt.figure(figsize=(30, 15))
    fig.suptitle('1D MEM of well %s for formation %s' %(las.well['WELL'].value, formation), fontsize = 20, y = 1.07)

    w_ratio = [1] + [2 for i in range(9)] + [4, 2, 6] + [2 for i in range(4)]

    gs = gridspec.GridSpec(ncols = 17, nrows = 1, width_ratios = w_ratio)

    axis0 = fig.add_subplot(gs[0])
    axis = [axis0] + [fig.add_subplot(gs[i+1], sharey = axis0) for i in range(16)]

    fig.subplots_adjust(wspace = 0.05)

    # Set interval from each well for selected formation

    start = float(tvd_top.loc[tvd_top.FORMATIONS == formation, 'TOP'])
    stop = float(tvd_top.loc[tvd_top.FORMATIONS == formation, 'BOTTOM'])

    condition = (well.index >= start) & (well.index <= stop)

    top_depth = well.loc[condition, 'TVDSS'].dropna().min()
    bottom_depth = well.loc[condition, 'TVDSS'].dropna().max()

    # General setting for all axis

    axis[0].set_ylabel('TVDSS[m]')
    
    for ax in axis:
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.get_xaxis().set_visible(False) 
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'blue')

    for ax in axis[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    # formations plot

    ax11 = axis[0].twiny()
    ax11.set_xlim(0, 1)
    ax11.set_xticks([0, 1])
    ax11.set_xticklabels(['', ''])
    ax11.axhline(y = top_depth, linewidth = 1.5, color = all_forms[formation], alpha = 0.3)
    ax11.axhspan(top_depth, bottom_depth, facecolor = all_forms[formation], alpha = 0.1)

    middle_depth = top_depth + (bottom_depth - top_depth)/2

    ax11.text(0.5, middle_depth, formation, ha = 'center', va = 'center', color = all_forms[formation],
                path_effects = [pe.withStroke(linewidth = 3, foreground = "white")], fontsize = 10, weight = 'bold')
    
    ax11.grid(False)

    # formation plots for other columns

    for ax in axis[1:-2]:
        ax.axhline(y = top_depth, linewidth = 1.5, color = all_forms[formation], alpha = 0.3)
        ax.axhspan(top_depth, bottom_depth, facecolor = all_forms[formation], alpha = 0.1)

    # Azimuth and angle plots

    plot21 = well[['AZIMUTH', 'TVDSS']].dropna()
    
    ax21 = axis[1].twiny()
    ax21.plot(plot21.AZIMUTH, plot21.TVDSS, color = 'blue', linewidth = '1')
    ax21.spines['top'].set_position(('axes', 1.02))
    ax21.spines['top'].set_edgecolor('blue')
    ax21.set_xlim(0, 360)
    ax21.set_xlabel('AZIMUTH[degree]', color = 'blue')    
    ax21.tick_params(axis = 'x', colors = 'blue')
    ax21.set_xticks(np.arange(0, 361, 90))
    ax21.set_xticklabels(['0', '', '180', '', '360'])
    xtick_loc(ax21)
    
    ax21.grid(True)

    plot22 = well[['ANGLE', 'TVDSS']].dropna()

    ax22 = axis[1].twiny()
    ax22.plot(plot22.ANGLE, plot22.TVDSS, color = 'red', linewidth = '1')
    ax22.spines['top'].set_position(('axes', 1.06))
    ax22.spines['top'].set_edgecolor('red')   
    ax22.set_xlim(0, 90)
    ax22.set_xlabel('ANGLE[degree]', color = 'red')    
    ax22.tick_params(axis = 'x', colors = 'red')
    ax22.set_xticks(np.arange(0, 91, 45))
    ax22.set_xticklabels(['0', '45', '90'])
    xtick_loc(ax22)

    # Gamma ray plot

    plot31 = well[['GR_NORM', 'TVDSS']].dropna()
    
    ax31 = axis[2].twiny()
    ax31.plot(plot31.GR_NORM, plot31.TVDSS, color = 'green', linewidth = '1')
    ax31.spines['top'].set_position(('axes', 1.02))
    ax31.spines['top'].set_edgecolor('green') 
    ax31.set_xlim(0, 150)
    ax31.set_xlabel('GR[API]', color = 'green')    
    ax31.tick_params(axis = 'x', colors = 'green')
    ax31.set_xticks(np.arange(0, 151, 30))
    ax31.set_xticklabels(['0', '', '', '', '','150'])
    xtick_loc(ax31)
    
    ax31.grid(True)
    
    # Resisitivity plots

    plot41 = well[['RT', 'TVDSS']].dropna()
    
    ax41 = axis[3].twiny()
    ax41.set_xscale('log')
    ax41.plot(plot41.RT, plot41.TVDSS, color = 'red', linewidth = '1')
    ax41.spines['top'].set_position(('axes', 1.02))
    ax41.spines['top'].set_edgecolor('red')
    ax41.set_xlim(0.2, 2000)
    ax41.set_xlabel('RT[ohm-m]', color = 'red')
    ax41.tick_params(axis = 'x', colors = 'red')
    ax41.set_xticks([0.2, 2, 20, 200, 2000])
    ax41.set_xticklabels(['0.2', '', '', '', '2000'])
    xtick_loc(ax41)
    
    ax41.grid(True)

    plot42 = well[['MSFL', 'TVDSS']].dropna()

    ax42 = axis[3].twiny()
    ax42.set_xscale('log')
    ax42.plot(plot42.MSFL, plot42.TVDSS, color = 'black', linewidth = '1')
    ax42.spines['top'].set_position(('axes', 1.06))
    ax42.spines['top'].set_edgecolor('black')
    ax42.set_xlim(0.2, 2000)
    ax42.set_xlabel('MSFL[ohm-m]', color = 'black')
    ax42.tick_params(axis = 'x', colors = 'black')
    ax42.set_xticks([0.2, 2, 20, 200, 2000])
    ax42.set_xticklabels(['0.2', '', '', '', '2000'])
    xtick_loc(ax42)
    
    # Density and neutron porosity plots

    plot51 = well[['RHOB_MRG', 'TVDSS']].dropna()
    
    ax51 = axis[4].twiny()
    ax51.plot(plot51.RHOB_MRG, plot51.TVDSS, color = 'red', linewidth = '1')
    ax51.spines['top'].set_position(('axes', 1.02))
    ax51.spines['top'].set_edgecolor('red')
    ax51.set_xlim(1.95, 2.95)
    ax51.set_xlabel('RHOB_MRG[g/c3]', color = 'red')    
    ax51.tick_params(axis = 'x', colors = 'red')
    ax51.set_xticks(np.arange(1.95, 2.96, 0.2))
    ax51.set_xticklabels(['1.95', '', '', '', '', '2.95'])
    xtick_loc(ax51)
    
    ax51.grid(True)

    plot52 = well[['NPHI_MRG', 'TVDSS']].dropna()

    ax52 = axis[4].twiny()
    ax52.plot(plot52.NPHI_MRG, plot52.TVDSS, color = 'blue', linewidth = '1')
    ax52.spines['top'].set_position(('axes', 1.06))
    ax52.spines['top'].set_edgecolor('blue')   
    ax52.set_xlim(0.45, -0.15)
    ax52.set_xlabel('NPHI_MRG[V/V]', color = 'blue')    
    ax52.tick_params(axis = 'x', colors = 'blue')
    ax52.set_xticks(np.arange(0.45, -0.16, -0.12))
    ax52.set_xticklabels(['0.45', '', '', '', '', '-0.15'])
    xtick_loc(ax52)
    
    # P_Sonic and S_Sonic plots

    plot61 = well[['DTC_MRG', 'TVDSS']].dropna()
    
    ax61 = axis[5].twiny()
    ax61.plot(plot61.DTC_MRG, plot61.TVDSS, color = 'blue', linewidth = '1')
    ax61.spines['top'].set_position(('axes', 1.02))
    ax61.spines['top'].set_edgecolor('blue')
    ax61.set_xlim(140, 40)
    ax61.set_xlabel('DTC_MRG[us/ft]', color = 'blue')    
    ax61.tick_params(axis = 'x', colors = 'blue')
    ax61.set_xticks(np.arange(140, 39, -20))
    ax61.set_xticklabels(['140', '', '', '', '', '40'])
    xtick_loc(ax61)

    ax61.grid(True)

    plot62 = well[['DTS_MRG', 'TVDSS']].dropna()

    ax62 = axis[5].twiny()
    ax62.plot(plot62.DTS_MRG, plot62.TVDSS, color = 'red', linewidth = '1')
    ax62.spines['top'].set_position(('axes', 1.06))
    ax62.spines['top'].set_edgecolor('red') 
    ax62.set_xlim(340, 40)
    ax62.set_xlabel('DTS_MRG[us/ft]', color = 'red')    
    ax62.tick_params(axis = 'x', colors = 'red')
    ax62.set_xticks(np.arange(340, 39, -60))
    ax62.set_xticklabels(['340', '', '', '', '', '40'])
    xtick_loc(ax62)

    # Young's modulus plot

    scale1, scalelist1 = plot_scale(well.YME.max(), 0, 100000)

    plot71 = well[['YME', 'TVDSS']].dropna()
    
    ax71 = axis[6].twiny()
    ax71.plot(plot71.YME, plot71.TVDSS, color = 'chocolate', linewidth = '1')
    ax71.spines['top'].set_position(('axes', 1.02))
    ax71.spines['top'].set_edgecolor('chocolate') 
    ax71.set_xlim(scale1[0], scale1[-1])
    ax71.set_xlabel('YME[psi]', color = 'chocolate')    
    ax71.tick_params(axis = 'x', colors = 'chocolate')
    ax71.set_xticks(scale1)
    sci_scale(ax71, scalelist1)
    xtick_loc(ax71)

    ax71.grid(True)

    ax72 = axis[6].twiny()
    ax72.scatter(tvd_core.YME, tvd_core.TVDSS, c = 'black', alpha = 1, marker = 'o')
    ax72.get_xaxis().set_visible(False) 
    ax72.set_xlim(scale1[0], scale1[-1])
    
    # Poisson's ratio plot

    plot81 = well[['PR', 'TVDSS']].dropna()
    
    ax81 = axis[7].twiny()
    ax81.plot(plot81.PR, plot81.TVDSS, color = 'salmon', linewidth = '1')
    ax81.spines['top'].set_position(('axes', 1.02))
    ax81.spines['top'].set_edgecolor('salmon') 
    ax81.set_xlim(0, 0.5)
    ax81.set_xlabel('PR[unitless]', color = 'salmon')    
    ax81.tick_params(axis = 'x', colors = 'salmon')
    ax81.set_xticks(np.arange(0, 0.51, 0.1))
    ax81.set_xticklabels(['0', '', '', '', '','0.5'])
    xtick_loc(ax81)

    ax81.grid(True)

    # unconfined compressive strength and Tensile formation strength plots

    max_scale2 = max([well.UCS.max(), well.TSTR.max()])
    scale2, scalelist2 = plot_scale(max_scale2, 0, 2000)

    plot91 = well[['UCS', 'TVDSS']].dropna()
    
    ax91 = axis[8].twiny()
    ax91.plot(plot91.UCS, plot91.TVDSS, color = 'red', linewidth = '1')
    ax91.spines['top'].set_position(('axes', 1.02))
    ax91.spines['top'].set_edgecolor('red')
    ax91.set_xlim(scale2[0], scale2[-1])
    ax91.set_xlabel('UCS[psi]', color = 'red')    
    ax91.tick_params(axis = 'x', colors = 'red')
    ax91.set_xticks(scale2)
    ax91.set_xticklabels(scalelist2)
    xtick_loc(ax91)

    ax91.grid(True)

    plot92 = well[['TSTR', 'TVDSS']].dropna()

    ax92 = axis[8].twiny()
    ax92.plot(plot92.TSTR, plot92.TVDSS, color = 'darkorange', linewidth = '1')
    ax92.spines['top'].set_position(('axes', 1.06))
    ax92.spines['top'].set_edgecolor('darkorange') 
    ax92.set_xlim(scale2[0], scale2[-1])
    ax92.set_xlabel('TSTR[psi]', color = 'darkorange')    
    ax92.tick_params(axis = 'x', colors = 'darkorange')
    ax92.set_xticks(scale2)
    ax92.set_xticklabels(scalelist2)
    xtick_loc(ax92)

    ax93 = axis[8].twiny() 
    ax93.scatter(tvd_core.UCS, tvd_core.TVDSS, c = 'black', alpha = 1, marker = 'o')
    ax93.get_xaxis().set_visible(False) 
    ax93.set_xlim(scale2[0], scale2[-1])
    
    # angle of internal friction plot

    plot101 = well[['FANG', 'TVDSS']].dropna()
    
    ax101 = axis[9].twiny()
    ax101.plot(plot101.FANG, plot101.TVDSS, color = 'green', linewidth = '1')
    ax101.spines['top'].set_position(('axes', 1.02))
    ax101.spines['top'].set_edgecolor('green') 
    ax101.set_xlim(0, 50)
    ax101.set_xlabel('FANG[degree]', color = 'green')    
    ax101.tick_params(axis = 'x', colors = 'green')
    ax101.set_xticks(np.arange(0, 51, 10))
    ax101.set_xticklabels(['0', '', '', '', '','50'])
    xtick_loc(ax101)
    
    ax101.grid(True)
    
    # principle stresses plots

    max_scale3 = max([well.OBP.max(), well.HYDRO.max(), well.SHmax.max(), well.Shmin.max(), tvd_pres.GAUGE.max(), well.PM.max()])
    scale3, scalelist3 = plot_scale(max_scale3, 0, 1000)

    plot111 = well[['SHmax', 'TVDSS']].dropna()

    ax111 = axis[10].twiny()
    ax111.plot(plot111.SHmax, plot111.TVDSS, color = 'blue', linewidth = '1')
    ax111.spines['top'].set_position(('axes', 1.02))
    ax111.spines['top'].set_edgecolor('blue')
    ax111.set_xlim(scale3[0], scale3[-1])
    ax111.set_xlabel('SHmax[psi]', color = 'blue')    
    ax111.tick_params(axis = 'x', colors = 'blue')
    ax111.set_xticks(scale3)
    ax111.set_xticklabels(scalelist3)
    xtick_loc(ax111)

    plot112 = well[['Shmin', 'TVDSS']].dropna()
    
    ax112 = axis[10].twiny()
    ax112.plot(plot112.Shmin, plot112.TVDSS, color = 'lime', linewidth = '1')
    ax112.spines['top'].set_position(('axes', 1.06))
    ax112.spines['top'].set_edgecolor('lime')
    ax112.set_xlim(scale3[0], scale3[-1])
    ax112.set_xlabel('Shmin[psi]', color = 'lime')    
    ax112.tick_params(axis = 'x', colors = 'lime')
    ax112.set_xticks(scale3)
    ax112.set_xticklabels(scalelist3)
    xtick_loc(ax112)

    plot113 = well[['OBP', 'TVDSS']].dropna()

    ax113 = axis[10].twiny()
    ax113.plot(plot113.OBP, plot113.TVDSS, color = 'black', linewidth = '1')
    ax113.spines['top'].set_position(('axes', 1.10))
    ax113.spines['top'].set_edgecolor('black')
    ax113.set_xlim(scale3[0], scale3[-1])
    ax113.set_xlabel('OBP[psi]', color = 'black')    
    ax113.tick_params(axis = 'x', colors = 'black')
    ax113.set_xticks(scale3)
    ax113.set_xticklabels(scalelist3)
    xtick_loc(ax113)

    plot114 = well[['HYDRO', 'TVDSS']].dropna()

    ax114 = axis[10].twiny()
    ax114.plot(plot114.HYDRO, plot114.TVDSS, color = 'deepskyblue', linewidth = '1')
    ax114.spines['top'].set_position(('axes', 1.14))
    ax114.spines['top'].set_edgecolor('deepskyblue')   
    ax114.set_xlim(scale3[0], scale3[-1])
    ax114.set_xlabel('HYDRO[psi]', color = 'deepskyblue')    
    ax114.tick_params(axis = 'x', colors = 'deepskyblue')
    ax114.set_xticks(scale3)
    ax114.set_xticklabels(scalelist3)
    xtick_loc(ax114)

    plot115 = well[['PM', 'TVDSS']].dropna()

    ax115 = axis[10].twiny()
    ax115.plot(plot115.PM, plot115.TVDSS, color = 'green', linewidth = '1', linestyle = '--')
    ax115.spines['top'].set_position(('axes', 1.18))
    ax115.spines['top'].set_edgecolor('green')   
    ax115.set_xlim(scale3[0], scale3[-1])
    ax115.set_xlabel('MW[psi]', color = 'green')    
    ax115.tick_params(axis = 'x', colors = 'green')
    ax115.set_xticks(scale3) 
    ax115.set_xticklabels(scalelist3)
    xtick_loc(ax115)

    ax116 = axis[10].twiny() 
    ax116.scatter(tvd_pres.GAUGE, tvd_pres.TVDSS, c = 'black', alpha = 1, marker = 'o')
    ax116.get_xaxis().set_visible(False) 
    ax116.set_xlim(scale3[0], scale3[-1])
    
    # minimum hoop stress and Tensile formation strength plots

    plot121 = well[['TSF', 'TVDSS']].dropna()
    plot122 = well[['TSTR', 'TVDSS']].dropna()

    negative_TSTR = plot122.TSTR * (-1)

    incre = 1000

    max_scale4 = max([well.TSF.max(), negative_TSTR.max()])
    min_scale4 = min([well.TSF.min(), negative_TSTR.min()])

    scale4, scalelist4 = plot_scale(max_scale4, min_scale4, incre)

    ax121 = axis[11].twiny()
    ax121.plot(plot121.TSF, plot121.TVDSS, color = 'red', linewidth = '1')
    ax121.spines['top'].set_position(('axes', 1.02))
    ax121.spines['top'].set_edgecolor('red') 
    ax121.set_xlim(scale4[0], scale4[-1])
    ax121.set_xlabel('TSF[psi]', color = 'red')
    ax121.tick_params(axis = 'x', colors = 'red')
    ax121.set_xticks(scale4)
    ax121.set_xticklabels(scalelist4)
    xtick_loc(ax121)

    ax121.grid(True)

    ax122 = axis[11].twiny()
    ax122.plot(negative_TSTR, plot122.TVDSS, color = 'darkorange', linewidth = '1')
    ax122.spines['top'].set_position(('axes', 1.06))
    ax122.spines['top'].set_edgecolor('darkorange') 
    ax122.set_xlim(scale4[0], scale4[-1])
    ax122.set_xlabel('TSTR[psi]', color = 'darkorange')    
    ax122.tick_params(axis = 'x', colors = 'darkorange')
    ax122.set_xticks(scale4)
    ax122.set_xticklabels(scalelist4)
    xtick_loc(ax122)
    
    # mud window plots

    plot131 = well[['CMW_KICK', 'TVDSS']].dropna()

    ax131 = axis[12].twiny()
    ax131.plot(plot131.CMW_KICK, plot131.TVDSS, color = 'gray', linewidth = '1')
    ax131.spines['top'].set_position(('axes', 1.02))
    ax131.spines['top'].set_edgecolor('gray')
    ax131.set_xlim(8, 18)
    ax131.set_xlabel('CMW_KICK[ppg]', color = 'gray')    
    ax131.tick_params(axis = 'x', colors = 'gray')
    ax131.set_xticks(np.arange(8, 19, 2))
    ax131.set_xticklabels(['8', '', '', '', '', '18'])
    xtick_loc(ax131)

    ax131.grid(True)

    plot132 = well[['CMW_MIN_MC', 'TVDSS']].dropna()
    
    ax132 = axis[12].twiny()
    ax132.plot(plot132.CMW_MIN_MC, plot132.TVDSS, color = 'red', linewidth = '1')
    ax132.spines['top'].set_position(('axes', 1.06))
    ax132.spines['top'].set_edgecolor('red')
    ax132.set_xlim(8, 18)
    ax132.set_xlabel('CMW_MIN_MC[ppg]', color = 'red')    
    ax132.tick_params(axis = 'x', colors = 'red')
    ax132.set_xticks(np.arange(8, 19, 2))
    ax132.set_xticklabels(['8', '', '', '', '', '18'])
    xtick_loc(ax132)

    plot133 = well[['CMW_LOSS', 'TVDSS']].dropna()

    ax133 = axis[12].twiny()
    ax133.plot(plot133.CMW_LOSS, plot133.TVDSS, color = 'indigo', linewidth = '1')
    ax133.spines['top'].set_position(('axes', 1.10))
    ax133.spines['top'].set_edgecolor('indigo')
    ax133.set_xlim(8, 18)
    ax133.set_xlabel('CMW_LOSS[ppg]', color = 'indigo')    
    ax133.tick_params(axis = 'x', colors = 'indigo')
    ax133.set_xticks(np.arange(8, 19, 2))
    ax133.set_xticklabels(['8', '', '', '', '', '18'])
    xtick_loc(ax133)

    plot134 = well[['CMW_MAX_MTS', 'TVDSS']].dropna()

    ax134 = axis[12].twiny()
    ax134.plot(plot134.CMW_MAX_MTS, plot134.TVDSS, color = 'darkslateblue', linewidth = '1')
    ax134.spines['top'].set_position(('axes', 1.14))
    ax134.spines['top'].set_edgecolor('darkslateblue')   
    ax134.set_xlim(8, 18)
    ax134.set_xlabel('CMW_MAX_MTS[ppg]', color = 'darkslateblue')    
    ax134.tick_params(axis = 'x', colors = 'darkslateblue')
    ax134.set_xticks(np.arange(8, 19, 2))
    ax134.set_xticklabels(['8', '', '', '', '', '18'])
    xtick_loc(ax134)

    plot135 = well[['MW', 'TVDSS']].dropna()
    
    ax135 = axis[12].twiny()
    ax135.plot(plot135.MW, plot135.TVDSS, color = 'green', linewidth = '1', linestyle = '--')
    ax135.spines['top'].set_position(('axes', 1.18))
    ax135.spines['top'].set_edgecolor('green')   
    ax135.set_xlim(8, 18)
    ax135.set_xlabel('MW[ppg]', color = 'green')    
    ax135.tick_params(axis = 'x', colors = 'green')
    ax135.set_xticks(np.arange(8, 19, 2)) 
    ax135.set_xticklabels(['8', '', '', '', '', '18'])
    xtick_loc(ax135)

    plot136 = well[['CMW_KICK', 'CMW_MIN_MC', 'CMW_LOSS', 'CMW_MAX_MTS', 'TVDSS']].dropna()

    loc1 = plot136.CMW_KICK > 8
    loc2 = plot136.CMW_MIN_MC > plot136.CMW_KICK
    loc3 = plot136.CMW_MAX_MTS > plot136.CMW_LOSS
    loc4 = 18 > plot136.CMW_MAX_MTS
    
    ax136 = axis[12].twiny()
    ax136.set_xlim(8, 18)
    ax136.fill_betweenx(plot136.TVDSS, 8, plot136.CMW_KICK, where = loc1, color='silver', capstyle = 'butt', linewidth = 0.5, alpha = 0.5, label = 'KICK')
    ax136.fill_betweenx(plot136.TVDSS, plot136.CMW_KICK, plot136.CMW_MIN_MC, where = loc2, color='yellow', capstyle = 'butt', linewidth = 0.5, alpha = 0.5, label = 'BREAKOUT')
    ax136.fill_betweenx(plot136.TVDSS, plot136.CMW_LOSS, plot136.CMW_MAX_MTS, where = loc3, color='slateblue', capstyle = 'butt', linewidth = 0.5, alpha = 0.5, label = 'LOSS')
    ax136.fill_betweenx(plot136.TVDSS, plot136.CMW_MAX_MTS, 18, where = loc4, color='darkslateblue', capstyle = 'butt', linewidth = 0.5, alpha = 0.5, label = 'BREAKDOWN')
    ax136.set_xticks(np.arange(8, 19, 2))
    ax136.set_xticklabels(['', '', '', '', '', ''])
    ax136.legend(loc = 'upper left')
    
    # breakout width plot

    wbo1 = well.WBO / 2
    wbo2 = (well.WBO / 2) * (-1)

    loc5 = well.WBO.notna()

    ax141 = axis[13].twiny()
    ax141.fill_betweenx(well.TVDSS, wbo2, wbo1, where = loc5, color = 'red', capstyle = 'butt', linewidth = 1, label = 'BAD')
    ax141.spines['top'].set_position(('axes', 1.02))
    ax141.spines['top'].set_edgecolor('red')   
    ax141.set_xlim(-90, 90)
    ax141.set_xlabel('WBO[degree]', color = 'red')    
    ax141.tick_params(axis = 'x', colors = 'red')
    ax141.set_xticks(np.arange(-90, 91, 45))
    ax141.set_xticklabels(['-90', '', '', '', '90'])
    xtick_loc(ax141)

    ax141.grid(True)

    # caliper and bitsize plots

    plot151 = well[['BS', 'TVDSS']].dropna()
    
    ax151 = axis[14].twiny()
    ax151.set_xlim(6, 11)
    ax151.plot(plot151.BS, plot151.TVDSS, color = 'black', linewidth = '1')
    ax151.spines['top'].set_position(('axes', 1.02))
    ax151.set_xlabel('BS[in]',color = 'black')
    ax151.tick_params(axis = 'x', colors = 'black')
    ax151.set_xticks(np.arange(6, 12, 1))
    ax151.set_xticklabels(['6', '', '', '', '', '11'])
    xtick_loc(ax151)

    ax151.grid(True)

    plot152 = well[['CAL', 'TVDSS']].dropna()
    
    ax152 = axis[14].twiny()
    ax152.set_xlim(6, 11)
    ax152.plot(plot152.CAL, plot152.TVDSS, color = 'grey', linewidth = '1')
    ax152.spines['top'].set_position(('axes', 1.06))
    ax152.spines['top'].set_edgecolor('grey')
    ax152.set_xlabel('CAL[in]',color = 'grey')
    ax152.tick_params(axis = 'x', colors = 'grey')
    ax152.set_xticks(np.arange(6, 12, 1))
    ax152.set_xticklabels(['6', '', '', '', '', '11'])
    xtick_loc(ax152)

    plot153 = well[['CAL', 'BS', 'TVDSS']].dropna()

    loc6 = plot153.BS > plot153.CAL
    loc7 = plot153.CAL > plot153.BS

    ax153 = axis[14].twiny()
    ax153.set_xlim(6, 11)
    ax153.fill_betweenx(plot153.TVDSS, plot153.CAL, plot153.BS, where = loc6, color='yellow', capstyle = 'butt', linewidth = 0.5, alpha = 0.5, label = 'SWELLING')
    ax153.fill_betweenx(plot153.TVDSS, plot153.BS, plot153.CAL, where = loc7, color='red', capstyle = 'butt', linewidth = 0.5, alpha = 0.5, label = 'CAVING')
    ax153.set_xticks(np.arange(6, 12, 1))
    ax153.set_xticklabels(['', '', '', '', '', ''])
    # ax153.legend(loc = 'upper left')

    # effective porosity, rock matrix, volume of clay plots

    plot161 = well[['VCL', 'TVDSS']].dropna()

    ax161 = axis[15].twiny()
    ax161.plot(plot161.VCL, plot161.TVDSS, color = 'SaddleBrown', linewidth = '1')
    ax161.spines['top'].set_position(('axes', 1.02))
    ax161.spines['top'].set_edgecolor('SaddleBrown')
    ax161.set_xlim(0, 1)
    ax161.set_xlabel('VCL[V/V]', color = 'SaddleBrown')    
    ax161.tick_params(axis = 'x', colors = 'SaddleBrown')
    ax161.set_xticks(np.arange(0, 1.1, 0.2))
    ax161.set_xticklabels(['0', '', '', '', '','1'])
    xtick_loc(ax161)

    plot162 = well[['PHIE', 'TVDSS']].dropna()

    ax162 = axis[15].twiny()
    ax162.plot(plot162.PHIE, plot162.TVDSS, color = 'gray', linewidth = '1')
    ax162.spines['top'].set_position(('axes', 1.06))
    ax162.spines['top'].set_edgecolor('gray')
    ax162.set_xlim(1, 0)
    ax162.set_xlabel('PHIE[V/V]', color = 'gray')    
    ax162.tick_params(axis = 'x', colors = 'gray')
    ax162.set_xticks(np.arange(1.0, -0.1, -0.2))
    ax162.set_xticklabels(['1', '', '', '', '','0'])
    xtick_loc(ax161)

    plot163 = well[['VCL', 'PHIE', 'TVDSS']].dropna()

    ax163 = axis[15].twiny()
    ax163.set_xlim(0, 1)
    ax163.fill_betweenx(plot163.TVDSS, 0, plot163.VCL, color='SaddleBrown', capstyle = 'butt', linewidth = 0.5, label = 'VCLAY')
    ax163.fill_betweenx(plot163.TVDSS, plot163.VCL, (1 - plot163.PHIE), color='yellow', capstyle = 'butt', linewidth = 0.5, label = 'MATRIX')
    ax163.fill_betweenx(plot163.TVDSS, (1 - plot163.PHIE), 1, color='gray', capstyle = 'butt', linewidth = 0.5, label = 'POROSITY')
    ax163.set_xticks([0, 1])
    ax163.set_xticklabels(['', ''])
    # ax163.legend(loc = 'upper left')

    ax163.grid(True)

    # plot sand-shale lithology

    well['liplot'] = np.nan
    well['liplot'].loc[well.LITHO == 'SAND'] = 1
    well['liplot'].loc[well.LITHO == 'SHALE'] = 0

    plot171 = well[['liplot', 'TVDSS']].dropna()
        
    ax171 = axis[16].twiny()
    ax171.fill_betweenx(plot171.TVDSS, plot171.liplot, 1, color = 'SaddleBrown', capstyle = 'butt', linewidth = 0.01, label = 'SHALE')
    ax171.fill_betweenx(plot171.TVDSS, 0, plot171.liplot, color = 'yellow', capstyle = 'butt', linewidth = 0.01, label = 'SAND')
    ax171.spines['top'].set_position(('axes', 1.02))
    ax171.spines['top'].set_edgecolor('gray')
    ax171.set_xlim(0, 1)
    ax171.set_xlabel('LITHOLOGY', color = 'gray')
    ax171.tick_params(axis = 'x', colors = 'gray')
    ax171.set_xticks([0, 1])
    ax171.set_xticklabels(['', ''])
    ax171.legend(loc = 'upper left')

    well.drop(columns = ['liplot'], inplace = True)

    # Save files

    mem_folder = 'MEM'
    mem_path = os.path.join(sav_path, mem_folder)

    if not os.path.isdir(mem_path):
        os.makedirs(mem_path)

    plt.savefig(os.path.join(mem_path, form_mem_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# construct 1D MEM for specific formation

for las, well, tvd_top, tvd_pres, tvd_core, data_range in zip(lases, wells, tvd_tops, tvd_press, tvd_cores, data_ranges):
    for form in selected_forms:
        form_mem_name = '%s_MEM_%s.png' %(las.well['WELL'].value, form)
        form_mem_plot(las, well, tvd_top, tvd_pres, tvd_core, data_range, all_forms, form, form_mem_name)

# Export las and csv files 2

for las, well in zip(lases, wells):
    las_name2 = 'LQC_%s_MEM.las' %las.well['WELL'].value
    csv_name2 = 'LQC_%s_MEM.csv' %las.well['WELL'].value
    export_well(las, well, las_name2, csv_name2)
    print('Well data of %s are exported to las and csv files.' %las.well['WELL'].value)

"""

Application for fracturing

"""

# Function of calculate initiation, reopening, and closure pressures for fracturing

def frac_cal(well):
    """
    This function will calculate initiation, reopening, and closure pressures for fracturing
    wells = completed well data in pandas dataframe (Merged data with the synthetics)
    """

    # convert unit 3.28084 for m to ft, 1/0.052 for psi/ft to ppg

    cal_data = well.copy()
    cal_data['ppg_Shmin'] = (cal_data.Shmin/cal_data.TVD) * (1/3.28084) * (1/0.052)
    cal_data['ppg_SHmax'] = (cal_data.SHmax/cal_data.TVD) * (1/3.28084) * (1/0.052)
    cal_data['ppg_TSTR'] = (cal_data.TSTR/cal_data.TVD) * (1/3.28084) * (1/0.052)

    # fracture initiation pressure in ppg

    cal_data['Pi'] = (3 * cal_data.ppg_Shmin) - cal_data.ppg_SHmax - cal_data.WPG + cal_data.ppg_TSTR

    # fracture reopening pressure in ppg

    cal_data['Pfr'] = (3 * cal_data.ppg_Shmin) - cal_data.ppg_SHmax - cal_data.WPG

    # fracture closure pressure in ppg

    cal_data['Pc'] = cal_data.ppg_Shmin

    print('Fracture initiation, reopening, and closure pressures are estimated.')

    return cal_data

# Calculate initiation, reopening, and closure

cal_datas = []

for las, well, tvd_top in zip(lases, wells, tvd_tops):
        print('For well %s' %las.well['WELL'].value)
        cal_data = frac_cal(well)
        cal_datas.append(cal_data)

# Function for hydraulic fracturing pressures plot for specific formation

def frac_mud_plot(las, cal_data, tvd_top, all_forms, formation, frac_name):
    """
    This function is plotting hydraulic fracturing pressures plot for specific formation.
    las = las file (.las) of the cal_data data.
    cal_data = cal_data logging data in pandas data frame with alias applied.
    tvd_top = formation top data in pandas data frame.
    all_forms = list of all formation names with color code in dictionary format
    formation = input the name of the formation where the data can be compared
    mem_name = name of saved figure.
    """
    # Create figure

    fig, axis = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 12), sharey = True)
    fig.suptitle('Hydraulic Fracturing Pressures \nFor %s of well %s' %(formation, las.well['WELL'].value), fontsize = 20, y = 1.0)

    # Set interval from each well for selected formation

    start = float(tvd_top.loc[tvd_top.FORMATIONS == formation, 'TOP'])
    stop = float(tvd_top.loc[tvd_top.FORMATIONS == formation, 'BOTTOM'])

    condition = (cal_data.index >= start) & (cal_data.index <= stop)

    top_depth = cal_data.loc[condition, 'TVDSS'].dropna().min()
    bottom_depth = cal_data.loc[condition, 'TVDSS'].dropna().max()

    # General setting

    axis.set_ylim(top_depth, bottom_depth)
    axis.set_ylabel('TVDSS[m]')
    axis.invert_yaxis()
    axis.minorticks_on() #Scale axis
    axis.get_xaxis().set_visible(False)
    axis.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
    axis.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'red')

    # formation plot

    axis.axhline(y = top_depth, linewidth = 1.5, color = all_forms[formation], alpha = 0.3)
    axis.axhspan(top_depth, bottom_depth, facecolor = all_forms[formation], alpha = 0.1)

    middle_depth = top_depth + (bottom_depth - top_depth)/2
                
    axis.text(0.01, middle_depth , formation, ha = 'left', va = 'center', color = all_forms[formation], 
                path_effects = [pe.withStroke(linewidth = 3, foreground = "white")], fontsize = 10, weight = 'bold')

    # initiation

    plot1 = cal_data[['Pi', 'TVDSS']].dropna()

    ax11 = axis.twiny()
    ax11.plot(plot1.Pi, plot1.TVDSS, color = 'red', linewidth = '1')
    ax11.spines['top'].set_position(('axes', 1.02))
    ax11.spines['top'].set_edgecolor('red')
    ax11.set_xlim(10, 28)
    ax11.set_xlabel('Initiation[ppg]', color = 'red')    
    ax11.tick_params(axis = 'x', colors = 'red')
    ax11.set_xticks(np.arange(10, 29, 3))
    ax11.set_xticklabels(['10', '', '', '', '', '', '28'])

    ax11.grid(True)

    # reopening

    plot2 = cal_data[['Pfr', 'TVDSS']].dropna()
    
    ax12 = axis.twiny()
    ax12.plot(plot2.Pfr, plot2.TVDSS, color = 'blue', linewidth = '1')
    ax12.spines['top'].set_position(('axes', 1.08))
    ax12.spines['top'].set_edgecolor('blue')
    ax12.set_xlim(10, 28)
    ax12.set_xlabel('Reopening[ppg]', color = 'blue')    
    ax12.tick_params(axis = 'x', colors = 'blue')
    ax12.set_xticks(np.arange(10, 29, 3))
    ax12.set_xticklabels(['10', '', '', '', '', '', '28'])

    # closure

    plot3 = cal_data[['Pc', 'TVDSS']].dropna()

    ax13 = axis.twiny()
    ax13.plot(plot3.Pc, plot3.TVDSS, color = 'black', linewidth = '1')
    ax13.spines['top'].set_position(('axes', 1.14))
    ax13.spines['top'].set_edgecolor('black')
    ax13.set_xlim(10, 28)
    ax13.set_xlabel('Closure[ppg]', color = 'black')    
    ax13.tick_params(axis = 'x', colors = 'black')
    ax13.set_xticks(np.arange(10, 29, 3))
    ax13.set_xticklabels(['10', '', '', '', '', '', '28'])

    fig.tight_layout()

    # Save files

    mem_folder = 'MEM'
    mem_path = os.path.join(sav_path, mem_folder)

    if not os.path.isdir(mem_path):
        os.makedirs(mem_path)

    plt.savefig(os.path.join(mem_path, frac_name), dpi = 200, format = 'png', bbox_inches = "tight")

    plt.show()

# construct fracturing mud weight profiles for specific formation

for las, cal_data, tvd_top in zip(lases, cal_datas, tvd_tops):
    for form in selected_forms:
        frac_name = '%s_frac_%s.png' %(las.well['WELL'].value, form)
        frac_mud_plot(las, cal_data, tvd_top, all_forms, form, frac_name)
