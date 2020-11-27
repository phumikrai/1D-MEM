"""
1. Data Audit
"""

# Support modules

import glob, os, re, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import lasio # Las file reader module
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

"""
Import data
"""

note = """
1.) This is for modeling based on isotropic homogeneous material assumption using statistic decisions or machine learning techniques. 

2.) The working directory should contain the data for modeling as sub-directory. 

3.) All data for modeling include well logging files (.las), deviation files (.csv) and formation top (.csv) must be separated
as sub-directory of the data directory. 
For example;
- Working directory is "Drive:/Working/".
- All data for modeling directory is "Drive:/Working/Data/".
- Well logging file directory is "Drive:/Working/Data/Well logging/" as Sub-directory of the data directory.
- Deviation file directory is "Drive:/Working/Data/Deviation/" as Sub-directory of the data directory.
- Formation top file directory is "Drive:/Working/Data/Formation top/" as Sub-directory of the data directory.

4.) Well name should be set as prefix for each file. Its name will cause file ordering and file pairing for each file of that well.
For example;
- Well name is "Well-01" (Noted: No underscore ('_') be contained in well name), so this name should be set as prefix followed by underscore ('_') for each modeling input file like this "Well-01_(...Specific name for file type indication...)"
- Example; Well logging file name, Deviation file name and Formation top file name could be "Well-01_las", "Well-01_dev" and "Well-01_top" respectively.

5.: Required data and file format;
- Well logging files include all necessary curves for 1D MEM such caliper (CAL), bitsize (BS), gamma ray (GR), density (RHOB), neutron porosity (NPHI), deep resistivity (RT), shallow resistivity (MSFL), P-sonic (DTC) and S-sonic (DTS).
- Deviation files include measured depth column named with 'MD', azimuth column named with 'AZIMUTH' and inclination or angle column named with 'ANGLE'.
- Formation top files include formation name column named with 'Formations' and top depth column named with 'Top'.
"""
print('Welcome to Automated 1D Mechanical Earth Modeling (Auto 1D MEM).')
print('Please take note on this;')
print(note)

# Setup data directory

cwd_dir_list = ', '.join(os.listdir(os.getcwd()))
confirm = 'no'

print('According to your working directory,')
print('%s\nwhich one is your data directory?' %cwd_dir_list)

while confirm.lower() == 'no':
    data_folder = input('Please indicate the data directory name: ').strip()
    data_path = os.path.join(os.getcwd(), data_folder)

    if data_folder == '':
        print('Please type the directory name!')
        continue

    elif os.path.isdir(data_path):
        data_dir_list = ', '.join(os.listdir(data_path))
        print('%s\nThese sub-directories are found.' %data_dir_list)

        while True:
            print('Which one is your Well logging file directory?')
            las_folder = input('Please indicate the well logging file directory name (.las): ').strip()
            las_path = os.path.join(os.getcwd(), data_folder, las_folder)

            if las_folder == '':
                print('Please type the directory name!')
                continue

            elif os.path.isdir(las_path):
                
                while True:
                    print('Which one is your deviation file directory?')
                    dev_folder = input('Please indicate the deviation file directory name (.csv): ').strip()
                    dev_path = os.path.join(os.getcwd(), data_folder, dev_folder)

                    if dev_folder == '':
                        print('Please type the directory name!')
                        continue

                    elif os.path.isdir(dev_path):

                        while True:
                            print('Which one is your formation top file directory?')
                            top_folder = input('Please indicate the formation top file directory name (.csv): ').strip()
                            top_path = os.path.join(os.getcwd(), data_folder, top_folder)

                            if top_folder == '':
                                print('Please type the directory name!')
                                continue

                            elif os.path.isdir(top_path):
                                print('Gotcha!')
                                print('Your well logging file directory is: %s.' %las_path)
                                print('Your deviation file directory is: %s.' %dev_path)
                                print('Your formation top file directory is: %s.' %top_path)

                                while True:
                                    confirm = input('Are these correct? [Yes/No]: ')

                                    if confirm.lower() == 'yes':
                                        break
                                    
                                    elif confirm.lower() == 'no':
                                        break

                                    else:
                                        print('Please confirm again!')
                                break

                            else:
                                print('Please try again, your directory \'%s\' is not found!' %top_folder)
                        break

                    else:
                        print('Please try again, your directory \'%s\' is not found!' %dev_folder)
                break

            else:
                print('Please try again, your directory \'%s\' is not found!' %las_folder)

    else:
        print('Please try again, your directory \'%s\' is not found!' %data_folder)

# Function for pairing the data and eliminating the incomplete

def pairing_files(las_files_paths, dev_files_paths, top_files_paths):
    """
    This function is going to pairing the data (las files, dev files and top files) and disable the incompleted one.
    las_files_paths = list of las files with paths
    dev_files_paths = list of deviation files with paths
    top_files_paths = list of formation top files with paths
    """
    paired_las_files_paths = []
    paired_dev_files_paths = []
    paired_top_files_paths = []

    # pairing lAS file to deviation

    for las in las_files_paths:
        for dev in dev_files_paths:
            for top in top_files_paths:

                las_well_name = os.path.basename(las).split('_', 1)[0].lower()
                dev_well_name = os.path.basename(dev).split('_', 1)[0].lower()
                top_well_name = os.path.basename(top).split('_', 1)[0].lower()

                if las_well_name == dev_well_name == top_well_name:
                    paired_las_files_paths.append(las)
                    paired_dev_files_paths.append(dev)
                    paired_top_files_paths.append(top)

    return paired_las_files_paths, paired_dev_files_paths, paired_top_files_paths

# Generate file path

las_files = glob.glob(os.path.join(las_path, '*.las'))
dev_files = glob.glob(os.path.join(dev_path, '*.csv'))
top_files = glob.glob(os.path.join(top_path, '*.csv'))

# Pairing files lAS files, dev files and top files

las_files, dev_files, top_files = pairing_files(las_files, dev_files, top_files)

# Import lAS files, dev files and top files

lases = [] # Storing well logging data
df_lases = [] # Storing well logging data in panda data frame
devs = [] # Storing deviation data in panda data frame
tops = []# Storing formation top data in panda data frame

for las_file, dev_file, top_file in zip(las_files, dev_files, top_files):

    # Well logging data

    las = lasio.read(las_file)
    lases.append(las)

    # Well logging data in panda data frame

    df = las.df()
    df = df.rename_axis('MD')
    df_lases.append(df)

    # Deviation data in panda data frame

    dev = pd.read_csv(dev_file)
    devs.append(dev)

    # Fomation top data in panda data frame

    top = pd.read_csv(top_file)
    tops.append(top)

# Set directory to save files

sav_folder = 'LQC files'
sav_path = os.path.join(data_path, sav_folder)

if not os.path.isdir(sav_path):
    os.makedirs(sav_path)

# Well names

well_names = []

for las in lases:
    well_names.append(las.well['WELL'].value)

print('The number of wells is %d.' %len(well_names))
print('Well names are %s.' %', '.join(well_names))

# Define field parameters to adjust (remove air gap) the well logging by oil field type in the next step

air_gaps = []
water_levels = [] # for offshore field only.

while True:
    field_type = input('What is this oil field type [Onshore/Offshore]: ').strip()

    if field_type.lower() == 'onshore':
        for name, df_las, dev in zip(well_names, df_lases, devs):
            print('Please type basic information for well %s' %name)

            kb = float(input('Kelly Bushing depth [Kelly bushing to sea level]: ').strip())
            gl = float(input('Ground level elevetion [Ground surface to sea level]: ').strip())
            gap = kb - gl
            air_gaps.append(gap)
        break
    
    elif field_type.lower() == 'offshore':
        for name, df_las, dev in zip(well_names, df_lases, devs):
            print('Please type basic information for well %s' %name)

            kb = float(input('Kelly Bushing depth: ').strip())
            wl = float(input('Water depth [Sea level to seafloor]: ').strip())
            water_levels.append(wl)
            air_gaps.append(kb)
        break
    
    else:
        print('Please type only \'Onshore\' or \'Offshore\'')
        continue

# Function for checking the curve completion of well

def check_curves(las_file, alias, mem_curves):
    """
    This function can check the curve completion of well.
    las_file = las file read by lasio
    alias = curve alias or alterative name of the curve
    mem_curves = necessary curve names for modeling
    """
    curves = [curve.mnemonic for curve in las_file.curves]
    extracted = []

    for curve in curves:
        for key, values in alias.items():
            if (curve.lower() in [value.lower() for value in values]) & (key in mem_curves):
                extracted.append(key)

    if set(extracted) == set(mem_curves):
        print('All necessary curves in well %s are completed' %las_file.well['WELL'].value)

    else:
        print('All necessary curves in well %s are incompleted.' %las_file.well['WELL'].value)
        
        if len(set(extracted).difference(set(mem_curves))) == 1:
            print('Curve %s is missing.' %', '.join([curve for curve in set(mem_curves) - set(extracted)]))

        else:
            print('Curves %s are missing.' %', '.join([curve for curve in set(mem_curves) - set(extracted)]))

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

based_curves = ['MD', 'AZIMUTH', 'ANGLE', 'BHF']

# Check available curves

print('Available curves for each well;')

for las, name in zip(lases, well_names):
    print(name, 'curves are: \n%s' %', '.join([curve.mnemonic for curve in las.curves]))
    check_curves(las, alias, mem_curves)

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

# Apply function to ordering all formation

all_forms = []

for forms in tops:
    only_forms = forms.dropna().Formations
    if all_forms == []:
        for form in only_forms:
            all_forms.append(form)
    else:
        all_forms = merge_sequences(all_forms, list(only_forms))

# Function for arranging the formation following the reference one
 
def forms_arr(ref_forms, app_forms):
    """
    This function will arrange the order of the formation in list following the reference.
    ref_forms = formation order will be arranged following this reference.
    app_forms = list of the formations will be applied.
    """
    for form in ref_forms:
        if form.lower() in [form.lower() for form in app_forms]:
            app_forms.pop([form.lower() for form in app_forms].index(form.lower()))
            app_forms.append(form)
    
    return app_forms

# Function for checking formation names of the input

def check_forms(ref_forms, input_forms):
    """
    This function will check the available formation following the reference and prepare the input for the next step.
    ref_forms = available formation will be checked based on this reference.
    input_names = names of the formation 
    """
    form_names = []

    for form in ref_forms:
        if form.lower() in [name.strip().lower() for name in input_forms.split(',')]:
            form_names.append(form)
    
    return form_names

# Function for adding the formation to the selected formation list

def add_forms(selected_forms, non_selected_forms):
    """
    This function can add more formation to selected formation list.
    selected_forms = the list of selected formations that will be added more by user.
    non_selected_forms = the list of non-selected formations
    *check_forms function is required.
    """
    print('Which one do you want to add more?')

    while True:
        select = input('[Comma can be used for multi-input]: ').strip()
        selected_form = check_forms(non_selected_forms, select)

        if select == '':
            print('Please type formation names!')
            continue

        elif selected_form == []:
            print('Please try again!, formation \'%s\' is not found.' %select)
            continue

        elif set([form.lower() for form in selected_form]).issubset(set([form.lower() for form in non_selected_forms])):
            for form in selected_form:
                selected_forms.append(form)
                non_selected_forms.pop([form.lower() for form in non_selected_forms].index(form.lower()))
            break             
                 
    return selected_forms, non_selected_forms

# Function for removing the formation in the selected formation list

def remove_forms(selected_forms, non_selected_forms):
    """
    This function can remove the seleted formation.
    selected_forms = the list of selected formations that will be removed by user.
    non_selected_forms = the list of non-selected formations
    *check_forms function is required.
    """
    print('Which one do you want to remove?')

    while True:
        delete = input('[Comma can be used for multi-input]: ').strip()
        del_forms = check_forms(selected_forms, delete)

        if delete == '':
            print('Please type formation names!')
            continue

        elif del_forms == []:
            print('Please try again!, formation \'%s\' is not found.' %delete)
            continue       

        elif set([form.lower() for form in del_forms]).issubset(set([form.lower() for form in selected_forms])):
            for form in del_forms:
                selected_forms.pop([form.lower() for form in selected_forms].index(form.lower()))
                non_selected_forms.append(form)

            if len(del_forms) == 1:
                print('Formation %s is removed' %''.join(del_forms))
            
            else:
                print('Formation %s are removed' %', '.join(del_forms))
            break
    
    return selected_forms, non_selected_forms

# Show all selectable formations

print('All formations in this field are: %s.' %', '.join(all_forms))

# Define selected formations in this project to focus

selected_forms = []
non_selected = all_forms.copy() # Be used only in this step

while True:
    print('Which one is your selected formation?')
    select = input('[Comma can be used for multi-input]: ').strip()
    selected_form = check_forms(all_forms, select)

    if select == '':
        print('Please type formation names!')
        continue

    elif selected_form == []:
        print('Please try again!, formation \'%s\' is not found.' %select)
        continue

    elif set([form.lower() for form in selected_form]).issubset(set([form.lower() for form in all_forms])):
        
        for form in selected_form:
            selected_forms.append(form)
            non_selected.pop([form.lower() for form in non_selected].index(form.lower()))

        while True:
            selected_forms = forms_arr(all_forms, selected_forms)
            non_selected = forms_arr(all_forms, non_selected)        

            print('Now, only formation \'%s\' will be your selected formations' %', '.join(selected_forms))
            confirm = input('Are you okay with this? [Ok/Not]: ').strip()
            
            if confirm.lower() == 'ok':
                print('Got it, sir/ma\'am!')
                break
            
            elif confirm.lower() == 'not':
                
                while True:
                    options = input('What do you want to do? add more or edit (remove)? [Add/Remove]: ').strip()

                    if options.lower() == 'add':
                        print('The other formation in this field are: %s.' %', '.join(non_selected))
                        selected_forms, non_selected = add_forms(selected_forms, non_selected)
                        break

                    elif options.lower() == 'remove':
                        selected_forms, non_selected = remove_forms(selected_forms, non_selected)

                        if selected_forms == []:
                            print('No formation is selected!, please select formation.')
                            print('The available formation in this field are: %s.' %', '.join(all_forms))
                            selected_forms, non_selected = add_forms(selected_forms, non_selected)
                        break

                    else:
                        print('Please confirm again!')
                        continue
                continue

            else:
                print('Please confirm again!')
                continue
        break

"""
Depth conversion
"""

# Create function for TVD computation by minimum curvature method

def tvd_mini_cuv(dev):
    """
    TVD computation function using minimum curvature survey calculation method
    dev = Deviation survey data in pandas data frame which contains:
             1. Measured depth (MD) in column name "MD"
             2. Azimuth direction (AZIMUTH) in column name "AZIMUTH"
             3. Inclination angle (ANGLE) in column name "ANGLE"
    """
    # setup parameters
    
    md = dev.MD
    prev_md = md.shift(periods = 1, fill_value = 0)
    diff_md = md - prev_md
    
    ang = dev.ANGLE
    prev_ang = ang.shift(periods = 1, fill_value = 0)
    diff_ang = ang - prev_ang
    
    azi = dev.AZIMUTH
    prev_azi = azi.shift(periods = 1, fill_value = 0)
    diff_azi = azi - prev_azi
    
    # computation
    
    cos_theta = np.cos(np.radians(diff_ang)) - (np.sin(np.radians(ang)) * np.sin(np.radians(prev_ang)) * (1 - np.cos(np.radians(diff_azi))))
    theta = np.arccos(cos_theta)
    
    rf = ((2 / theta) * np.tan(theta/2)).fillna(0)
    
    dev['TVD'] = np.cumsum((diff_md / 2) * (np.cos(np.radians(ang)) + np.cos(np.radians(prev_ang))) * rf)
    
    return dev

# Calculate TVD for all well deviations in deviation files

devs = list(map(tvd_mini_cuv, devs))

# Generate function to convert MD to TVD in data with deviation survey data 

def tvd_interpolate(las, df_las, dev):
    """
    MD to TVD interpolation using linear interpolation method and update las file
    las = las file (.las) of the well data
    df_las = las input in pandas data frame contains depth column in measured depth (MD)
    dev = deviation survey data in pandas data frame contains depth columns in both measured depth (MD) and true vertical depth (TVD)
    """
    # Merge deviation file with well data 
    
    df_las = df_las.reset_index()
    df_las = pd.concat([dev[['MD', 'AZIMUTH', 'ANGLE', 'TVD']], df_las]).sort_values(by = ['MD']).reset_index(drop = True)
    
    # Insert true vertical depth using linear interpolation
    
    for col in df_las[['AZIMUTH', 'ANGLE', 'TVD']].columns:
        df_las[col] = df_las[col].interpolate(method = 'linear', limit_area = 'inside')
        
    # Set true vertical depth as file indices
        
    df_las = df_las.dropna(subset = ['TVD']).set_index('TVD')
    df_las = df_las.drop(list(dev['TVD']))
    
    # Update las files

    las.insert_curve(0, 'TVD', df_las.index, unit = 'm', descr = 'True Vertical Depth', value = '')
    las.insert_curve(1, 'MD', df_las.index, unit = 'm', descr = 'Measured Depth', value = '')
    las.insert_curve(2, 'AZIMUTH', df_las.index, unit = 'degree', descr = 'well Deviation in Azimuth', value = '')
    las.insert_curve(3, 'ANGLE', df_las.index, unit = 'degree', descr = 'well Deviation in Angle', value = '')
    del las.curves['DEPTH']

    print('Measured depth (MD) is converted to True vertical depth (TVD) for well %s' %las.well['WELL'].value)
    
    return las, df_las

# Function for adjust true vertical depth of well logging data

def remove_gap(tvd_las, gap):
    """
    This function can remove air gap from the true vertical depth (TVD) for well logging data.
    tvd_las = las input in pandas data frame contains depth column in true vertical depth (TVD)
    gap = air gap value (onshore = kelly bushing - ground level, offshore = kelly bushing)
    """
    tvd_las = tvd_las.reset_index()
    tvd_las['TVD'] = tvd_las['TVD'] - gap
    tvd_las['MD'] = tvd_las['MD'] - gap
    tvd_las = tvd_las.set_index('TVD')

    return tvd_las

# Implement interpolation function

tvd_lases = []

for las, df_las, dev, gap in zip(lases, df_lases, devs, air_gaps):
    las, tvd_las = tvd_interpolate(las, df_las, dev)
    tvd_las = remove_gap(tvd_las, gap)
    tvd_lases.append(tvd_las)

"""
Bad Hole Flag (BHF)
"""

# Function for create Bad Hole flag

def create_bhf(las, tvd_las, alias):
    """
    This function can compute Bad Hole Flag using confidential interval (ci) and update las file
    las = las file (.las) of the well data
    tvd_las = well logging data in data frame (converted true vertical depth)
    * Caliper and bitsize data are required.
    """
    ci = 0.75 # default

    for col in tvd_las.columns:
        if col in alias['CAL']:
            caliper = col
        elif col in alias['BS']:
            bitsize = col

    confirm = 'not'

    while not confirm.lower() == 'ok':
        diff = tvd_las[caliper] - tvd_las[bitsize]
        interval = st.norm.interval(alpha = ci, loc = round(np.mean(diff), 2), scale = round(np.std(diff), 2))

        tvd_las['BHF'] = (diff.dropna() > interval[0]) & (diff.dropna() < interval[1])
        tvd_las['BHF'] = tvd_las['BHF']*1
        tvd_las['BHF'] ^= 1

        #create figure

        fig, axis = plt.subplots(nrows = 1, ncols = 2, figsize = (6, 10))
        fig.suptitle(las.well['WELL'].value, fontsize = 15, y = 1.0)

        #General setting for all axis
        
        for ax in axis:
            ax.set_ylim(tvd_las.index.min(), tvd_las.index.max())
            ax.invert_yaxis()
            ax.minorticks_on() #Scale axis
            ax.get_xaxis().set_visible(False)
            ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
            ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black') 

        # caliper - bitsize plot
        
        ax11 = axis[0].twiny()
        ax11.set_xlim(6,15)
        ax11.plot(tvd_las[bitsize], tvd_las.index, color = 'black')
        ax11.spines['top'].set_position(('outward',0))
        ax11.set_xlabel('BS[in]',color = 'black')
        ax11.tick_params(axis = 'x', colors = 'black')
        
        ax12 = axis[0].twiny()
        ax12.set_xlim(6,15)
        ax12.plot(tvd_las[caliper], tvd_las.index, color = 'grey' )
        ax12.spines['top'].set_position(('outward',40))
        ax12.set_xlabel('CAL[in]',color = 'grey')
        ax12.tick_params(axis = 'x', colors = 'grey')

        ax12.grid(True)

        # Bad Hole Flag plot
    
        ax21 = axis[1].twiny()
        ax21.plot(tvd_las['BHF'], tvd_las.index, color = 'red')
        ax21.fill_betweenx(tvd_las.index, 0, tvd_las['BHF'], color = 'red', label = 'Bad hole')
        ax21.spines['top'].set_position(('outward',0))
        ax21.set_xlabel('BHF', color = 'red')
        ax21.tick_params(axis = 'x', colors = 'red')
        ax21.legend(loc = 'upper right')
        
        ax21.grid(True)
        
        fig.tight_layout()
        
        plt.show()

        while True:
            confirm = input('Are you ok with this created bad hole flag? [Ok/Not]: ').strip()   

            if confirm.lower() == 'ok':
                las.append_curve('BHF', tvd_las['BHF'], unit = 'unitless', descr = 'Bad Hole Flag', value = '')
                break

            elif confirm.lower() == 'not':
                
                while True:
                    ci = float(input('Please type value between 0.00 - 1.00 (0.75 is default) to adjust: ').strip())
                
                    if 0 < ci < 1:
                        break

                    else:
                        print('Input value is out of range!')
                break

            else:
                print('please confirm again!')

    return las, tvd_las

# create Bad Hole flag for each well

for las, tvd_las in zip(lases, tvd_lases):
    las, tvd_las = create_bhf(las, tvd_las, alias)

"""
Quality Control (QC)
"""

# Function for initial plot for first inspection

def initial_inspection(las, tvd_las, inspect_name):
    """
    For all curve initial inspection
    las = las file (.las) of the well data
    tvd_las = well logging data in data frame (converted true vertical depth)
    """
    # Create figure

    fig, axis = plt.subplots(nrows = 1, ncols = len(tvd_las.columns), figsize = (30,20), sharey = True)
    fig.suptitle(las.well['WELL'].value, fontsize = 30, y = 1.0)
    
    units = [curve.unit for curve in las.curves]
    units.pop(0)

    # Plot setting for all axis

    for ax, col, unit in zip(axis, tvd_las.columns, units):
        ax.set_ylim(tvd_las.index.min(), tvd_las.index.max())
        ax.invert_yaxis()
        ax.minorticks_on() #Scale axis
        ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
        ax.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black')
        ax.set_xlabel(col + '\n(%s)' %unit, fontsize = 15)

        if (col in alias['RT']) or (col in alias['ILM']) or (col in alias['MSFL']):
            ax.plot(tvd_las[col], tvd_las.index)
            ax.set_xscale('log')

        elif col == 'BHF':
            ax.plot(tvd_las[col], tvd_las.index)
            ax.fill_betweenx(tvd_las.index, 0, tvd_las[col], label = 'Bad hole')
        
        else:
            ax.plot(tvd_las[col], tvd_las.index)

    fig.tight_layout()

    # save files
    inspect_folder = 'LQC_Inspect'
    inspect_path = os.path.join(sav_path, inspect_folder)

    if not os.path.isdir(inspect_path):
        os.makedirs(inspect_path)

    plt.savefig(os.path.join(inspect_path, inspect_name), dpi = 200, format = 'png')

    plt.show()

# Plot available curves

for las, tvd_las in zip(lases, tvd_lases):
    inspect_name = 'LQC_' + las.well['WELL'].value + '_Inspect' + '.png'
    initial_inspection(las, tvd_las, inspect_name)

# Function for removing some zone.

def eliminate_zone(tvd_las, alias, top_zone, bottom_zone, based_curves):
    """
    This function is for data elimination within the interested zone. 
    All curves within that zone will be removed.
    Except measured depth (MD), azimuth direction (AZIMUTH), angle or deviation (ANGLE), bitsize (BS), caliper (CAL) and bad hole flag (BHF).
    tvd_las = well logging data in pandas data frame in TVD depth.
    alias = curve alias or alterative name of the curve.
    top_zone = Top TVD depth of the zone you want to remove.
    bottom_zone = Bottom TVD depth of the zone you want to remove.
    based_curves = list of based curve names
    """
     # set data columns for elimination

    data_cols = tvd_las.columns
    base_cols = based_curves.copy()
    edit_cols = []

    for bs in alias['BS']:
        base_cols.append(bs)

    for cal in alias['CAL']:
        base_cols.append(cal)

    for col in data_cols:
        if col not in base_cols:
            edit_cols.append(col)

    # Eliminate the data within the assigned interval

    tvd_las.loc[(tvd_las.index > float(top_zone)) & (tvd_las.index < float(bottom_zone)), edit_cols] = np.nan

    return tvd_las

# Define well and zone of interest

while True:
    print('Are there any depth interval you would like to remove or delete?')
    answer = input('Please type Yes or No [Yes/No]: ').strip()

    if answer.lower() == 'yes':

        while True:
            print('There are %d wells.' %len(well_names))
            print('%s, Which one you want to edit?' %', '.join(well_names))
            name = input('Please indicate the well name you want to edit: ').strip()         
            
            if name.lower() in [name.lower() for name in well_names]:
                i = [name.lower() for name in well_names].index(name.lower())

                while True:
                    depth_min = tvd_lases[i].index.min()
                    depth_max = tvd_lases[i].index.max()
                    print('This well has data in depth from', round(depth_min, 2), 'to', round(depth_max, 2))
                    top_zone = float(input('Please indicate top depth of the zone or interval you want to edit in TVD depth: ').strip())

                    if (top_zone > depth_min) & (top_zone < depth_max):

                        while True:
                            bottom_zone = float(input('Please indicate bottom depth of the zone or interval you want to edit in TVD depth: ').strip())
                            
                            if (bottom_zone > top_zone) & (bottom_zone < depth_max):
                                
                                while True:
                                    print('The data of well', lases[i].well['WELL'].value, 'in TVD depth from', top_zone, 'to', bottom_zone, 'will be eliminated.')
                                    confirm = input('Are you sure? [Yes/No]: ').strip() 
                                    
                                    if confirm.lower() == 'yes':
                                        tvd_las = eliminate_zone(tvd_lases[i], alias, top_zone, bottom_zone, based_curves)
                                        inspect_edited_name = 'LQC_' + lases[i].well['WELL'].value + '_Inspect_Edited' + '.png'
                                        initial_inspection(lases[i], tvd_lases[i], inspect_edited_name)
                                        print('The data has been eliminated.')
                                        break
                                        
                                    elif confirm.lower() == 'no':
                                        break

                                    else:
                                        print('Please comfirm again!')
                                        continue
                                break
                                        
                            else:
                                print('Your bottom depth is out of range')
                                continue
                        break

                    else:
                        print('Your top depth is out of range')
                        continue
                break

            else:
                print('Your well %s is not found., Please select again!' %name)
                continue
        continue

    elif answer.lower() == 'no':
        print('Noted, Sir/Ma\'am!')
        break

    else:
        print('Please comfirm again!')
        continue

# Function for standardizing or renaming based on alias

def apply_alias(las, tvd_las, alias):
    """
    This function is going to rename curves based on alias. The duplicates will be named followed by the number.
    las = las file (.las) of the well data
    tvd_las = well logging data in pandas data frame in TVD depth.
    alias = curve alias or alterative name of the curve.
    """
    # get standard curve name from alias

    new_cols = {}
    seen = {}
    dupes = []

    for col in tvd_las.columns:
        for key, values in alias.items():
            
            if col in values:
                new_col = key

                if key not in seen:
                    seen[key] = 1
                
                else:
                    if seen[key] == 1:
                        dupes.append(key)

                    seen[key] += 1
                    new_col = "{}_{}".format(key, seen[key])

                new_cols[col] = new_col

    # apply to tvd_las

    tvd_las = tvd_las.rename(columns = new_cols)

    # apply to las

    for key, value in new_cols.items():
        las.curves[key].mnemonic = value

    print('All curve names of well %s are standardized' %las.well['WELL'].value)
    
    return las, tvd_las, seen, dupes

# Function for setting up the well logging data without the duplicate

def setup_curve(las, well, seen, dupes, mem_curves, based_curves):
    """
    This function will select and eliminate curve data for setting up modeling curve inputs.
    las = las file (.las) of the well data
    well = well logging data in pandas data frame in TVD depth with alias applied.
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
                select = input('Please select a curve for %s: ' %key).strip()

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

# Rename curve and setup curve for modeling

wells = []

print('The system is renaming the curve columns.')

for las, tvd_las in zip(lases, tvd_lases):
    las, well, seen, dupes = apply_alias(las, tvd_las.copy(), alias)
    las, well = setup_curve(las, well, seen, dupes, mem_curves, based_curves)
    wells.append(well)

