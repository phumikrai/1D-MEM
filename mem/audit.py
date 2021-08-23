"""

Data audit

"""

# function for decision confirmation

from os import name
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
    input:  dataframe = well logging data in dataframe,
            toprange = start logging point,
            step = step or rate of logging
    ...
    output: extended dataframe (to measured depth = 0) filled with nan
    """

    dataframe = kwargs.get('dataframe')
    toprange = kwargs.get('toprange')
    step = kwargs.get('step')

    import pandas as pd
    import numpy as np

    # create empty dataframe
    
    ex_depth = pd.DataFrame(np.arange(0, toprange, round(step, 2)), columns=['MD'])

    # merge with well logging data

    dataframe.reset_index(inplace=True)
    dataframe = pd.concat([ex_depth, dataframe]).sort_values(by=['MD'])
    dataframe.set_index('MD', inplace=True)

    return dataframe

# function for transfering the deviation data to another dataframe

def merge_dev(**kwargs):
    """
    input:  dataframe = data in dataframe contains MD column,
            dev = deviation data in dataframe
    ...
    output: merged data with deviation data, the azimuth (AZI) and angle (ANG) data are included.
    """
    
    dataframe = kwargs.get('dataframe')
    dev = kwargs.get('dev')

    import pandas as pd

    # merge deviation file

    dataframe = pd.concat([dev, dataframe]).sort_values(by=['MD'])
    dataframe = dataframe.groupby('MD').max()

    # fill-in data using linear interpolation

    for col in ['AZI', 'ANG']:
        dataframe[col].interpolate(method='linear', limit_area='inside', inplace=True)
    dataframe.reset_index(inplace=True)

    return dataframe

# function for True Vertical Depth (TVD) computation by minimum curvature method

def mini_cuv(**kwargs):
    """
    input:  dataframe = data in dataframe contains azimuth (AZI) and angle (ANG),
            ag = air gap from field parameter determination
    ...
    output: true vertical depth
    """

    dataframe = kwargs.get('dataframe')
    ag = kwargs.get('ag')

    import numpy as np

    # ignore RuntimeWarning due to np.nan in input

    with np.errstate(invalid='ignore'):

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
    input:  dataframe = well logging data in dataframe
    ...
    output: arranged dataframe of well logging 
    """

    dataframe = kwargs.get('dataframe')

    cols = dataframe.columns.tolist()
    cols = cols[:1] + cols[-2:] + cols[1:-2]
    dataframe = dataframe[cols]
    dataframe.set_index('MD', inplace=True)

    return dataframe

# function for formation top alignment

def setuptop(**kwargs):
    """
    input:  dataframe = formation top data in dataframe
    ...
    output: setup formation top data
    """

    dataframe = kwargs.get('dataframe')

    last_tvd = dataframe.TVD.max()
    last_tvdss = dataframe.TVDSS.max()

    dataframe.dropna(inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    dataframe.rename(columns = {'TVD':'TVD_TOP', 'TVDSS':'TVDSS_TOP'}, inplace=True)

    dataframe['TVD_BOTTOM'] = dataframe['TVD_TOP'].shift(periods = -1)
    dataframe['TVDSS_BOTTOM'] = dataframe['TVDSS_TOP'].shift(periods = -1)

    dataframe.fillna(value = {'TVD_BOTTOM': last_tvd, 'TVDSS_BOTTOM': last_tvdss}, inplace=True)

    return dataframe

"""

Bad zone elimination

"""

# function for calculate bad hole flag

def bhf_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las),
            cif = confidential interval factor (0.00-1.00, default = 0.75)
    ...
    output: bad hole flag and cut-off interval
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')
    cif = kwargs.get('cif')

    import numpy as np
    import scipy.stats as st

    # calculate confidential interval
    
    diff = dataframe.CAL - dataframe.BS
    interval = st.norm.interval(alpha = cif, loc = round(np.mean(diff), 2), scale = round(np.std(diff), 2))

    # apply interval as BHF cut-off

    condition1 = (diff < interval[0]) | (diff > interval[1])
    condition2 = (diff >= interval[0]) & (diff <= interval[1])

    dataframe['BHF'] = np.nan
    dataframe['BHF'].loc[condition1] = 'BAD'
    dataframe['BHF'].loc[condition2] = 'GOOD'

    # update LAS file

    las.append_curve('BHF', dataframe['BHF'], unit='unitless', descr='Bad Hole Flag', value='')

    return dataframe, las, interval

# function for eliminating bad data using bad hole flag

def bhf_control(**kwargs):
    """
    input:  dataframe = well logging data in dataframe
    ...
    output: well logging data replaced with nan at bad hole flag
    """

    dataframe = kwargs.get('dataframe')

    import numpy as np

    # eliminate the data based on bad hole flag

    applied = ['GR', 'RHOB', 'NPHI', 'MSFL', 'RT', 'DTC', 'DTS']
    dataframe.loc[dataframe.BHF == 'BAD', applied] = np.nan
        
    return dataframe

# function for eliminating low Vp/Vs ratio data

def ratio_control(**kwargs):
    """
    input:  dataframe = well logging data in dataframe
    ...
    output: well logging data replaced with nan at low Vp/Vs ratio
    """

    dataframe = kwargs.get('dataframe')

    import numpy as np

    # eliminate the data based on Vp/Vs ratio

    dataframe.loc[dataframe.DTS/dataframe.DTC < 1.6, ['DTC', 'DTS']] = np.nan
    
    return dataframe

"""

Data normalization

"""

# function for gamma ray normalization using squeeze and stretch method

def norm_gr(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las),
            ref_high = reference GR max value,
            ref_low = reference GR min value
    ...
    output: normalized gamma ray log
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')
    ref_high = kwargs.get('ref_high')
    ref_low = kwargs.get('ref_low')

    # normolize gamma ray curve

    q95 = dataframe.GR.quantile(0.95)
    q05 = dataframe.GR.quantile(0.05)
    log = dataframe.GR

    norm = (log - q05) / (q95 - q05)
    dataframe['GR_NORM'] = ref_low + (ref_high - ref_low) * norm

    # update las file

    las.append_curve('GR_NORM', dataframe.GR_NORM, unit='API', descr='Normalized Gamma Ray', value='')

    return dataframe, las

"""

Data systhetic

"""

# function for dataset creation

def set_data(**kwargs):
    """
    input:  dataframes = list of well logging data in dataframe
    ...
    output: dataset for data training
    """
    
    dataframes = kwargs.get('dataframes')

    import pandas as pd

    # create dataset from all logging data

    cols = ['GR_NORM', 'MSFL', 'RT', 'NPHI', 'RHOB', 'DTC', 'DTS']
    dataset = pd.DataFrame() # an empty dataframe
    for dataframe in dataframes:
        dataset = pd.concat([dataset, dataframe[cols]])
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    # window ratio (window/total number of the data points)

    w_ratio = 0.0005
    w_length = int(round(w_ratio * len(dataset.index)))

    # noise filter using moving/running average

    for col in cols:
        dataset[col] = dataset[col].rolling(window=w_length).mean()
    dataset.dropna(inplace = True)

    return dataset

# function for data synthesis

def synthesis(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las),
            dataset = data for training
    ...
    output: dataset for data training
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')
    dataset = kwargs.get('dataset')

    import pandas as pd
    from sklearn.model_selection import train_test_split

    # test_size = size of test data for modeling (0.00 - 1.00, default = 0.3)
    
    test_size = 0.3

    # define initial and synthesized data

    initial = ['GR_NORM', 'MSFL', 'RT']
    syns = ['NPHI', 'RHOB', 'DTC', 'DTS']
    curvenames  = ['neutron porosity', 'density', 'compressional slowness', 'shear slowness']
    units = ['V/V', 'g/c3', 'us/ft', 'us/ft']
    prefixs = ['Synthetic ', 'Merged ']
    cols = initial.copy()

    # synthesize data one at the time

    for syn, name, unit in zip(syns, curvenames, units):

        curves = [syn+'_SYN', syn+'_MRG']
        pred_outputs = {}
        
        # split the training data
        
        input_train = dataset[initial]
        output_train = dataset[syn]
        x_train, x_test, y_train, y_test = train_test_split(input_train, output_train, test_size = test_size, random_state = 0)

        # Setup synthesizing input

        pred_input = dataframe[cols].dropna()

        # multi-linear regression

        mlr_output, mlr_r2 = multilinear(pred_input=pred_input, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        pred_outputs['Multi-linear Regression'] = [mlr_output, mlr_r2]

        # random forest regression

        rfr_output, rfr_r2 = randomforest(pred_input=pred_input, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        pred_outputs['Random Forest Regression'] = [rfr_output, rfr_r2]

        # decision tree regression

        dtr_output, dtr_r2 = decisiontree(pred_input=pred_input, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        pred_outputs['Decision Tree Regression'] = [dtr_output, dtr_r2]

        # select the best regression

        best_output, method, best_r2 = best_pred(pred_outputs=pred_outputs)
        dataframe[curves[0]] = pd.DataFrame(best_output, index = pred_input.index)
        
        # merge the synthesis

        dataframe[curves[1]] = dataframe[syn].fillna(dataframe[curves[0]])
                
        # setup new initial curves
        
        initial.append(syn)
        cols.append(curves[0])

        # update las file
        
        for curve, prefix in zip(curves, prefixs):
            las.append_curve(curve, dataframe[curve], unit=unit, descr=prefix+name, value = '')
            
        # announce to user

        corr_value = dataframe[syn].corr(dataframe[syn + '_SYN'])
        print('%s log is synthesized using %s' %(syn, method))
        print('R-squared value = %.2f, Correlation = %.2f' %(best_r2, corr_value))
        
    return dataframe, las

# function for multi-linear regression

def multilinear(**kwargs):
    """
    input:  pred_input = input for prediction
            x_train = training inputs
            x_test = test inputs
            y_train = training output
            y_test = test output
    """
    
    pred_input = kwargs.get('pred_input')
    x_train = kwargs.get('x_train')
    x_test = kwargs.get('x_test')
    y_train = kwargs.get('y_train')
    y_test = kwargs.get('y_test')
        
    from sklearn.linear_model import LinearRegression

    # prediction

    model = LinearRegression()
    model.fit(x_train, y_train)
    r_square = model.score(x_test, y_test)
    pred_output = model.predict(pred_input)

    return pred_output, r_square


# function for random forest regression

def randomforest(**kwargs):
    """
    input:  pred_input = input for prediction
            x_train = training inputs
            x_test = test inputs
            y_train = training output
            y_test = test output
    """
    
    pred_input = kwargs.get('pred_input')
    x_train = kwargs.get('x_train')
    x_test = kwargs.get('x_test')
    y_train = kwargs.get('y_train')
    y_test = kwargs.get('y_test')

    from sklearn.ensemble import RandomForestRegressor

    # n_tree = number of decision tree in random forest regression technique (default = 10)

    n_tree = 10

    # prediction

    model = RandomForestRegressor(n_estimators = n_tree)
    model.fit(x_train, y_train)
    r_square = model.score(x_test, y_test)
    pred_output = model.predict(pred_input)

    return pred_output, r_square

# function for decision tree regression

def decisiontree(**kwargs):
    """
    input:  pred_input = input for prediction
            x_train = training inputs
            x_test = test inputs
            y_train = training output
            y_test = test output
    """
    
    pred_input = kwargs.get('pred_input')
    x_train = kwargs.get('x_train')
    x_test = kwargs.get('x_test')
    y_train = kwargs.get('y_train')
    y_test = kwargs.get('y_test')

    from sklearn.tree import DecisionTreeRegressor

    # max_depth = the maximum depth of tree

    max_depth = 10

    # prediction

    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(x_train, y_train)
    r_square = model.score(x_test, y_test)
    pred_output = model.predict(pred_input)

    return pred_output, r_square

# function for selecting the best regression by r-square

def best_pred(**kwargs):
    """
    input:  pred_outputs = dictionary of regression method containing predicted output and r-square
    ...
    output: best predicted output
    """

    pred_outputs = kwargs.get('pred_outputs')

    # selection

    best_output = None
    method = None
    best_r2 = 0

    for output in pred_outputs:
        if  pred_outputs[output][1] > best_r2:
            best_output = pred_outputs[output][0]
            best_r2 = pred_outputs[output][1]
            method = output
            
    return best_output, method, best_r2