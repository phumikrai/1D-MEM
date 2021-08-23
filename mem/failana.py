"""

Failure analysis

"""

# function for transfering the mud weight data to well logging data

def merge_mud(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las),
            mud = mud-weight data in dataframe
    ...
    output: well logging data merged with mud-weight data
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')
    mud = kwargs.get('mud')

    from pandas import DataFrame

    # merge mud-weight data

    cols = ['MW', 'ECD']
    descrs = ['Mud weight', 'Equivalent circulating density']
    
    for col, descr in zip(cols, descrs):
        dataframe[col] = mud.set_index('MD')[col]
        dataframe[col].interpolate(method = 'linear', limit_area = 'inside', inplace = True)

        # update LAS file

        las.append_curve(col, dataframe[col], unit = 'ppg', descr=descr, value='')

    return dataframe, las

# function for mud window component calculation

def mudwindow_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las)
    ...
    output: mud window components including;
            1. Kick mud weight
            2. Breakout mud weight or minimum mud weight
            3. Loss mud weight
            4. Breakdown mud weight or maximum mud weight
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')

    # calculate kick mud weight

    dataframe['CMW_KICK'] = dataframe.PG

    # Breakout or minimum mud weight caluculation

    min_mw = (3 * dataframe.SHmax) - dataframe.Shmin - dataframe.PP - dataframe.UCS
    dataframe['CMW_MIN_MC'] = (min_mw/dataframe.TVD) * (1/3.28084) * (1/0.052) # 3.28084 for m to ft, 1/0.052 for psi/ft to ppg

    # Loss mud weight calculation

    dataframe['CMW_LOSS'] = (dataframe.Shmin/dataframe.TVD) * (1/3.28084) * (1/0.052) # 3.28084 for m to ft, 1/0.052 for psi/ft to ppg

    # Breakdown or maximum mud weight caluculation

    max_mw = (3 * dataframe.Shmin) - dataframe.SHmax - dataframe.PP + dataframe.TSTR
    dataframe['CMW_MAX_MTS'] = (max_mw/dataframe.TVD) * (1/3.28084) * (1/0.052) # 3.28084 for m to ft, 1/0.052 for psi/ft to ppg

    # update LAS file

    cols = ['CMW_KICK', 'CMW_MIN_MC', 'CMW_LOSS', 'CMW_MAX_MTS']
    descrs = ['Kick', 'Breakout or minimum', 'Loss', 'Breakdown or maximum']
    
    for col, descr in zip(cols, descrs):
        las.append_curve(col, dataframe[col], unit='ppg', descr='%s mud weight' %descr, value='')

    return dataframe, las


# function for breakout width calculation

def wbo_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las)
    ...
    output: breakout width in degree
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')

    import numpy as np

    # mud weight to mud pressure

    pressure = (dataframe.MW * 0.052) * (dataframe.TVD * 3.28084)  # 0.052 for ppg to psi/ft, 3.28084 for m to ft

    # ignore RuntimeWarning due to np.nan in input

    with np.errstate(invalid='ignore'):

        # breakout width caluculation

        term1 = dataframe.SHmax + dataframe.Shmin - pressure - dataframe.PP - dataframe.UCS
        term2 = 2 * (dataframe.SHmax - dataframe.Shmin)
        cos_2 = term1/term2
        dataframe['WBO'] = np.rad2deg(np.pi - np.arccos(cos_2))

    # update LAS file

    las.append_curve('WBO', dataframe['WBO'], unit = 'degree', descr = 'Breakout width', value = '')

    return dataframe, las