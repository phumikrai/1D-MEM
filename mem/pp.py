"""

5.Pore presure

"""

# Function for hydrostatic pressure calculation

def pp_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las)
    ...
    output: pore pressure
    """
    
    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')

    import numpy as np

    # calculate using normal pressure assumption (equal hydrostatic/water pressure)

    wg = 0.44 # water pressure gredient in psi/ft

    dataframe['PP'] = wg * dataframe.TVD * (3.28084) # 3.28084 for m to ft
    dataframe.loc[dataframe.PP < 0, 'PP'] = np.nan

    dataframe['PG'] = wg * (1/0.052) # 1/0.052 for psi/ft to ppg

    # update LAS file

    las.append_curve('PP', dataframe['PP'], unit = 'psi', descr = 'Pore Pressure', value = '')
    las.append_curve('PG', dataframe['PG'], unit = 'ppg', descr = 'Pore Pressure Gradient', value = '')

    return dataframe, las


