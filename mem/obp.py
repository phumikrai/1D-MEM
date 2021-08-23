"""

Overburden stress

"""

# function for density extrapolation

def denextra(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las),
            ml = mudline density,
            surface = surface level,
            coefs = extrapolation function coefficients
    ...
    output: extrapolated density
    """
    
    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')
    ml = kwargs.get('ml')
    surface = kwargs.get('surface')
    coefs = kwargs.get('coefs')

    # density extrapolation

    depth = dataframe.loc[dataframe.TVD >= surface, 'TVD']
    dataframe['RHOB_EX'] = ml + (coefs[0] * (depth**coefs[1]))

    # update las file
    
    unit = las.curves['RHOB_MRG'].unit
    las.append_curve('RHOB_EX', dataframe['RHOB_EX'], unit=unit, descr='Extrapolated Density', value = '')

    return dataframe, las

# function for extrapolation function coefficient

def extracoef(**kwargs):
    """
    input:  dataframes = list of well logging data in dataframe,
            mls = list of mudline density,
            surfaces = list of surface level,
    ...
    output: density extrapolation function coefficients
    """

    dataframes = kwargs.get('dataframes')
    mls = kwargs.get('mls')
    surfaces = kwargs.get('surfaces')

    import numpy as np
    from scipy.optimize import curve_fit

    # setup dictionary for storing

    points = ['A', 'B', 'C']    
    densities = {}
    depths = {}

    for point in points:
        densities[point] = []
        depths[point] = []

    for dataframe, ml, surface in zip(dataframes, mls, surfaces):
        
        # extract density

        n_data = int(round(len(dataframe.RHOB_MRG.dropna().index) * 0.05))
        B_den = dataframe.RHOB_MRG.dropna().head(n_data).quantile(0.6)
        C_den = dataframe.RHOB_MRG.dropna().head(n_data).quantile(0.7)
        dens = [ml, B_den, C_den]

        # extract depths

        B_dep = dataframe.loc[dataframe.RHOB_MRG.notna(), 'TVD'].min()
        C_dep = dataframe.loc[dataframe.RHOB_MRG.notna(), 'TVD'].max()
        deps = [surface, B_dep, C_dep]

        # data arrangement

        for point, den, dep in zip(points, dens, deps):
            densities[point].append(den)
            depths[point].append(dep)

    # density and position for each point

    train_den = tuple([np.mean(densities[point]) for point in points])
    train_dep = tuple([np.mean(depths[point]) for point in points])
    train_ml = [np.mean(densities['A']) for i in points]

    coefs, _ = curve_fit(extra_equation, (train_ml, train_dep), train_den)

    return coefs

# function for curve fitting

def extra_equation(X, A0, alpha):
    """
    input:  X = independent variables including mudline density and depth for each point,
            A0 = first fitting parameter
            alpha = second fitting parameter
    ...
    output: extrapolated density value
    """
    # independent variables

    ml, dep = X

    # density extrapolation equation

    return ml + (A0 * (dep**alpha))

# function for overburden stress calculation using integration

def obp_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las),
            surfaces = list of surface level
    ...
    output: overburden pressure
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')

    import numpy as np
    from scipy.integrate import cumtrapz
    from scipy import constants 

    # prepare density input data

    connect_depth = dataframe.loc[dataframe.RHOB_MRG.notna(), 'TVD'].min()
    dataframe['obp_input'] = dataframe.RHOB_MRG.interpolate(method='linear', limit_area='inside') # fill nan using interpolation method
    dataframe.loc[dataframe.TVD < connect_depth, 'obp_input'] = dataframe.RHOB_EX

    # input parameters

    condition = dataframe.obp_input.notna()

    density = dataframe.loc[condition, 'obp_input']
    depth = dataframe.loc[condition, 'TVD']

    # calculate OBP and OBG using integration along TVD depth (output unit in psi)

    dataframe['OBP'] = np.nan

    if 'wl' in kwargs:
        wl = kwargs.get('wl')
        wg = 0.44 # water pressure gredient in psi/ft
        obp_water = wg * wl * (3.28084) # 3.28084 for m to ft
        dataframe.loc[condition, 'OBP'] = obp_water + (constants.g * cumtrapz(density, depth, initial = 0) * 1e3 * 0.000145038) # 0.000145038 for Pa to psi
    else:
        dataframe.loc[condition, 'OBP'] = constants.g * cumtrapz(density, depth, initial = 0) * 1e3 * 0.000145038 # 0.000145038 for Pa to psi

    dataframe['OBG'] = (dataframe.OBP / dataframe.TVD) * (1/3.28084) * (1/0.052) # 3.28084 for m to ft, 1/0.052 for psi/ft to ppg
    dataframe.drop(columns=['obp_input'], inplace=True)

    # update LAS file

    las.append_curve('OBP', dataframe['OBP'], unit='psi', descr='Overburden Pressure', value='')
    las.append_curve('OBG', dataframe['OBG'], unit='ppg', descr='Overburden Gradient', value='')

    return dataframe, las