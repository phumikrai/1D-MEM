"""

Minimum & maximum horizontal stresses

"""

# function for maximum and minimum tectonic strain calculation

from scipy.sparse import data
from scipy.stats.stats import DescribeResult


def strain_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            drill = drilling test data (.csv)
    ...
    output: maximum and minimum tectonic strains
    """

    dataframe = kwargs.get('dataframe')
    drill = kwargs.get('drill')

    # input parameters

    cols = ['OBP', 'PP', 'YME', 'PR']
    dataset = dataframe[cols].dropna().copy()
    ndata = int(round(len(dataset.index) * 0.1))
    drill['PRESSURE'] = (drill.VAL * 0.052) * (drill.TVD * 3.28084)  # 0.052 for ppg to psi/ft, 3.28084 for m to ft

    # iterative parameters

    maxstrain, minstrain = 0, 0 # maximum and minimum tectonic strains
    increrate = 0.00001 # increment rate

    # calculate tectonic strains using iteration process

    for row in drill.iterrows():
        strtidx, endidx = idxbound(idxdep=dataset.index, middep=row[1].MD, ndata=ndata)
        selected_data = dataset.iloc[strtidx:endidx]
        print('Minimum horizontal stress from %s at depth %.2f (TVD) is %s (psi)' %(row[1].TYPE, row[1].TVD, row[1].PRESSURE))
        stress = stress_pred(dataframe=selected_data, maxstrain=maxstrain, minstrain=minstrain, depth=row[1].MD)

        while stress <= row[1].PRESSURE if row[1].TYPE == 'FIT' else stress < row[1].PRESSURE:
            maxstrain += increrate
            minstrain += increrate
            stress = stress_pred(dataframe=selected_data, maxstrain=maxstrain, minstrain=minstrain, depth=row[1].MD)

        maxstrain -= increrate
        minstrain -= increrate
        stress = stress_pred(dataframe=selected_data, maxstrain=maxstrain, minstrain=minstrain, depth=row[1].MD)

        while stress <= row[1].PRESSURE if row[1].TYPE == 'FIT' else stress < row[1].PRESSURE:
            maxstrain += increrate
            stress = stress_pred(dataframe=selected_data, maxstrain=maxstrain, minstrain=minstrain, depth=row[1].MD)
        
        print('')

    maxstrain, minstrain = [0 if strain < 0 else strain for strain in [maxstrain, minstrain]]
    
    return maxstrain, minstrain

# function for minimum horizontal stress prediction

def stress_pred(**kwargs):
    """
    input:  dataframe = selected well logging data in dataframe,
            maxstrain = maximum tectonic strain,
            minstrain = minimum tectonic strain,
            depth = depth of prediction
    ...
    output: predicted minimum horizontal stress
    """

    dataframe = kwargs.get('dataframe')
    maxstrain = kwargs.get('maxstrain')
    minstrain = kwargs.get('minstrain')
    depth = kwargs.get('depth')

    from sklearn.linear_model import LinearRegression
    from numpy import array

    # calculate minimum horizontal stress

    stress_output = stress_eq(OBP=dataframe.OBP, PP=dataframe.PP, YME=dataframe.YME, 
                                PR=dataframe.PR, alpha=1, strain1=minstrain, strain2=maxstrain)

    # linear regression at prediction depth

    X_data = dataframe.index.values.reshape(-1,1) # reshape for model prediction
    Y_data = stress_output.values.reshape(-1,1) # reshape for model prediction
    X_pred = array([depth]).reshape((-1, 1))

    lr = LinearRegression()
    lr.fit(X_data, Y_data)
    stress = lr.predict(X_pred).item(0)

    return stress

# function for horizontal stress equation

def stress_eq(**kwargs):
    """
    input:  OBP     = overburden stress [psi],
            PP      = pore pressure [psi],
            YME     = Young's modulus [psi],
            PR      = Poisson's ratio,
            alpha   = Biot's constant (equal to one),
            strain1 = minimum tectonic strain for minimum horizontal stress calculation
                        / maximum tectonic strain for maximum horizontal stress calculation,
            strain2 = maximum tectonic strain for minimum horizontal stress calculation
                        / minimum tectonic strain for maximum horizontal stress calculation 
    ...
    output: horizontal stress in pound per square inch unit [psi]
    """

    OBP = kwargs.get('OBP')
    PP = kwargs.get('PP')
    YME = kwargs.get('YME')
    PR = kwargs.get('PR')
    alpha = kwargs.get('alpha')
    strain1 = kwargs.get('strain1')
    strain2 = kwargs.get('strain2')

    # compute horizontal stress

    tectonic = (YME / (1 - (PR**2))) * (strain1 + (PR * strain2))
    stress = ((PR / (1 - PR)) * (OBP - (alpha * PP))) + (alpha * PP) + tectonic

    return stress

# function for extracting indices of data boundary

def idxbound(**kwargs):
    """
    input:  idxdep = indexed depth (MD) of well logging data in dataframe,
            middep = middle depth,
            ndata = a number of data
    ...
    output: indices of data boundary
    """

    idxdep = kwargs.get('idxdep')
    middep = kwargs.get('middep')
    ndata = kwargs.get('ndata')

    from numpy import searchsorted

    endidx = int(searchsorted(idxdep, middep) + (ndata/2))
    strtidx = int(searchsorted(idxdep, middep) - (ndata/2))
    strtidx, endidx = [0 if idx < 0 else idx for idx in [strtidx, endidx]]

    return strtidx, endidx

# function for maximum and minimum horizontal stresses calculation

def stress_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las),
            maxstrain = maximum tectonic strain,
            minstrain = minimum tectonic strain
    ...
    output: maximum and minimum horizontal stresses
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')
    maxstrain = kwargs.get('maxstrain')
    minstrain = kwargs.get('minstrain')

    # calculate minimum and maximum horizontal stresses

    cols = ['SHmax', 'Shmin']
    descrs = ['Maximum', 'Minimum']

    for col, descr in zip(cols, descrs):
        if col == 'SHmax':
            strain1, strain2 = maxstrain, minstrain
        else:
            strain1, strain2 = minstrain, maxstrain

        dataframe[col] = stress_eq(OBP=dataframe.OBP, PP=dataframe.PP, YME=dataframe.YME, 
                                    PR=dataframe.PR, alpha=1, strain1=strain1, strain2=strain2)

        # update LAS file

        las.append_curve(col, dataframe[col], unit = 'psi', descr = '%s horizontal stress' %descr, value = '')

    return dataframe, las