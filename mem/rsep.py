"""

6.Rock Strength & Elastic Properties

"""

# function for Young's modulus computation

def yme_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las)
    ...
    output: Young's modulus 
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')

    from mem.rsepeq import syme_sand, syme_shale

    # calculate dynamic Young's modulus

    dataframe['DYME'] = dyme_eq(RHOB=dataframe.RHOB_MRG, DTC=dataframe.DTC_MRG, DTS=dataframe.DTS_MRG)

    # calculate static Young's modulus

    for litho in ['SAND', 'SHALE']:
        
        # prepare input parameters

        Rhob = dataframe.loc[dataframe.LITHO == litho, 'RHOB_MRG'].dropna()
        dt = dataframe.loc[dataframe.LITHO == litho, 'DTC_MRG'].dropna()
        Vp = (1/dt)
        Phie = dataframe.loc[dataframe.LITHO == litho, 'PHIE'].dropna()
        Dyme = dataframe.loc[dataframe.LITHO == litho, 'DYME'].dropna()

        # unit conversion for empirical equations

        Dyme *= 6.8948e-6 # 6.8948e-6 for psi unit to GPa unit
        Vp *= 304.800 # 304.800 for ft/us to km/s

        # calculation

        if litho == 'SAND':
            Syme = syme_sand(Rhob=Rhob, dt=dt, Vp=Vp, Phie=Phie, Dyme=Dyme)
            dataframe['YME'] = Syme
        else:
            Syme = syme_shale(Rhob=Rhob, dt=dt, Vp=Vp, Phie=Phie, Dyme=Dyme)
            dataframe['YME'].fillna(Syme, inplace = True)

    # unit conversion

    dataframe['YME'] *= 145038 # 145038 for GPa unit to psi unit

    # update LAS file

    las.append_curve('DYME', dataframe['DYME'], unit = 'psi', descr = 'Dynamic Young\'s modulus', value = '')
    las.append_curve('YME', dataframe['YME'], unit = 'psi', descr = 'Static Young\'s modulus', value = '')

    return dataframe, las

# dynamic Young's modulus equation

def dyme_eq(**kwargs):
    """
    input:  RHOB = density in grams per cubic centimetre unit [g/c3]
            DTC = slowness in microseconds per foot unit [us/ft],
            DTS = slowness in microseconds per foot unit [us/ft]
    ...
    output: dynamic Young's modulus in pound per square inch unit [psi]
    """

    RHOB = kwargs.get('RHOB')
    DTC = kwargs.get('DTC')
    DTS = kwargs.get('DTS')

    # convert slowness (us/ft) to velosity (m/s)

    Vp = (1/DTC) * 304800 # 304800 for ft/us to m/s
    Vs = (1/DTS) * 304800

    # compute dynamic Young's modulus (psi)

    term1 = (3*(Vp**2)) - (4*(Vs**2))
    term2 = (Vp**2) - (Vs**2)

    DYME = RHOB * (Vs**2) * (term1/term2) * 0.145038 # 0.145038 for conversion factor

    return DYME

# function for Poisson's ratio computation

def pr_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las)
    ...
    output: Poisson's ratio
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')

    # calculate dynamic Poisson's ratio

    dataframe['DPR'] = dpr_eq(DTC=dataframe.DTC_MRG, DTS=dataframe.DTS_MRG)

    # calculate static Poisson's ratio

    factor = 1.0

    dataframe['PR'] = factor * dataframe.DPR

    # update LAS file

    las.append_curve('DPR', dataframe['DPR'], unit = 'unitless', descr = 'Dynamic Poisson\'s ratio', value = '')
    las.append_curve('PR', dataframe['PR'], unit = 'unitless', descr = 'Static Poisson\'s ratio', value = '')

    return dataframe, las

# dynamic Possion's ratio equation

def dpr_eq(**kwargs):
    """
    input:  DTC = slowness in microseconds per foot unit [us/ft],
            DTS = slowness in microseconds per foot unit [us/ft]
    ...
    output: dynamic Poisson's ratio
    """

    DTC = kwargs.get('DTC')
    DTS = kwargs.get('DTS')

    # convert slowness (us/ft) to velosity (m/s)

    Vp = (1/DTC) * 304800 # 304800 for ft/us to m/s
    Vs = (1/DTS) * 304800

    # compute dynamic Poisson's ratio

    term1 = (Vp**2) - (2*(Vs**2))
    term2 = (Vp**2) - (Vs**2)

    DPR = 0.5 * (term1/term2)

    return DPR

# function for UCS computation

def ucs_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las)
    ...
    output: unconfined compressive strength (UCS)
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')

    from mem.rsepeq import ucs_sand, ucs_shale

    # calculate unconfined compressive strength

    for litho in ['SAND', 'SHALE']:
        
        # prepare input parameters

        Rhob = dataframe.loc[dataframe.LITHO == litho, 'RHOB_MRG'].dropna()
        dt = dataframe.loc[dataframe.LITHO == litho, 'DTC_MRG'].dropna()
        Vp = (1/dt)
        Phie = dataframe.loc[dataframe.LITHO == litho, 'PHIE'].dropna()
        Syme = dataframe.loc[dataframe.LITHO == litho, 'YME'].dropna()

        # unit conversion for empirical equations

        Syme *= 6.8948e-6 # 6.8948e-6 for psi unit to GPa unit
        Vp *= 304.800 # 304.800 for ft/us to km/s

        # calculation

        if litho == 'SAND':
            UCS = ucs_sand(Rhob=Rhob, dt=dt, Vp=Vp, Phie=Phie, Syme=Syme)
            dataframe['UCS'] = UCS
        else:
            UCS = ucs_shale(Rhob=Rhob, dt=dt, Vp=Vp, Phie=Phie, Syme=Syme)
            dataframe['UCS'].fillna(UCS, inplace = True)

    # unit conversion

    dataframe['UCS'] *= 145.038 # 145.038 for MPa unit to psi unit

    # update LAS file

    las.append_curve('UCS', dataframe['UCS'], unit = 'psi', descr = 'Unconfined compressive strength', value = '')

    return dataframe, las

# function for friction angle computation 

def fang_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las)
    ...
    output: friction angle
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')

    # calculate friction angle using linear correlation method with volume of clay
    # 100% shale relate to 20 degree friction angle
    # 100% sand relate to 40 degree friction angle

    dataframe['FANG'] = ((1 - dataframe.VCL) + 1) * 20

    # update LAS file

    las.append_curve('FANG', dataframe['FANG'], unit = 'degree', descr = 'Angle of internal friction', value = '')

    return dataframe, las

# function for tensile strength computation

def tstr_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las)
    ...
    output: tensile strength
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')

    # calculate tensile strength

    dataframe['TSTR'] = dataframe.UCS * 0.1

    # update LAS file

    las.append_curve('TSTR', dataframe['TSTR'], unit = 'psi', descr = 'Tensile formation strength', value = '')

    return dataframe, las
