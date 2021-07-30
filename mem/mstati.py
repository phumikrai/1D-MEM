"""

Mechanical Stratigraphy

"""

# function for volume of clay calculation

def vcl_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las)
    ...
    output: volume of clay
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')

    # input parameters

    RHOB = dataframe.RHOB_MRG.dropna()
    NPHI = dataframe.NPHI_MRG.dropna()
    GR = dataframe.GR_NORM.dropna()

    # matrix and fluid parameters

    RHOBm, RHOBf = 2.65, 1.0 # density
    NPHIm, NPHIf = 0, 1.0 # neutron porosity

    # shale parameters

    RHOBsh = RHOB.quantile(0.55)
    NPHIsh = NPHI.quantile(0.55)

    # volume of clay computation from Neutron-Density crossplot equation (Bhuyan and Passey, 1994)

    term1 = (RHOBm-RHOBf)*(NPHI-NPHIf) - (RHOB-RHOBf)*(NPHIm-NPHIf)
    term2 = (RHOBm-RHOBf)*(NPHIsh-NPHIf) - (RHOBsh-RHOBf)*(NPHIm-NPHIf)
    dataframe['VCL'] = term1/term2

    # volume of clay from GR

    VCLgr = (GR - GR.quantile(0.10)) / (GR.quantile(0.80) - GR.quantile(0.10))

    # replace volume of clay from GR in hydrocarbon zone (VCL < 0)

    dataframe.loc[dataframe.VCL < 0, 'VCL'] = VCLgr

    # limit exceeding volume of clay (VCL > 1) to maximum value (VCL = 1)

    dataframe.VCL.clip(0, 1, inplace = True)

    # update LAS file

    las.append_curve('VCL', dataframe['VCL'], unit='V/V', descr='Volume of clay', value='')

    return dataframe, las

# function for effective porosity calculation

def phie_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las)
    ...
    output: effective porosity
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')

    # input parameters

    RHOB = dataframe.RHOB_MRG.dropna()
    NPHI = dataframe.NPHI_MRG.dropna()
    VCL = dataframe.VCL.dropna()

    # matrix and fluid parameters

    RHOBm, RHOBf = 2.65, 1.0

    # shale parameters

    RHOBsh = RHOB.quantile(0.55)
    NPHIsh = NPHI.quantile(0.55)

    # density porosity computation with shale correction

    DPHI = (RHOBm - RHOB) / (RHOBm - RHOBf)
    DPHIsh = (RHOBm - RHOBsh) / (RHOBm - RHOBf)
    DPHIshcor = DPHI - (VCL * DPHIsh)

    # neutron porosity with shale correction

    NPHIshcor = NPHI - (VCL * NPHIsh)

    # total porosity

    POR = (NPHIshcor + DPHIshcor)/2

    # effective porosity

    dataframe['PHIE'] = POR * (1 - VCL)

    # update LAS file

    las.append_curve('PHIE', dataframe['PHIE'], unit='V/V', descr='Effective Porosity', value='')

    return dataframe, las

# function for lithology prediction using gamma ray cut-off

def litho_cal(**kwargs):
    """
    input:  dataframe = well logging data in dataframe,
            las = well logging data in las file (.las)
    ...
    output: lithology either sandstone or shale
    """

    dataframe = kwargs.get('dataframe')
    las = kwargs.get('las')

    import pandas as pd

    # define sand-shale cut off from volume of clay (VCL)

    cutoff = 0.4
    dataframe['LITHO'] = pd.cut(dataframe.VCL, bins=[0, cutoff, 1], labels=['SAND', 'SHALE'])

    # update LAS file

    las.append_curve('LITHO', dataframe['LITHO'], unit='unitless', descr='Lithology', value='')

    return dataframe, las