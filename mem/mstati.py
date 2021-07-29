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

    # matrix and fluid parameters

    RHOBm, NPHIm = 2.65, 0
    RHOBf, NPHIf = 1.0, 1.0

    # shale parameters

    RHOBsh = RHOB.quantile(0.55)
    NPHIsh = NPHI.quantile(0.55)

    # volume of clay computation from Neutron-Density crossplot equation (Bhuyan and Passey, 1994)

    term1 = (RHOBm-RHOBf)*(NPHI-NPHIf) - (RHOB-RHOBf)*(NPHIm-NPHIf)
    term2 = (RHOBm-RHOBf)*(NPHIsh-NPHIf) - (RHOBsh-RHOBf)*(NPHIm-NPHIf)
    dataframe['VCL'] = term1/term2

    # volume of clay from GR

    GR = dataframe.GR_NORM.dropna()
    VCLgr = (GR - GR.quantile(0.10)) / (GR.quantile(0.80) - GR.quantile(0.10))

    # replace volume of clay from GR in gas bearing formation (VCL < 0)

    dataframe.loc[dataframe.VCL < 0, 'VCL'] = VCLgr

    # limit exceeding volume of clay (VCL > 1) to maximum value (VCL = 1)

    dataframe.VCL.clip(0, 1, inplace = True)

    # update LAS file

    las.append_curve('VCL', dataframe['VCL'], unit = 'V/V', descr = 'Volume of clay', value = '')
