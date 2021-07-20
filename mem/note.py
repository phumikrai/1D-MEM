def announce():
    """
    output: note while starting program
    """
    note = """
    Welcome to 1D Mechanical Earth Modeling by Python

    Please take note on these;
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
        - Pressure test (.csv)
        - Core test (.csv)
        - Drilling test (.csv)
        - Mud-weight log (.csv)
        must be separated as sub-directory of the data directory. 

    For example;
        - Working directory is "Drive:/Working/".
        - All data for modeling directory is "Drive:/Working/Data/".
        - Well logging file directory is "Drive:/Working/Data/Well logging/" as Sub-directory of the data directory.
        - Deviation file directory is "Drive:/Working/Data/Deviation/" as Sub-directory of the data directory.
        - Formation top file directory is "Drive:/Working/Data/Formation top/" as Sub-directory of the data directory.
        - Pressure test file directory is "Drive:/Working/Data/Pressure test/" as Sub-directory of the data directory.
        - Core test file directory is "Drive:/Working/Data/Core test/" as Sub-directory of the data directory.
        - Drilling test file directory is "Drive:/Working/Data/Drilling test/" as Sub-directory of the data directory.
        - Mud-weight log file directory is "Drive:/Working/Data/Mud weight/" as Sub-directory of the data directory.

    4.) Well name should be set as prefix for each file. Its name will cause file ordering and file pairing for each file of that well.
        
    Assuming that; 
    well name is "Well-01" (Noted: No underscore ('_') be contained in well name) So this name should be set as prefix followed by underscore ('_') for each modeling input file like this "Well-01_(...Specific name for file type indication...)". 

    For example;
        - well logging      is   "Well-01_las"    
        - Deviation         is   "Well-01_dev"
        - Formation top     is   "Well-01_top"
        - Pressure test     is   "Well-01_pp"
        - Core test         is   "Well-01_core"
        - Drilling test     is   "Well-01_test"
        - Mud-weight log    is   "Well-01_mw"

    5.: Required data and file format;

    - Well logging files including all necessary curves for 1D MEM such; 
        Measured depth                  (MD or DEPTH)   in meter unit [m] 
        Bitsize                         (BS)            in inch unit [in] 
        Caliper                         (CAL)           in inch unit [in] 
        Gamma ray                       (GR)            in American Petroleum Institute unit [API]
        Density                         (RHOB)          in grams per cubic centimetre unit [g/c3]
        Neutron porosity                (NPHI)          in fractional unit [V/V]
        Shallow resistivity             (MSFL)          in ohm-meter unit [ohm-m]
        Deep resistivity                (RT)            in ohm-meter unit [ohm-m]
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

    - Pressure test files including; 
        Measured depth                  (DEPTH)         in meter unit [m]           
        Pressure                        (PRESSURE)      in pound per square inch unit [psi]

    - Core test files including; 
        Measured depth                  (DEPTH)         in meter unit [m]           
        Young's modulus                 (YME)           in pound per square inch unit [psi]
        Uniaxial compressive strength   (UCS)           in pound per square inch unit [psi]

    - Drilling test files including; 
        Measured depth                  (DEPTH)         in meter unit [m]           
        Test type                       (TEST)          formation test such FIT, LOT, Minifrac, and etc.
        Result or value                 (RESULT)        in mud weight unit [ppg]

    - Mud-weight log files including; 
        Measured depth                  (DEPTH)         in meter unit [m]           
        Mud weight                      (MUDWEIGHT)     in mud weight unit [ppg]
        Equivalent Circulating Density  (ECD)           in mud weight unit [ppg]
    """
    return print(note)

