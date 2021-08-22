"""

** This file is for equation customization **

Customizable equations:
1. Static Young's modulus
2. Unconfined compressive strength [UCS]

...
Math operation:
+           = plus [addition]
-           = minus [subtraction]
/           = obelus [division]
*           = times [multiplication]
**          = power [Exponentiation]
np.log10(x) = logarithm base 10 of X variable

...
Variables:
Rhob        = density                           in grams per cubic centimetre unit [g/c3]
dt          = compressional wave slowness       in in microseconds per foot unit [us/ft]
Vp          = compressional wave velocity       in kilometer per second [km/s]
Phie        = effective porosity
Dyme        = dynamic Young's modulus           in gigapascal [GPa]
Syme        = static Young's modulus            in gigapascal [GPa]
Dpr         = dynamic Poisson's ratio
Spr         = static Poisson's ratio
UCS         = Unconfined compressive strength   in megapascal [MPa]
"""

# static Young's modulus for sandstone

def syme_sand(**kwargs):
    """
    input:  Rhob    = density            
            dt      = compressional wave slowness
            Vp      = compressional wave velocity
            Phie    = effective porosity
            Dyme    = dynamic Young's modulus              
    ...
    output: static Young's modulus (Syme) in gigapascal [GPa]
    """

    Rhob = kwargs.get('Rhob')
    dt = kwargs.get('dt')
    Vp = kwargs.get('Vp')
    Phie = kwargs.get('Phie')
    Dyme = kwargs.get('Dyme')

    import numpy as np

    """
    Customizable zone is below.
    ...
    Recommended equations for sandstone:

    1. Eissa and Kazi (1988);
        Syme = (0.74 * Dyme) - 0.82

    2. Eissa and Kazi (1988);
        Syme = 10**(0.02 + (0.7 * (np.log10(Rhob * Dyme))))

    3. Lacy (1997);
        Syme = (0.018 * (Dyme**2)) + (0.422 * Dyme)
    
    4. Test equation by Kanitthorn (P'Toh) (2020);
        Syme = ((-2.21 * Phie) + 0.965) * Dyme
    """

    # customizable equation

    Syme = Dyme * 0.8

    """
    End of customizable zone
    """
    return Syme

# static Young's modulus for shale

def syme_shale(**kwargs):
    """
    input:  Rhob    = density            
            dt      = compressional wave slowness
            Vp      = compressional wave velocity
            Phie    = effective porosity
            Dyme    = dynamic Young's modulus              
    ...
    output: static Young's modulus (Syme) in gigapascal [GPa]
    """

    Rhob = kwargs.get('Rhob')
    dt = kwargs.get('dt')
    Vp = kwargs.get('Vp')
    Phie = kwargs.get('Phie')
    Dyme = kwargs.get('Dyme')

    import numpy as np

    """
    Customizable zone is below.
    ...
    Recommended equations for shale:

    1. Ohen (2003);
        Syme = 0.0158 * (Dyme**2.74)
    
    2. Horsrud (2001);
        Syme = 0.076 * (Vp**3.23)
    """

    # customizable equation

    Syme = Dyme * 0.8

    """
    End of customizable zone
    """
    return Syme

# Unconfined compressive strength (UCS) for sandstone

def ucs_sand(**kwargs):
    """
    input:  Rhob    = density            
            dt      = compressional wave slowness
            Vp      = compressional wave velocity
            Phie    = effective porosity
            Syme    = static Young's modulus              
    ...
    output: Unconfined compressive strength (UCS) in megapascal [MPa]
    """

    Rhob = kwargs.get('Rhob')
    dt = kwargs.get('dt')
    Vp = kwargs.get('Vp')
    Phie = kwargs.get('Phie')
    Syme = kwargs.get('Syme')

    import numpy as np

    """
    Customizable zone is below.
    ...
    Recommended equations for sandstone:

    1. Bradford (1998);
        UCS = 2.28 + (4.0189 * Syme)

    2. Vernik (1993);
        UCS = 254 * ((1 - (2.7 * Phie))**2)

    3. Lacy (1997);
        UCS = (0.278 * (Syme**2)) + (2.458 * Syme) 
    """

    # customizable equation

    UCS = (2.28 + (4.0189 * Syme)) * 0.7

    """
    End of customizable zone
    """
    return UCS

# Unconfined compressive strength (UCS) for shale

def ucs_shale(**kwargs):
    """
    input:  Rhob    = density            
            dt      = compressional wave slowness
            Vp      = compressional wave velocity
            Phie    = effective porosity
            Syme    = static Young's modulus              
    ...
    output: Unconfined compressive strength (UCS) in megapascal [MPa]
    """

    Rhob = kwargs.get('Rhob')
    dt = kwargs.get('dt')
    Vp = kwargs.get('Vp')
    Phie = kwargs.get('Phie')
    Syme = kwargs.get('Syme')

    import numpy as np

    """
    Customizable zone is below.
    ...
    Recommended equations for shale:

    1. Horsrud (2001);
        UCS = 1.35 * (Vp**2.6)
    
    2. Horsrud (2001);
        UCS = 7.22 * (Syme**0.712)
    """

    # customizable equation

    UCS = (7.22 * (Syme**0.712)) * 0.7

    """
    End of customizable zone
    """
    return UCS