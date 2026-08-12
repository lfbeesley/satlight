import numpy as np

def AB_mag(I):
    """
    Docstring for AB_mag:
      -given in V-band
      param I: integrated flux in units of W/m^2
    """
    # for V-band
    wvl_1 = 500e-9 
    wvl_2 = 600e-9  
    
    # Convert wavelength range to frequency range
    c = 3e8
    nu_1 = c / wvl_2
    nu_2 = c / wvl_1
    delta_nu = nu_2 - nu_1 

    # Calc AB_magnitude
    AB_mag = -2.5 * np.log10(I/delta_nu) + 8.9 - 65

    return AB_mag