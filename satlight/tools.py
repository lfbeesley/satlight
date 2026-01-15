import numpy as np

def AB_mag(I):
    """
    Docstring for AB_mag:
      -given in V-band (freq. between 40 and 75 GHz)

      param I: integrated flux in units of W/m^2
    """

    # for V-band
    lambda_1 = 500e-9 
    lambda_2 = 600e-9  
    
    # Convert wavelength range to frequency range
    c = 3e8
    nu_1 = c / lambda_2
    nu_2 = c / lambda_1
    delta_nu = nu_2 - nu_1 

    # Calc AB_magnitude
    AB_mag = -2.5 * np.log10(I/delta_nu) + 8.9 - 62 # - 62 convert eqn. from Jy to W

    return AB_mag