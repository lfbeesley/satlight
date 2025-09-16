import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 12

# Read data file
file_name = 'data/solar_spectra.txt'

# Skip header
data = np.loadtxt(file_name, skiprows=142) 

# Extract columns
wavelength = data[:, 0]  # nm
irradiance_mar25_29 = data[:, 1]  # W/m²/nm
irradiance_mar30_apr4 = data[:, 2]  # W/m²/nm  
irradiance_apr10_16 = data[:, 3]  # W/m²/nm
data_source = data[:, 4]  # Source identifier

# Create DataFrame
df = pd.DataFrame({
    'wavelength_nm': wavelength,
    'irradiance_mar25_29': irradiance_mar25_29,
    'irradiance_mar30_apr4': irradiance_mar30_apr4,
    'irradiance_apr10_16': irradiance_apr10_16,
    'data_source': data_source
})

# Plot all three reference periods
plt.figure(figsize=(9, 6))
plt.plot(wavelength, irradiance_mar25_29, label='Mar 25-29', alpha=0.8)
plt.plot(wavelength, irradiance_mar30_apr4, label='Mar 30-Apr 4', alpha=0.8)
plt.plot(wavelength, irradiance_apr10_16, label='Apr 10-16', alpha=0.8)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Solar Spectral Irradiance (W/m²/0.1nm)')
plt.title('Solar Reference Spectra - WHI 2008')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 2400)

# plot RGB
RGB = np.array([(780 + 618) / 2, (570 + 497 / 2), (476 + 427) / 2,400,1100]) # central frequency of spectral range
color = ['r','g','b','k','k']
for i in range(len(RGB)):
    irradiance_at_wvl = irradiance_mar25_29[np.argmin(np.abs(wavelength-RGB[i]))]
    plt.vlines(RGB[i],0,irradiance_at_wvl,color = color[i])

# Expected integrated irrandiance for observer
mask = (wavelength > 400) & (wavelength < 1100)
indices = np.where(mask)
I = np.sum(irradiance_mar25_29[indices])/10 #W/m^2
print('Intensity of recieved light between 400 nm and 1100 nm:')
