import os
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

def calculate_light_flux_exr(image_path):
    exr_file = OpenEXR.InputFile(image_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    size = (height, width)

    # Read R, G, B channels
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    r = np.frombuffer(exr_file.channel("R", pt), dtype=np.float32).reshape(size)
    g = np.frombuffer(exr_file.channel("G", pt), dtype=np.float32).reshape(size)
    b = np.frombuffer(exr_file.channel("B", pt), dtype=np.float32).reshape(size)

    # Compute luminance approximation (Rec. 709 weights)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Total light flux (sum of luminance values)
    return np.sum(luminance)

# load contact information
json_path = os.path.join(os.getcwd(), "GS_Contacts/contacts.json")

with open(json_path, "r") as f:
    contacts = json.load(f)

# Use first contact
contact = contacts[0] # make sure the contact being used is the same as the one in Blender

ranges=contact['range_km']
sunlit=contact['sunlit']

zenith_angles_rad=np.pi/2-np.radians(contact['elevation_deg'])
time_strings = contact['timearr']
times = [datetime.fromisoformat(t) for t in time_strings]

directory_renders = os.path.join(os.getcwd(), "renders")

exr_files = sorted([f for f in os.listdir(directory_renders ) if f.lower().endswith('.exr')])

# atmospheric extinction coefficient
ext_coeff=0.1 # probably quite hard to model accurately in practice

#try different pixel sizes and scaling with the distance!

magnitudes_final=[]

for i, file in enumerate(exr_files):

    path = os.path.join(directory_renders, file)
    light_flux = calculate_light_flux_exr(path)

    # if it is not illuminated due to eclipse we get rid of it here
    final_flux=light_flux*sunlit[i]

    # calculate the 'magnitude'
    if final_flux!=0:
        mag_inst = -2.5*np.log10(final_flux)
        mag_range = -5*np.log10(ranges[i]/1000) #normalised for range
        mag_atmos = ext_coeff*1/np.cos(zenith_angles_rad[i])

        magnitude = mag_inst + mag_range + mag_atmos

        magnitudes_final.append(magnitude)
    else:
        magnitudes_final.append(None)


plt.figure(figsize=(10, 5))
plt.plot(times, magnitudes_final, marker='o')
plt.title("Magnitude (Physical meaningfulness TBD) during pass")
plt.xlabel("Time (UTC)")
plt.ylabel("Magnitude)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()

