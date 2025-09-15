import os
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
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
    luminance = 0.333 * r + 0.333 * g + 0.333* b

    # Total light flux (sum of luminance values)
    return np.mean(luminance)

df = glob.glob('/Users/l.beesley@bham.ac.uk/light-characterisation-data/render*.exr')

alt = [int(idf.split('_')[1].strip('alt')) for idf in df]
az = [int(idf.split('_')[2].strip('az.exr')) for idf in df]

brightness = [calculate_light_flux_exr(idf) for idf in df]

plt.plot(az,brightness,'.')
plt.figure()
plt.plot(alt,brightness,'.')

