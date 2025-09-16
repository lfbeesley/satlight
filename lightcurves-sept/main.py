from datetime import datetime, timedelta
from lightcurver import blender, geometry_gruff
import matplotlib.pyplot as plt
import os

folder = 'test1'

datapath = os.path.join(os.getcwd(), folder)
if not os.path.exists(folder):
    os.makedirs(datapath)

CADname = 'OneWeb.stl'
CADfolder = 'C:/Users/gxj236/Desktop/Satellite CADs/Oneweb'

# Set the full path to the CAD file
CADpath = os.path.join(CADfolder, CADname)

line1 = '1 55341U 23013L   25094.09942582  .00001345  00000-0  11466-3 0  9990'
line2 = '2 55341  43.0031 345.7949 0001445 255.2239 104.8444 15.02528780121056'
observer_lat = 43.6469
observer_lon = 41.4406
observer_alt_m = 2070
time_utc_start = (2025, 5, 4, 0, 42, 0)
time_utc_end = (2025, 5, 4, 0, 47, 0)
timestep = 3 # seconds

# Generate time steps
start = datetime(*time_utc_start)
end = datetime(*time_utc_end)
num_steps = int((end - start).total_seconds() / timestep) + 1
time_tuples = [(start + timedelta(seconds=i*timestep)).timetuple()[:6] for i in range(num_steps)]

# Create geometry object
geom = geometry_gruff.GCRSGeometry()
geom.create_observer(observer_lat, observer_lon, observer_alt_m)
geom.create_satellite(line1, line2)
geom.set_times(time_tuples)

lightcurve=blender.render(geom,datapath=datapath,CADpath=CADpath, samples=128)

plt.plot(lightcurve)
plt.show()