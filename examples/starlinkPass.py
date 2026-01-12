from satlight.raytracer import Renderer
from satlight.geometry import Geometry
from astropy.time import Time, TimeDelta

# STARLINK-1933           
line1 = '1 55961U 23037AZ  25317.64931522  .00000048  00000+0  11789-4 0  9992'
line2 = '2 55961  70.0003 196.9458 0001784 286.2026  73.8936 14.98326354147098'
time_utc = (2025, 11, 13, 0, 25, 30)  # 60 seconds

# Place observer at La Palma
observer_lat = 28.7547
observer_lon = -17.8853
observer_alt_m = 2070

# Crate geometry object
geometry = Geometry()
geometry.create_observer(observer_lat, observer_lon, observer_alt_m)
geometry.create_satellite(line1, line2)
geometry.set_time(time_utc)

t = geometry.next_visibility((2025, 5, 4, 22, 30),N=120)

for tup in t:
    year, month, day, hour, minute, r = tup
    duration = r.stop - r.start  # extract from range
    print(f"Observable above 30° from ({year}, {month}, {day}, {hour}, {minute}, {r.start}) "
        f"for {duration} seconds.")
    
# Set time to beginning of next observation
geometry.set_time(t[0])
sat = geometry.positions['satellite']
obs = geometry.positions['observer']

in_vector = geometry.incident_vector_lvlh
out_vector = geometry.outgoing_vector_lvlh

