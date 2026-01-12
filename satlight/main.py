from datetime import datetime,timedelta
from load_TLEs import load_tles_and_select, download_tles
from lightcurver.observed_satellite import ObservedSatellite

download_tles()
name, line1, line2 = load_tles_and_select()
tle_lines = [name, line1, line2]

sat = ObservedSatellite(tle_lines, start_time=datetime.now(), num_secs=50000)

ground_stations = [
    {'name': 'London', 'lat': 51.4934, 'lon': -0.0098, 'alt': 0},
    {'name': 'Paris', 'lat': 48.8575, 'lon': 2.3514, 'alt': 0},
    {'name': 'Gibraltar', 'lat': 36.1408, 'lon': -5.3536, 'alt': 0},
]

sat.get_contacts(ground_stations, min_elevation_deg=20)
sat.compute_vectors(ground_stations, frame='CIRS')
sat.plot_groundtrack(ground_stations)
sat.save_contacts('GS_contacts/contacts.json')
