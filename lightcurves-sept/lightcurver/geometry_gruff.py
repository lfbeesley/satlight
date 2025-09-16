import numpy as np
from skyfield.api import load, wgs84, EarthSatellite

class GCRSGeometry():
    """
    Calculate geometric relationships between satellite, observer, Sun, and Earth
    in the Geocentric Celestial Reference System (GCRS) for multiple times.
    """
    def __init__(self):
        """Initialise SceneGeometry with required astronomical data."""
        self.ts = load.timescale()
        self.planets = load('de440s.bsp') 
        self.earth = self.planets['earth']
        self.sun = self.planets['sun']
        self.moon = self.planets['moon']

    def create_satellite(self, tle_line1, tle_line2, name='satellite'):
        """Create a satellite object from TLE data."""
        self.satellite = EarthSatellite(tle_line1, tle_line2, name)
        return self.satellite

    def create_observer(self, observer_lat, observer_lon, observer_alt_m):
        """Create an observer object."""
        self.observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_alt_m)
        return self.observer

    def set_times(self, time_tuples):
        """Set multiple observation times.

        Parameters:
        -----------
        time_tuples : list of tuples
            Each tuple: (year, month, day, hour, minute, second)
        """
        if self.satellite is None or self.observer is None:
            raise ValueError("Satellite and observer must be defined before setting times.")

        # Create Skyfield time array
        self.times = self.ts.utc(*zip(*time_tuples))
        self._calculate_all()

    def _calculate_all(self):
        """Automatically calculate positions and vectors for all time steps."""
        sat_at = self.satellite.at(self.times)
        obs_at = self.observer.at(self.times)
        sun_at = self.sun.at(self.times) - self.earth.at(self.times)

        # Positions: shape (N, 3)
        self.positions = {
            'satellite': sat_at.position.km.T,
            'observer': obs_at.position.km.T,
            'sun': sun_at.position.km.T
        }

        sat_pos = self.positions['satellite']
        obs_pos = self.positions['observer']
        sun_pos = self.positions['sun']

        # Compute vectors and distances
        sat_to_obs = obs_pos - sat_pos
        sat_to_obs_dist = np.linalg.norm(sat_to_obs, axis=1)
        sat_to_obs_unit = sat_to_obs / sat_to_obs_dist[:, None]

        sun_to_sat_unit = (sat_pos - sun_pos)
        sun_to_sat_unit /= np.linalg.norm(sun_to_sat_unit, axis=1)[:, None]

        sat_to_earth_unit = -sat_pos / np.linalg.norm(sat_pos, axis=1)[:, None]

        self.vectors = {
            'sat_to_obs_unit': sat_to_obs_unit,
            'sat_to_obs_distance': sat_to_obs_dist,
            'sun_to_sat_unit': sun_to_sat_unit,
            'sat_to_earth_center_unit': sat_to_earth_unit
        }

    @property
    def is_ready(self):
        """Check if all calculations have been performed."""
        return self.positions is not None and self.vectors is not None
