import numpy as np
from skyfield.api import load, wgs84, EarthSatellite

class Geometry():
    """
    Calculate geometric relationships between satellite, observer, Sun, and Earth
    in the Geocentric Celestial Reference System (GCRS).
    """
    def __init__(self):
        """Initialise scene geometry."""
        self.ts = load.timescale()
        self.planets = load('de421.bsp') 
        self.earth = self.planets['earth']
        self.sun = self.planets['sun']
        self.moon = self.planets['moon']
    
    def create_satellite(self, tle_line1, tle_line2, name = 'satellite'):
        """Create a satellite object from TLE data."""
        self.satellite = EarthSatellite(tle_line1, tle_line2, name)
        return self.satellite
    
    def create_observer(self, observer_lat, observer_lon, observer_alt_m):
        """Create a observer object."""
        self.observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_alt_m)
        return self.observer
    
    def set_time(self, time_utc):
        """Set observation time in UTC.

        For SINGLE time calculation:
        ---------------------------
        geom.set_time(2025, 5, 4, 0, 44, 38)           # Individual arguments
        geom.set_time((2025, 5, 4, 0, 44, 38))         # Tuple format
        
        For MULTIPLE time calculations:
        ------------------------------
        # Use ranges or arrays for any time component:
        geom.set_time(2025, 5, 4, 0, 44, range(0, 60))           # 60 seconds
        geom.set_time(2025, 5, 4, 0, range(44, 46), 0)           # 2 minutes

        See https://rhodesmill.org/skyfield/time.html for other formats.
        """
        
        self.time = self.ts.utc(*time_utc)  
        self._auto_calculate()

    def _auto_calculate(self):
        """Automatically calculate positions and vectors if we have all required inputs."""
        if self.satellite is not None and self.observer is not None and self.time is not None:
            self._calculate_positions()
            self._calculate_vectors()
            self._calculate_solar_phase_angle()
    
    def _calculate_positions(self):
        """Get positions of satellite, observer, and sun (in GCRS)."""
        self.positions = {
            'satellite': self.satellite.at(self.time).position.km,
            'observer': self.observer.at(self.time).position.km,
            'sun': (self.sun.at(self.time) - self.earth.at(self.time)).position.km
        }

    def _calculate_vectors(self):
        """Calculate and store all vectors and distances."""
        sat_pos = self.positions['satellite']
        obs_pos = self.positions['observer']
        sun_pos = self.positions['sun']
        
        # Compute unit vectors and distances
        # Light scattering vectors
        self.incident_vector = (sat_pos - sun_pos) / np.linalg.norm(sat_pos - sun_pos)

        sat_to_obs_vector = obs_pos - sat_pos
        self.outgoing_vector = sat_to_obs_vector / np.linalg.norm(sat_to_obs_vector)

        self.prop_distance = np.linalg.norm(sat_to_obs_vector)

        # Satellite reference frame
        self.nadir = -sat_pos / np.linalg.norm(sat_pos)  # Points toward Earth (nadir)
        self.zenith = - self.nadir # Zenith is Z-axis

        sat_vel = self.satellite.at(self.time).velocity.km_per_s 
        self.along_track_unit = sat_vel / np.linalg.norm(sat_vel) # Along-track is X-axis
        
        self.cross_track = np.cross(self.along_track_unit, self.zenith) # Cross-track is Y-axis

        self.vectors = {
            'sun_to_sat_unit': self.incident_vector,
            'sat_to_obs_unit': self.outgoing_vector,
            'obs_to_sat_distance': self.prop_distance,
            'sat_to_earth_center_unit': self.nadir,
            'sat_vel': sat_vel,
            'along_track_unit' : self.along_track_unit
        }

    def _calculate_solar_phase_angle(self):
        """Calculate and store solar phase angle."""
        self.phase_angle = np.arccos(np.clip(np.dot(self.incident_vector, self.outgoing_vector), -1.0, 1.0))
    
    @property
    def is_ready(self):
        """Check if all calculations have been performed."""
        return (self.satellite is not None and 
                self.observer is not None and 
                self.time is not None and
                self.positions is not None and 
                self.vectors is not None)



if __name__ == '__main__':
    line1 = '1 55341U 23013L   25094.09942582  .00001345  00000-0  11466-3 0  9990'
    line2 = '2 55341  43.0031 345.7949 0001445 255.2239 104.8444 15.02528780121056'
    time_utc = (2025, 5, 4, 21, 44, 38)
    observer_lat = 43.6469
    observer_lon = 41.4406
    observer_alt_m = 2070

    # Create geometry object
    geometry = Geometry()

    # Set up scene
    geometry.create_observer(observer_lat, observer_lon, observer_alt_m)
    geometry.create_satellite(line1, line2)
    geometry.set_time(time_utc)

    # Print calculated geometries
    print(geometry.positions)
    print(geometry.vectors)
    print(np.degrees(geometry.phase_angle))


