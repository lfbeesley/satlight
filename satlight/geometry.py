import numpy as np
import os
import datetime
from skyfield.api import Loader, wgs84, EarthSatellite
from sgp4.api import Satrec, WGS84 as WGS84_gravity
from astropy.constants import R_sun, R_earth, au
from astropy import units as u
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# store de421.bsp in users local dir
_data_dir = os.path.join(os.path.expanduser('~'), '.satlight')
os.makedirs(_data_dir, exist_ok=True)
load = Loader(_data_dir)

# Constants
R_earth = R_earth.to(u.km).value
R_sun =  R_sun.to(u.km).value
AU = au.to(u.km).value

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
    
    def create_satellite(self, tle_line1, tle_line2, name = 'satellite'): #future will include OMM and custom orbits?
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
    
    def next_visibility(self, t0_utc, N=24, i=30.0):
        """Calculate the next times the object will be observable above i° within the following N hours."""
        t0 = self.ts.utc(*t0_utc)
        t1 = t0 + datetime.timedelta(hours=N)

        # Check elevation at t0 itself, in case the satellite is already above threshold
        alt0, _, _ = (self.satellite - self.observer).at(t0).altaz()
        already_up = alt0.degrees >= i

        t, events = self.satellite.find_events(self.observer, t0, t1, altitude_degrees=i)

        results = []
        pending_rise = t0 if already_up else None

        for ti, ev in zip(t, events):
            if ev == 0:  # rise
                pending_rise = ti
            elif ev == 2 and pending_rise is not None:  # set
                duration = (ti - pending_rise) * 86400.0
                y, mo, d, h, mi, s = pending_rise.utc
                results.append((
                    int(y), int(mo), int(d), int(h), int(mi),
                    range(int(s), int(s + duration))
                ))
                pending_rise = None

        # Pass still ongoing when the window closes
        if pending_rise is not None:
            duration = (t1 - pending_rise) * 86400.0
            y, mo, d, h, mi, s = pending_rise.utc
            results.append((
                int(y), int(mo), int(d), int(h), int(mi),
                range(int(s), int(s + duration))
            ))

        if results:
            return results
        else:
            print("Satellite is not observable during this period. Increase the search time or change observer location.")

    def _auto_calculate(self):
        """Automatically calculate positions and vectors if we have all required inputs."""
        if self.satellite is not None and self.observer is not None and self.time is not None:
            self._calculate_positions()
            self._calculate_vectors()
            self._calculate_solar_phase_angle()
            self._calculate_lvlh_frame()
    
    def _calculate_positions(self):
        """Get positions of satellite, observer, and sun (in GCRS).
        
        Returns in km's."""
        if self.time.shape == ():
            self.positions = {
                'satellite': self.satellite.at(self.time).position.km, # automatically in GCRS
                'observer': self.observer.at(self.time).position.km,
                'sun': (self.sun.at(self.time) - self.earth.at(self.time)).position.km, # Subtract to move from Barycentric to GCRS
                'earth': np.zeros(3) 
            }

        else:
            self.positions = {
                'satellite': self.satellite.at(self.time).position.km, # automatically in GCRS
                'observer': self.observer.at(self.time).position.km,
                'sun': (self.sun.at(self.time) - self.earth.at(self.time)).position.km, # Subtract to move from Barycentric to GCRS
                'earth': np.zeros((3, len(self.time))) 
            }

    def _calculate_vectors(self):
        """Calculate and store all vectors and distances."""
        sat_pos = self.positions['satellite']
        obs_pos = self.positions['observer']
        sun_pos = self.positions['sun']
        
        # Compute unit vectors and distances
        self.incident_vector = (sun_pos - sat_pos) / np.linalg.norm(sun_pos - sat_pos, axis=0)  # unit vector from satellite toward sun
        self.outgoing_vector = (obs_pos - sat_pos) / np.linalg.norm(obs_pos - sat_pos, axis=0)
        self.prop_distance = np.linalg.norm(obs_pos - sat_pos, axis=0)

        self.vectors = {
            'sat_to_sun_unit': self.incident_vector,
            'sat_to_obs_unit': self.outgoing_vector,
            'obs_to_sat_distance': self.prop_distance}
        
    
    def _calculate_lvlh_frame(self):
        """Calculates vectors in satellite's inertial frame (LVLH)."""
        sat_pos = self.positions['satellite']
        sat_vel = self.satellite.at(self.time).velocity.km_per_s
        
        if sat_pos.ndim == 1:
            # Single time case
            radial = -sat_pos / np.linalg.norm(sat_pos)
            h_vector = np.cross(sat_pos, sat_vel)
            orbit_normal = h_vector / np.linalg.norm(h_vector)
            along_track = np.cross(orbit_normal, radial)
            along_track = along_track / np.linalg.norm(along_track)
        else:
            # Array case - vectors are (3, N) shape
            radial = -sat_pos / np.linalg.norm(sat_pos, axis=0)
            
            # For cross product with (3, N) arrays, use axis=0 to indicate vector dimension
            h_vector = np.cross(sat_pos, sat_vel, axis=0)
            orbit_normal = h_vector / np.linalg.norm(h_vector, axis=0)
            
            along_track = np.cross(orbit_normal, radial, axis=0)
            along_track = along_track / np.linalg.norm(along_track, axis=0)
        
        # LVLH frame axis
        self.nadir = radial              # Toward Earth
        self.along_track = along_track  # X-axis (velocity direction)
        self.cross_track = orbit_normal # Y-axis (orbit normal)  
        self.zenith = -radial       # Z-axis (away from Earth)

        # Rotate 
        if sat_pos.ndim == 1:
            # Single time: R is (3,3), vectors are (3,)
            R = np.stack([along_track, orbit_normal, -radial], axis=1)
            self.incident_vector_lvlh = R.T @ self.incident_vector
            self.outgoing_vector_lvlh = R.T @ self.outgoing_vector
        else:
            # Array time: R is (3,3,N), vectors are (3,N)
            R = np.stack([along_track, orbit_normal, -radial], axis=1)
            self.incident_vector_lvlh = np.einsum('ijn,jn->in', R.transpose(1,0,2), self.incident_vector)
            self.outgoing_vector_lvlh = np.einsum('ijn,jn->in', R.transpose(1,0,2), self.outgoing_vector)
        

    def _calculate_solar_phase_angle(self):
        """Calculate and store solar phase angle in radians."""
        incident = self.incident_vector
        outgoing = self.outgoing_vector
        
        if incident.ndim == 1:
            dot_product = np.dot(incident, outgoing)
        else:
            # If time is array
            dot_product = np.sum(incident * outgoing, axis=0)
        
        self.phase_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    def eclipse_type(self):
        """Calculate the nature of the eclipse, used to identify object illumination"""
        # Positions
        sat_pos = self.positions['satellite']
        earth_pos = self.positions['earth']
        
        # Distances
        dist_sat_to_earth = np.linalg.norm(sat_pos - earth_pos, axis=0) # Distance from satellite to Earth

        # Apparent angular radii
        theta_earth = np.arctan(R_earth/dist_sat_to_earth)
        theta_sun= np.arctan(R_sun/AU)

        # Angular seperation between Earth and Sun
        theta = self.phase_angle

        # Calculate fraction of sun that is visible
        f = np.zeros_like(theta)

        # Object is fully illumintated
        f = np.where(theta > theta_sun + theta_earth, 1, f)

        # Object is fully eclipsed
        f = np.where(theta < theta_sun - theta_earth, 0, f)

        # Object is partially illuminated
        mask = (theta <= theta_sun + theta_earth) & (theta >= np.abs(theta_sun - theta_earth))

        if np.any(mask):  # Only compute if there’s at least one element
            r1 = theta_sun
            r2 = theta_earth
            d = theta[mask]

            phi1 = np.arccos(np.clip((d**2 + r1**2 - r2**2) / (2 * d * r1), -1, 1))
            phi2 = np.arccos(np.clip((d**2 + r2**2 - r1**2) / (2 * d * r2), -1, 1))
            A = r1**2 * phi1 + r2**2 * phi2 - 0.5 * np.sqrt(np.clip((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2), 0, None))
            f[mask] = 1 - A / (np.pi * r1**2)
        
        else:
            r1 = theta_sun
            r2 = theta_earth
            d = theta

            phi1 = np.arccos(np.clip((d**2 + r1**2 - r2**2) / (2 * d * r1), -1, 1))
            phi2 = np.arccos(np.clip((d**2 + r2**2 - r1**2) / (2 * d * r2), -1, 1))
            A = r1**2 * phi1 + r2**2 * phi2 - 0.5 * np.sqrt(np.clip((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2), 0, None))
            f = 1 - A / (np.pi * r1**2)

        return f  

    def create_satellite_from_elements(self, a_km, e, i_deg, raan_deg,
                                        argp_deg, nu_deg, epoch=None):
        if epoch is None:
            epoch = datetime.datetime.utcnow()

        # True anomaly -> mean anomaly
        nu = np.radians(nu_deg)
        E  = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2),
                            np.sqrt(1 + e) * np.cos(nu / 2))
        M  = (E - e * np.sin(E)) % (2 * np.pi)

        # Mean motion in rad/s
        GM = 398600.4418
        n  = np.sqrt(GM / a_km**3)

        # SGP4 epoch: days from 1949-12-31 00:00:00 UTC
        epoch_base = datetime.datetime(1949, 12, 31, 0, 0, 0)
        epoch_days = (epoch - epoch_base).total_seconds() / 86400.0

        satrec = Satrec()
        satrec.sgp4init(
            WGS84_gravity,
            'i',
            99999,
            (epoch - datetime.datetime(1949, 12, 31, 0, 0, 0)).total_seconds() / 86400.0,
            0.0,   # bstar
            0.0,   # ndot
            0.0,   # nddot
            e,
            np.radians(argp_deg),
            np.radians(i_deg),
            M,
            n * 60, # rad/min 
            np.radians(raan_deg),
        )

        self.satellite = EarthSatellite.from_satrec(satrec, self.ts)
        return self.satellite
