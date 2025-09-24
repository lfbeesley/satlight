import numpy as np
from skyfield.api import load, wgs84, EarthSatellite
from datetime import timedelta
from astropy.constants import R_sun, R_earth, au
from astropy import units as u
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    
    def create_satellite(self, tle_line1, tle_line2, name = 'satellite'): #future will include OMM and custom orbits
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
    
    def next_visibility(self, t0_utc, N = 24, i = 30.0):
        """Calculate the next times the object will be observable above i° within the following N hours.
        Includes illumination and local daytime.

        Args:
            time_utc: Individual start time in (2025, 5, 4, 0, 44, 38) format.
            N: Number of hours to search ahead (default: 24)
            i: Minimum altitude in degrees (default: 30.0)
        
        Returns:
            Start time of observation (UTC).
            Duration of pass (seconds).
            """

        t0 = self.ts.utc(*t0_utc)
        t1 = t0 + timedelta(hours = N)

        t, events = self.satellite.find_events(self.observer, t0, t1, altitude_degrees= i)
        
        if len(events) > 0:
            mask_rise = (events == 0)
            mask_set = (events == 2)

            t_rise = t[mask_rise] # Rising above threshold
            t_set = t[mask_set]   # Setting below threshold

            duration = (t_set - t_rise) * 86400.0 # To convert to seconds
            return t_rise, duration
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
        self.incident_vector = (sat_pos - sun_pos) / np.linalg.norm(sat_pos - sun_pos, axis=0)
        self.outgoing_vector = (obs_pos - sat_pos) / np.linalg.norm(obs_pos - sat_pos, axis=0)
        self.prop_distance = np.linalg.norm(obs_pos - sat_pos, axis=0)

        self.vectors = {
            'sun_to_sat_unit': self.incident_vector,
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
        
        # LVLH frame assignments
        self.nadir = radial              # Toward Earth
        self.along_track_unit = along_track  # X-axis (velocity direction)
        self.cross_track_unit = orbit_normal # Y-axis (orbit normal)  
        self.zenith = -self.nadir        # Z-axis (away from Earth)


    def _calculate_solar_phase_angle(self):
        """Calculate and store solar phase angle."""
        incident = self.incident_vector
        outgoing = self.outgoing_vector
        
        if incident.ndim == 1:
            # Single time: regular dot product
            dot_product = np.dot(incident, outgoing)
        else:
            # Array time: element-wise multiply then sum along vector dimension (axis=0)
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

        if self.phase_angle > theta_sun + theta_earth:
            # Object is fully illumintated
            f = 1 

        elif self.phase_angle < theta_sun:
            # Object is fully eclipsed
            f = 0
        
        else:
            # Object is partially illuminated
            r1 = theta_sun
            r2 = theta_earth
            d = theta

            phi1 = np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
            phi2 = np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
            A = r1**2 * phi1 + r2**2 * phi2 - 0.5 * np.sqrt((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))

            f = 1 - A / (np.pi * r1**2)

        return f  





if __name__ == '__main__':
    print("="*50)
    print("SATELLITE GEOMETRY TEST - TIME SERIES")
    print("="*50)
    
    # Test parameters
    line1 = '1 55341U 23013L   25094.09942582  .00001345  00000-0  11466-3 0 9990'
    line2 = '2 55341  43.0031 345.7949 0001445 255.2239 104.8444 15.02528780121056'
    time_utc = (2025, 5, 4, 0, 25, range(0,200))  # 60 seconds
    observer_lat = 28.29156
    observer_lon = -16.62
    observer_alt_m = 2070
    
    print(f"Satellite TLE: {line1[2:7]} ({line1[18:32].strip()})")
    print(f"Observer: {observer_lat:.4f}°, {observer_lon:.4f}°, {observer_alt_m}m")
    print(f"Time range: 60 seconds starting from {time_utc[:5]}...")
    print()
    
    # Create geometry object
    print("Creating geometry calculator...")
    geometry = Geometry()
    
    # Set up scene
    print("Setting up observation scenario...")
    geometry.create_observer(observer_lat, observer_lon, observer_alt_m)
    geometry.create_satellite(line1, line2)
    geometry.set_time(time_utc)

    # Check Satellite is visibile:
    print("\n" + "-"*40)
    print("OBSERVATION TIMES")
    print("-"*40)
    t, dur = geometry.next_visibility((2025, 5, 4, 22, 30))
    for iN in range(len(t)):
        print(f"Observable above 30° from {t[iN].utc_strftime('%d/%m/%Y %H:%M:%S')} for {dur[iN]:.0f} seconds.")

    
    # Print first and last time steps
    print("\n" + "-"*40)
    print("POSITIONS - FIRST & LAST TIME STEPS (km)")
    print("-"*40)
    for obj, pos in geometry.positions.items():
        if obj == 'sun':
            print(f"{obj.capitalize()} (t=0):  [{pos[0,0]:11.0f}, {pos[1,0]:11.0f}, {pos[2,0]:11.0f}]")
            print(f"{obj.capitalize()} (t=59): [{pos[0,-1]:11.0f}, {pos[1,-1]:11.0f}, {pos[2,-1]:11.0f}]")
        else:
            print(f"{obj.capitalize()} (t=0):  [{pos[0,0]:8.1f}, {pos[1,0]:8.1f}, {pos[2,0]:8.1f}]")
            print(f"{obj.capitalize()} (t=59): [{pos[0,-1]:8.1f}, {pos[1,-1]:8.1f}, {pos[2,-1]:8.1f}]")
    
    # Print vector shapes and sample values
    print("\n" + "-"*40)
    print("VECTOR ARRAY SHAPES & SAMPLE VALUES")
    print("-"*40)
    print(f"Distance array shape: {geometry.prop_distance.shape}")
    print(f"Phase angle array shape: {geometry.phase_angle.shape}")
    print(f"Reference frame shapes:")
    print(f"  Nadir: {geometry.nadir.shape}")
    print(f"  Along-track: {geometry.along_track_unit.shape}")
    print(f"  Cross-track: {geometry.cross_track_unit.shape}")
    
    # Show distance and phase angle variation
    print("\n" + "-"*40)
    print("TIME VARIATION ANALYSIS")
    print("-"*40)
    print(f"Distance range: {geometry.prop_distance.min():.1f} - {geometry.prop_distance.max():.1f} km")
    print(f"Distance change: {geometry.prop_distance[-1] - geometry.prop_distance[0]:+.1f} km over 60s")
    print(f"Phase angle range: {np.degrees(geometry.phase_angle).min():.1f}° - {np.degrees(geometry.phase_angle).max():.1f}°")
    print(f"Phase angle change: {np.degrees(geometry.phase_angle[-1] - geometry.phase_angle[0]):+.2f}° over 60s")
    
    # Test orthogonality at first and last time steps
    print("\n" + "-"*40)
    print("ORTHOGONALITY CHECK (First & Last)")
    print("-"*40)
    
    # First time step
    nadir_0 = geometry.nadir[:, 0]
    along_0 = geometry.along_track_unit[:, 0]
    cross_0 = geometry.cross_track_unit[:, 0]
    
    print("t=0:")
    print(f"  Nadir · Along-track: {np.dot(nadir_0, along_0):8.5f}")
    print(f"  Nadir · Cross-track: {np.dot(nadir_0, cross_0):8.5f}")
    print(f"  Along · Cross-track: {np.dot(along_0, cross_0):8.5f}")
    
    # Last time step  
    nadir_59 = geometry.nadir[:, -1]
    along_59 = geometry.along_track_unit[:, -1]
    cross_59 = geometry.cross_track_unit[:, -1]
    
    print("t=59:")
    print(f"  Nadir · Along-track: {np.dot(nadir_59, along_59):8.5f}")
    print(f"  Nadir · Cross-track: {np.dot(nadir_59, cross_59):8.5f}")
    print(f"  Along · Cross-track: {np.dot(along_59, cross_59):8.5f}")
    
    # Show some sample time steps for verification
    print("\n" + "-"*40)
    print("SAMPLE TIME SERIES DATA")
    print("-"*40)
    sample_times = [0, 14, 29, 44, 59]  # Show 5 time steps
    print(f"{'Time':>4} {'Distance':>8} {'Phase°':>7} {'Altitude':>8}")
    print("-" * 30)
    
    for i in sample_times:
        sat_alt = np.linalg.norm(geometry.positions['satellite'][:, i]) - 6371
        print(f"{i:4d} {geometry.prop_distance[i]:8.1f} {np.degrees(geometry.phase_angle[i]):7.1f} {sat_alt:8.1f}")
    
    print(f"\n{'='*50}")
    print("TIME SERIES TEST COMPLETE")
    print(f"Successfully processed {geometry.positions['satellite'].shape[1]} time steps")
    print(f"{'='*50}")



    # Extract positions from geometry
    sat = geometry.positions['satellite']
    obs = geometry.positions['observer']

    # Earth parameters
    ui = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_earth = R_earth * np.outer(np.cos(ui), np.sin(v))
    y_earth = R_earth * np.outer(np.sin(ui), np.sin(v))
    z_earth = R_earth * np.outer(np.ones_like(ui), np.cos(v))

    # Create 3D figure
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth as a sphere
    ax.plot_surface(x_earth, y_earth, z_earth, color='b', alpha=0.3, edgecolor='k')

    # Plot satellite and observer
    ax.plot(sat[0], sat[1], sat[2], label='Satellite', color='red', marker='o')
    ax.plot(obs[0], obs[1], obs[2], label='Observer', color='green', marker='^')

    # Labels and legend
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.legend()
    ax.set_title('Satellite and Observer with Earth Sphere')

    # Make aspect ratio equal
    ax.set_box_aspect([1,1,1])

    plt.show()
