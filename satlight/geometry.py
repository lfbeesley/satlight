import numpy as np
import os
import datetime
from skyfield.api import Loader, wgs84, EarthSatellite
from sgp4.api import WGS84 as WGS84_gravity, Satrec
from astropy.constants import R_sun, R_earth, au
from astropy import units as u

# store de421.bsp in users local dir
_data_dir = os.path.join(os.path.expanduser('~'), '.satlight')
os.makedirs(_data_dir, exist_ok=True)
load = Loader(_data_dir)

# Constants
R_EARTH_KM = R_earth.to(u.km).value
# WGS84. The 21 km equatorial-polar difference matters for occultation tests.
# The MEAN radius, for problems that only need the Earth's angular size (the
# earthshine cap). A sphere is adequate there: the 21 km equatorial-polar
# difference changes the subtended solid angle by 0.02%. It is NOT adequate for
# line_of_sight, which grazes the surface - see that method.
R_EARTH_MEAN_KM = 6371.0

R_EARTH_EQ_KM = 6378.137
R_EARTH_POLAR_KM = 6378.137 * (1.0 - 1.0 / 298.257223563)
R_SUN_KM =  R_sun.to(u.km).value
AU = au.to(u.km).value

class ViewGeometry:
    """The minimal viewing geometry a Renderer needs, for one epoch.

    `Renderer` takes a `geometry=`, which is normally a full `Geometry` built
    from an orbit. Plenty of legitimate work has no orbit behind it, though -
    BRDF studies, phase-angle sweeps, attitude scans - and those still need a
    way in. This is it: the same three attributes `Renderer` reads, and nothing
    else.

        geom = ViewGeometry(sun_lvlh, observer_lvlh, distance_km=550.0)
        r = Renderer(obj_path, geometry=geom)

    DISTANCE IS IN KILOMETRES, matching `Geometry.prop_distance`. The renderer
    converts to metres internally. That conversion living in one place is the
    whole reason the explicit three-vector constructor was removed: it used to
    sit in every caller, and getting it wrong is a factor-1000 in flux, i.e.
    7.5 magnitudes, with a perfectly normal-looking image to show for it.
    """

    __slots__ = ('incident_vector_lvlh', 'outgoing_vector_lvlh', 'prop_distance',
                 'altitude_km')

    def __init__(self, sun_lvlh, observer_lvlh, distance_km, altitude_km=None):
        sun = np.asarray(sun_lvlh, dtype=float)
        obs = np.asarray(observer_lvlh, dtype=float)
        if sun.shape != (3,) or obs.shape != (3,):
            raise ValueError(
                f'ViewGeometry takes one epoch: sun and observer must be shape '
                f'(3,), got {sun.shape} and {obs.shape}.')
        n_sun, n_obs = np.linalg.norm(sun), np.linalg.norm(obs)
        if n_sun == 0.0 or n_obs == 0.0:
            raise ValueError('sun_lvlh and observer_lvlh must be non-zero.')
        d = float(distance_km)
        if not np.isfinite(d) or d <= 0.0:
            raise ValueError(f'distance_km must be positive, got {distance_km}.')
        self.incident_vector_lvlh = sun / n_sun
        self.outgoing_vector_lvlh = obs / n_obs
        self.prop_distance = d
        # Only earthshine needs this: the Earth's angular size depends on
        # altitude, which the slant range alone does not give.
        self.altitude_km = None if altitude_km is None else float(altitude_km)

    def __repr__(self):
        return (f'ViewGeometry(sun={self.incident_vector_lvlh.round(4).tolist()}, '
                f'observer={self.outgoing_vector_lvlh.round(4).tolist()}, '
                f'distance_km={self.prop_distance:g})')


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

            self.satellite = None
            self.observer = None
            self.time = None

    def create_satellite(self, tle_line1, tle_line2, name = 'satellite'): #future will include OMM and custom orbits?
        """Create a satellite object from TLE data."""
        self.satellite = EarthSatellite(tle_line1, tle_line2, name)
        return self.satellite
    
    def create_observer(self, observer_lat, observer_lon, observer_alt_m):
        """Create a observer object."""
        self.observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_alt_m)
        return self.observer
    
    def set_time(self, *time_utc):
        """Set the observation time, in UTC.

        SINGLE epoch - all three forms are equivalent:

            geom.set_time(2025, 5, 4, 0, 44, 38)        # separate arguments
            geom.set_time((2025, 5, 4, 0, 44, 38))      # one tuple
            geom.set_time(datetime(2025, 5, 4, 0, 44, 38))   # a datetime

        MANY epochs - put a range or array in any one component:

            geom.set_time(2025, 5, 4, 0, 44, range(0, 60))     # 60 seconds
            geom.set_time(2025, 5, 4, 0, range(44, 46), 0)     # 2 minutes

        THE TUPLE FORM USED TO BE A TRAP. `ts.utc((2025, 5, 4, 0, 44, 38))`
        hands skyfield ONE sequence, which it reads as an array of YEARS -
        2025, then 5, then 4 - giving six epochs including year 0, or
        EphemerisRangeError. The docstring advertised the form regardless, so
        it is unpacked here rather than removed.

        To pass a genuine array of years, use a numpy array
        (`set_time(np.array([2024, 2025]))`); only `tuple` and `list` are
        unpacked, so arrays reach skyfield untouched.

        See https://rhodesmill.org/skyfield/time.html for other formats.
        """
        if len(time_utc) == 1:
            only = time_utc[0]
            if isinstance(only, datetime.datetime):
                if only.tzinfo is None:          # naive means UTC here
                    only = only.replace(tzinfo=datetime.timezone.utc)
                time_utc = (only,)
            elif (isinstance(only, (tuple, list)) and 2 <= len(only) <= 7
                  and all(np.isscalar(v) for v in only)):
                time_utc = tuple(only)           # the documented tuple form

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
            # along-track is n x r_hat, and r_hat is ZENITH (-radial), not the
            # nadir vector stored in `radial`. Crossing with `radial` returns
            # -velocity and makes the (along, cross, zenith) triad left-handed.
            along_track = np.cross(orbit_normal, -radial)
            along_track = along_track / np.linalg.norm(along_track)
        else:
            # Array case - vectors are (3, N) shape
            radial = -sat_pos / np.linalg.norm(sat_pos, axis=0)
            
            # For cross product with (3, N) arrays, use axis=0 to indicate vector dimension
            h_vector = np.cross(sat_pos, sat_vel, axis=0)
            orbit_normal = h_vector / np.linalg.norm(h_vector, axis=0)
            
            along_track = np.cross(orbit_normal, -radial, axis=0)
            along_track = along_track / np.linalg.norm(along_track, axis=0)
        
        # LVLH frame axis
        # Right-handed: along x cross = zenith. Verified in
        # tests/test_geometry_frames.py, which also pins along_track to +velocity.
        self.nadir = radial              # Toward Earth
        self.along_track = along_track   # X-axis, along velocity
        self.cross_track = orbit_normal  # Y-axis, orbit normal
        self.zenith = -radial            # Z-axis, away from Earth

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

    def line_of_sight(self, atmosphere_km=0.0):
        """Tangent height of the observer-to-satellite path, in km.

        Positive means the path clears the Earth; negative means the satellite
        is BEHIND the Earth as seen from the observer and cannot be observed at
        all. For a ground station it changes sign at the geometric horizon.

        This is the observer-side counterpart to eclipse_type(), which asks
        only whether the SUN is blocked from the satellite. Both are needed: a
        satellite can be in full sunlight and still be unobservable because the
        Earth is in the way.

        The Earth is treated as the WGS84 ELLIPSOID, not a sphere. That is not
        a refinement - it is required for correctness. The equatorial radius
        exceeds the polar by 21 km, so a spherical model of equatorial radius
        places every station outside the tropics below its own surface: La
        Palma at 28.8 deg comes out 2.5 km underground and every sightline is
        reported as blocked. The test is done by scaling the polar axis so the
        ellipsoid becomes a unit sphere, which makes it exact rather than
        approximate.

        atmosphere_km raises the blocking surface, so a path grazing the denser
        atmosphere counts as obstructed; 20-50 km is reasonable if you want to
        drop heavily extincted sightlines.

        Returns a float for a single epoch, or an array for array times.
        """
        a = R_EARTH_EQ_KM + atmosphere_km
        b = R_EARTH_POLAR_KM + atmosphere_km

        sat = np.atleast_2d(self.positions['satellite'].T).T.astype(float).copy()
        obs = np.atleast_2d(self.positions['observer'].T).T.astype(float).copy()

        # Scale the polar axis so the ellipsoid becomes a sphere of radius a.
        # The ellipsoid is fixed in the rotating frame, but its polar axis is
        # the GCRS z-axis to within precession and polar motion - tens of
        # metres here, far below anything that matters for an occultation.
        k = a / b
        sat[2] *= k
        obs[2] *= k

        d = sat - obs
        L = np.linalg.norm(d, axis=0)
        u = d / L

        t = -np.sum(obs * u, axis=0)
        closest = np.linalg.norm(obs + t * u, axis=0)

        # When the closest approach falls outside the segment, the path's
        # nearest approach to the centre is an endpoint, so nothing between
        # observer and satellite can block it.
        inside = (t > 0.0) & (t < L)
        closest = np.where(inside, closest,
                           np.minimum(np.linalg.norm(obs, axis=0),
                                      np.linalg.norm(sat, axis=0)))

        h = np.atleast_1d(closest - a)
        return h if h.size > 1 else float(h[0])

    def _blocked(self, atmosphere_km=0.0):
        """Whether the Earth actually intervenes between observer and satellite.

        Deliberately NOT derived from line_of_sight() > 0. When the satellite
        is high in the sky the path's lowest point is the observer's own
        position, so the tangent height is just the site altitude - and a site
        at sea level would give exactly 0.0 and be misreported as blocked. The
        obstruction only exists if the closest approach lies BETWEEN the two
        endpoints and dips below the surface.
        """
        a = R_EARTH_EQ_KM + atmosphere_km
        b = R_EARTH_POLAR_KM + atmosphere_km
        sat = np.atleast_2d(self.positions['satellite'].T).T.astype(float).copy()
        obs = np.atleast_2d(self.positions['observer'].T).T.astype(float).copy()
        k = a / b
        sat[2] *= k
        obs[2] *= k
        d = sat - obs
        L = np.linalg.norm(d, axis=0)
        u = d / L
        t = -np.sum(obs * u, axis=0)
        closest = np.linalg.norm(obs + t * u, axis=0)
        return np.atleast_1d(((t > 0.0) & (t < L)) & (closest < a))

    def is_visible(self, atmosphere_km=0.0):
        """True when the satellite is not hidden behind the Earth."""
        blocked = self._blocked(atmosphere_km)
        vis = ~blocked
        return vis if vis.size > 1 else bool(vis[0])

    def observable_fraction(self, atmosphere_km=0.0):
        """Fraction of full-sunlight flux that actually reaches the observer.

        eclipse_type() * is_visible(): zero if the satellite is eclipsed OR
        occulted, otherwise the illuminated fraction. This is the factor a
        rendered flux must be multiplied by.
        """
        return np.asarray(self.eclipse_type()) * np.asarray(
            self.is_visible(atmosphere_km), dtype=float)

    def eclipse_type(self):
        """Fraction of the solar disc visible from the satellite.

        Returns 1.0 in full sunlight, 0.0 in umbra, and the unobscured
        fraction in penumbra. Uses the Sun-satellite-Earth angle; the observer
        plays no part in this.
        """
        sat_pos = np.atleast_2d(self.positions['satellite'].T).T   # (3,) or (3,N)
        incident = np.atleast_2d(self.incident_vector.T).T         # match sat_pos shape
        d = np.linalg.norm(sat_pos, axis=0)

        # Angular radii as seen from the satellite
        theta_earth = np.arcsin(np.clip(R_EARTH_KM / d, -1.0, 1.0))
        theta_sun = np.arcsin(R_SUN_KM / AU)

        # Angular separation of the Earth's centre from the Sun's centre
        to_earth = -sat_pos / d
        cos_sep = np.sum(to_earth * incident, axis=0)
        theta = np.arccos(np.clip(cos_sep, -1.0, 1.0))

        theta = np.atleast_1d(theta)
        theta_earth = np.atleast_1d(theta_earth)

        f = np.ones_like(theta)
        f[theta < theta_earth - theta_sun] = 0.0                 # umbra

        pen = ((theta >= np.abs(theta_earth - theta_sun)) &
               (theta <= theta_earth + theta_sun))
        if np.any(pen):
            r1 = theta_sun                       # solar disc
            r2 = theta_earth[pen] if theta_earth.size > 1 else theta_earth[0]
            dd = theta[pen]
            phi1 = np.arccos(np.clip((dd**2 + r1**2 - r2**2) / (2*dd*r1), -1, 1))
            phi2 = np.arccos(np.clip((dd**2 + r2**2 - r1**2) / (2*dd*r2), -1, 1))
            A = (r1**2 * phi1 + r2**2 * phi2
                 - 0.5*np.sqrt(np.clip((-dd+r1+r2)*(dd+r1-r2)
                                       * (dd-r1+r2)*(dd+r1+r2), 0, None)))
            f[pen] = 1.0 - A / (np.pi * r1**2)

        return f if f.size > 1 else float(f[0])

    def create_satellite_from_elements(self, a_km, e, i_deg, raan_deg,
                                        argp_deg, nu_deg, epoch=None):
        if epoch is None:
            epoch = datetime.datetime.now(datetime.timezone.utc)
        elif epoch.tzinfo is None:
            # A naive datetime used to raise "can't subtract offset-naive and
            # offset-aware datetimes" from the SGP4 epoch arithmetic below.
            # Every time in this class is UTC, so read it as UTC.
            epoch = epoch.replace(tzinfo=datetime.timezone.utc)

        # True anomaly -> mean anomaly
        nu = np.radians(nu_deg)
        E  = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2),
                            np.sqrt(1 + e) * np.cos(nu / 2))
        M  = (E - e * np.sin(E)) % (2 * np.pi)

        # Mean motion in rad/s
        GM = 398600.4418
        n  = np.sqrt(GM / a_km**3)

        # SGP4 epoch: days from 1949-12-31 00:00:00 UTC
        epoch_base = datetime.datetime(1949, 12, 31, tzinfo=datetime.timezone.utc)
        epoch_days = (epoch - epoch_base).total_seconds() / 86400.0

        satrec = Satrec()
        satrec.sgp4init(
            WGS84_gravity, 'i', 99999, epoch_days,
            0.0, 0.0, 0.0,
            e, np.radians(argp_deg), np.radians(i_deg), M,
            n * 60, np.radians(raan_deg),
        )

        self.satellite = EarthSatellite.from_satrec(satrec, self.ts)
        return self.satellite
