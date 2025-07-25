
import numpy as np
import satkit as sk
from datetime import datetime, timedelta

def calc_observations(ground_station, pITRF, time_array, min_elevation_deg):
    """
    Compute contact times for a single ground station given satellite position in Earth-fixed frame
    """

    # Create an "itrfcoord" object for the ground statoin
    coord = sk.itrfcoord(latitude_deg=ground_station['lat'], longitude_deg=ground_station['lon'], altitude_m=ground_station['alt'])

    # Get the North-East-Down coordinates of the satellite relative to the ground station
    # at all times by taking the difference between the satellite position and the ground
    # coordinated, then rotating to the "North-East-Down" frame relative to the ground station
    pNED = np.array([coord.qned2itrf.conj * (x - coord.vector) for x in pITRF])

    # Normalize the NED coordinates
    pNED_hat = pNED / np.linalg.norm(pNED, axis=1)[:, None]

    # Find the elevation from the ground station at all times
    # This is the arcsign of the "up" portion of the NED-hat vetor
    elevation_deg = np.degrees(np.arcsin(-pNED_hat[:,2]))

    # We can see ground station when elevation is greater than min_elevation_deg
    inview_idx = np.argwhere(elevation_deg > min_elevation_deg).flatten().astype(int)

    # return empty list if no contacts
    if len(inview_idx) == 0:
        return []
    # split indices into groups of consecutive indices
    # This indicates contiguous contacts
    inview_idx = np.split(inview_idx, np.where(np.diff(inview_idx) != 1)[0]+1)

    def get_single_contacts(inview_idx):
        for cidx in inview_idx:
            # cidx are indices to the time array for this contact

            # the North-East-Down position of the satellite relative to
            # ground station over the single contact
            cpNED = pNED[cidx,:]

            # Compute the range in meters
            range = np.linalg.norm(cpNED, axis=1)

            # elevation in degrees over the contact
            contact_elevation_deg = elevation_deg[cidx]

            # Heading clockwise from North is arctangent of east/north'
            heading_deg = np.degrees(np.arctan2(cpNED[:,1], cpNED[:,0]))

            # Yield a dictionary describing the results
            yield {
                'groundstation': ground_station['name'],
                'timeindices': cidx,
                'timearr': time_array[cidx],
                'range_km': range*1.0e-3,
                'elevation_deg': contact_elevation_deg,
                'heading_deg': heading_deg,
                'start': time_array[cidx[0]],
                'end': time_array[cidx[-1]],
                'max_elevation_deg': np.max(contact_elevation_deg),
                'duration': time_array[cidx[-1]] - time_array[cidx[0]]
            }
    return list(get_single_contacts(inview_idx))

def get_vecs(contacts, satellite, time_array, coordinate_system,ground_stations):
    """
    Calculate the vectors to the sun and the ground station at every timestep during contacts
    """

    for contact in contacts:

        contact_time_array=time_array[contact['timeindices']]

        gs = next(gs for gs in ground_stations if gs['name'] == contact['groundstation'])
        # Get satellite positions in TEME frame (pseudo-inertial) via SGP4
        pTEME, vTEME = sk.sgp4(satellite, contact_time_array)

        # satellite position and velocity conversion from TEME to ITRF
        p_ITRF = np.array([q*x for q,x in zip(sk.frametransform.qteme2itrf(contact_time_array), pTEME)])
        v_ITRF = np.array([q*x for q,x in zip(sk.frametransform.qteme2itrf(contact_time_array), vTEME)])

        #sun positions in GCRF
        sun_positions = np.array([sk.jplephem.geocentric_pos(sk.solarsystem.Sun, t) for t in contact_time_array])

        #convert sun positions to ITRF
        p_sun_ITRF = np.array([q*x for q,x in zip(sk.frametransform.qgcrf2itrf(contact_time_array), sun_positions)])

        #create coord object for gs
        coord = sk.itrfcoord(latitude_deg=gs['lat'], longitude_deg=gs['lon'], altitude_m=gs['alt'])

        #get ITRF coordinate of the gs
        p_GS_ITRF_1=np.array([coord.vector])

        p_GS_ITRF=np.repeat(p_GS_ITRF_1,len(contact_time_array),axis=0)
        
        # calculate the vector (in ITRF) from the satellite to the sun
        sunvecs_ITRF = p_sun_ITRF-p_ITRF

        # calculate the vector (in ITRF) from the satellite to the ground station
        gsvecs_ITRF = p_GS_ITRF-p_ITRF
        
        # -- eclipse determination -- #
        def eclipse(sat_pos, sun_pos,
                       earth_radius=sk.consts.earth_radius,
                       sun_radius=696000000.0):
            """
            Returns scalar illumination from 0 (fully eclipsed) to 1 (fully illuminated).
            
            Parameters:
            - sat_pos: (N,3) satellite positions in ITRF [m]
            - sun_pos: (N,3) sun positions in ITRF [m]
            
            Returns:
            - illumination: (N,) float array in [0, 1]
            """
            sat_pos = np.asarray(sat_pos)
            sun_pos = np.asarray(sun_pos)
            
            r_vec = -sat_pos
            s_vec = sun_pos - sat_pos
            
            r_norm = np.linalg.norm(r_vec, axis=1)
            s_norm = np.linalg.norm(s_vec, axis=1)
            
            cos_theta = np.sum(r_vec * s_vec, axis=1) / (r_norm * s_norm)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)  # angular separation in radians
            
            theta_E = np.arcsin(np.clip(earth_radius / r_norm, -1.0, 1.0))  # Earth angular radius
            theta_S = np.arcsin(np.clip(sun_radius / s_norm, -1.0, 1.0))    # Sun angular radius
            
            # Initialize output
            illumination = np.ones_like(theta)
            
            # Fully eclipsed
            full_eclipse = (theta <= np.abs(theta_E - theta_S)) & (theta_E > theta_S)
            illumination[full_eclipse] = 0.0

            # Partial overlap
            partial = (theta < (theta_E + theta_S)) & ~full_eclipse
            d = theta[partial]
            R1 = theta_S[partial]
            R2 = theta_E[partial]

            # Compute intersection area
            a1 = R1**2 * np.arccos(np.clip((d**2 + R1**2 - R2**2) / (2 * d * R1), -1.0, 1.0))
            a2 = R2**2 * np.arccos(np.clip((d**2 + R2**2 - R1**2) / (2 * d * R2), -1.0, 1.0))
            a3 = 0.5 * np.sqrt(
                np.clip(
                    (-d + R1 + R2) * (d + R1 - R2) * (d - R1 + R2) * (d + R1 + R2),
                    0.0, np.inf
                )
            )
            A = a1 + a2 - a3
            illum = 1 - A / (np.pi * R1**2)
            
            illumination[partial] = illum
            return illumination

        # --- Define the coordinate system and transform the sun and gs vectors into the one chosen --- #

        def LVLH(p_ITRF, v_ITRF):
            """
            Derive the local vertical-local horizontal (LVLH) reference frame and define the 
            sun and gs vectors using the satellite as the origin.
            This is for active satellites which will be maintaining an Earth-pointing attitude.
            """
            # down
            zhat=-p_ITRF/np.linalg.norm(p_ITRF,axis=1,keepdims=True)

            # orbit normal (angular velocity)
            h_vec=np.cross(p_ITRF,v_ITRF)

            yhat=-h_vec/np.linalg.norm(h_vec,axis=1,keepdims=True)

            # the other one
            xhat=np.cross(yhat,zhat)

            # define rotation matrices at each timestep
            R=np.stack((xhat, yhat, zhat),axis=-1)

            # --- Rotate my lovely vectors into my lovely custom frame--- #

            sunvecs_LVLH = np.einsum('nij,nj->ni', R.transpose(0, 2, 1), sunvecs_ITRF)
            gsvecs_LVLH  = np.einsum('nij,nj->ni', R.transpose(0, 2, 1), gsvecs_ITRF)

            # Store in contact dict
            contact['gs_positions'] = gsvecs_LVLH
            contact['sun_positions'] = sunvecs_LVLH

        def CIRS(sunvecs_ITRF, gsvecs_ITRF, contact_time_array):
            """
            Define the sun and gs vectors in the Celestial Intermediate Reference System (CIRS) 
            using the satellite as the origin.
            This is for inactive satellites which will be tumbling, assuming principle rotation
            axis pointing at a point on the celestial sphere.
            """

            sunvecs_TIRS= np.array([q*x for q,x in zip(sk.frametransform.qitrf2tirs(contact_time_array), sunvecs_ITRF)])
            sunvecs_CIRS= np.array([q*x for q,x in zip(sk.frametransform.qtirs2cirs(contact_time_array), sunvecs_ITRF)])

            gsvecs_TIRS= np.array([q*x for q,x in zip(sk.frametransform.qitrf2tirs(contact_time_array), gsvecs_ITRF)])
            gsvecs_CIRS= np.array([q*x for q,x in zip(sk.frametransform.qtirs2cirs(contact_time_array), gsvecs_ITRF)])

            contact['gs_positions'] = gsvecs_CIRS
            contact['sun_positions'] = sunvecs_CIRS

        if coordinate_system == 'LVLH':
            LVLH(p_ITRF,v_ITRF)
        elif coordinate_system == 'CIRS':
            CIRS(sunvecs_ITRF, gsvecs_ITRF, contact_time_array)

        # finally determine if eclipsed
        contact['sunlit'] = eclipse(p_ITRF,p_sun_ITRF)

