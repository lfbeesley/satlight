import satkit as sk
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import os
from load_TLEs import load_tles_and_select, download_tles

def calc_contacts(ground_station, pITRF, time_array):
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

def get_sun_and_gs_vecs(contacts, satellite, time_array):
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

        # -- derive LVLH frame of satellite -- # in theory this could be a sub function if i wanted a different reference frame

        # down
        zhat=-p_ITRF/np.linalg.norm(p_ITRF,axis=1,keepdims=True)

        # orbit normal (angular velocity)
        h_vec=np.cross(p_ITRF,v_ITRF)

        yhat=-h_vec/np.linalg.norm(h_vec,axis=1,keepdims=True)

        # the other one
        xhat=np.cross(yhat,zhat)

        # define rotation matrices at each timestep

        R=np.stack((xhat, yhat, zhat),axis=-1)

        # --- Make my lovely custom vectors in my lovely custom frame--- #

        # calculate the vector (in ITRF) from the satellite to the sun
        sunvecs = p_sun_ITRF-p_ITRF

        # calculate the vector (in ITRF) from the satellite to the ground station
        gsvecs = p_GS_ITRF-p_ITRF

        # rotate the vectors at each timestep to the LVLH frame
        sunvecs_rot = np.einsum('nij,nj->ni', R.transpose(0, 2, 1), sunvecs)
        gsvecs_rot  = np.einsum('nij,nj->ni', R.transpose(0, 2, 1), gsvecs)

        # Store in contact dict
        contact['gs_positions'] = gsvecs_rot
        contact['sun_positions'] = sunvecs_rot

# Select TLE of satellite from file - make sure time is set to the near future
download_tles()
name, line1, line2 = load_tles_and_select()

print(f'Name:   {name}')
print(f'Line 1: {line1}')
print(f'Line 2: {line2}')

tle_lines = [name, line1, line2]

satellite = sk.TLE.from_lines(tle_lines)

# The mininum elevation for a contact
min_elevation_deg = 20

# The date
date = datetime.now()

# Any array of times (seconds)
time_array = np.array([date + timedelta(seconds=x) for x in range(20000)])

# Get satellite positions in TEME frame (pseudo-inertial) via SGP4
pTEME, _vTEME = sk.sgp4(satellite, time_array)

# Get ITRF coordinates (Earth-Fixed) by rotating the position in the TEME frame
# to ITRF frame using the frametransform module
pITRF = np.array([q*x for q,x in zip(sk.frametransform.qteme2itrf(time_array), pTEME)])

# plot ground track
coord = [sk.itrfcoord(p) for p in pITRF]
lats, lons, alts = zip(*[(c.latitude_deg, c.longitude_deg, c.altitude) for c in coord])

# Define ground stations
ground_stations = [
    {'name': 'London', 'lat': 51.4934, 'lon': -0.0098, 'alt': 0},
    {'name': 'Paris', 'lat': 48.8575, 'lon': 2.3514, 'alt': 0},
    # {'name': 'Gibraltar', 'lat': 36.1408, 'lon': -5.3536, 'alt': 0},
    # {'name': 'Svalbard', 'lat': 78.2232, 'lon': 15.6267, 'alt': 0},
    # {'name': 'Alice Springs', 'lat': -23.6980, 'lon': 133.8807, 'alt': 0},
    # {'name': 'Sioux Falls', 'lat': 43.5446, 'lon': -96.7311, 'alt': 0},
]

# --- Calculate ground station contacts --- #

# Calculate the contacts
contacts = [calc_contacts(g, pITRF, time_array) for g in ground_stations]

# Flatten contacts into 1D list
contacts = [item for sublist in contacts for item in sublist]

# Get and save the vectors to the gs and sun during every contact
get_sun_and_gs_vecs(contacts, satellite, time_array)

# --- Split groundtrack into segments where contact with a ground station can be established --- #

# Create a boolean array the same length as time_array, defaulting to False (not in contact)
in_contact = np.zeros(len(time_array), dtype=bool)

# Mark time indices that are in contact
for contact in contacts:
    # Get indices of time_array that are in contact
    start_idx = np.where(time_array == contact['timearr'][0])[0][0]
    end_idx = np.where(time_array == contact['timearr'][-1])[0][0]
    in_contact[start_idx:end_idx+1] = True

# Split ground track into segments based on contact status
segments = []
current_segment = {'lat': [], 'lon': [], 'in_contact': in_contact[0]}

for i in range(len(time_array)):
    if in_contact[i] == current_segment['in_contact']:
        current_segment['lat'].append(lats[i])
        current_segment['lon'].append(lons[i])
    else:
        # Save completed segment and start a new one
        segments.append(current_segment)
        current_segment = {
            'lat': [lats[i]],
            'lon': [lons[i]],
            'in_contact': in_contact[i]
        }

# last segment
segments.append(current_segment)

# --- Plot groundtrack and contacts --- #

fig = go.Figure()

# plot each segment 
for seg in segments:
    fig.add_trace(go.Scattergeo(
        lat=seg['lat'],
        lon=seg['lon'],
        mode='lines',
        line=dict(color='red' if seg['in_contact'] else 'blue'),
        name='Ground Link Available' if seg['in_contact'] else 'No Ground Link Available',
        showlegend=False
    ))

# Add ground station markers
fig.add_trace(go.Scattergeo(
    lat=[gs['lat'] for gs in ground_stations],
    lon=[gs['lon'] for gs in ground_stations],
    mode='markers+text',
    text=[gs['name'] for gs in ground_stations],
    textposition='top center',
    marker=dict(size=8, color='red'),
    name='Ground Stations'
))

fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, title='Ground Track', geo=dict(showland=True, showcountries=True))
fig.show()

# Convert to pandas dataframe
data = pd.DataFrame(contacts)

# --- Format and save as a JSON file ----#

if data.empty:
    print("No contacts with groundstation in given time interval")
else:
    # Convert datetime arrays to ISO strings
    data['timearr'] = data['timearr'].apply(lambda arr: [t.isoformat() for t in arr])

    # Convert arrays to lists
    for col in ['range_km', 'elevation_deg', 'heading_deg']:
        data[col] = data[col].apply(lambda arr: arr.tolist() if isinstance(arr, np.ndarray) else arr)

    # Convert single datetimes to ISO format
    data['start'] = data['start'].apply(lambda t: t.isoformat())
    data['end'] = data['end'].apply(lambda t: t.isoformat())

    # Convert timedelta to seconds
    data['duration'] = data['duration'].apply(lambda d: d.total_seconds())

    # Ensure output directory exists
    os.makedirs('GS_contacts', exist_ok=True)

    # Save to JSON
    data.to_json('GS_contacts/contacts.json', orient='records', indent=2)
    print("Contacts saved to GS_contacts/contacts.json")