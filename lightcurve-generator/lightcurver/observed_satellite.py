
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import satkit as sk

from datetime import timedelta

from lightcurver.utils import (
    calc_observations, get_vecs
)

class ObservedSatellite:
    def __init__(self, tle_lines, start_time, num_secs):
        self.sat = sk.TLE.from_lines(tle_lines)
        self.time_array = np.array([start_time + timedelta(seconds=x) for x in range(num_secs)])
        self.pTEME, _ = sk.sgp4(self.sat, self.time_array)
        self.pITRF = np.array([
            q * x for q, x in zip(sk.frametransform.qteme2itrf(self.time_array), self.pTEME)
        ])
        self.contacts = []

    def get_contacts(self, ground_stations, min_elevation_deg):
        self.contacts = [
            c for gs in ground_stations
            for c in calc_observations(gs, self.pITRF, self.time_array, min_elevation_deg)
        ]
        return self.contacts

    def compute_vectors(self, ground_stations, frame='CIRS'):
        if not self.contacts:
            raise RuntimeError("Run get_contacts first.")
        get_vecs(self.contacts, self.sat, self.time_array, frame, ground_stations)

    def plot_groundtrack(self, ground_stations):
        in_contact = np.zeros(len(self.time_array), dtype=bool)
        for contact in self.contacts:
            start_idx = np.where(self.time_array == contact['timearr'][0])[0][0]
            end_idx = np.where(self.time_array == contact['timearr'][-1])[0][0]
            in_contact[start_idx:end_idx+1] = True

        coords = [sk.itrfcoord(p) for p in self.pITRF]
        lats, lons, _ = zip(*[(c.latitude_deg, c.longitude_deg, c.altitude) for c in coords])

        segments = []
        current_segment = {'lat': [], 'lon': [], 'in_contact': in_contact[0]}

        for i in range(len(self.time_array)):
            if in_contact[i] == current_segment['in_contact']:
                current_segment['lat'].append(lats[i])
                current_segment['lon'].append(lons[i])
            else:
                segments.append(current_segment)
                current_segment = {'lat': [lats[i]], 'lon': [lons[i]], 'in_contact': in_contact[i]}
        segments.append(current_segment)

        fig = go.Figure()
        for seg in segments:
            fig.add_trace(go.Scattergeo(
                lat=seg['lat'],
                lon=seg['lon'],
                mode='lines',
                line=dict(color='red' if seg['in_contact'] else 'blue'),
                showlegend=False
            ))
        fig.add_trace(go.Scattergeo(
            lat=[gs['lat'] for gs in ground_stations],
            lon=[gs['lon'] for gs in ground_stations],
            mode='markers+text',
            text=[gs['name'] for gs in ground_stations],
            textposition='top center',
            marker=dict(size=8, color='red'),
        ))
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, title='Ground Track', geo=dict(showland=True, showcountries=True))
        fig.show()

    def save_contacts(self, outpath):
        data = pd.DataFrame(self.contacts)
        if data.empty:
            print("No contacts")
            return
        data['timearr'] = data['timearr'].apply(lambda arr: [t.isoformat() for t in arr])
        for col in ['range_km', 'elevation_deg', 'heading_deg']:
            data[col] = data[col].apply(lambda arr: arr.tolist() if isinstance(arr, np.ndarray) else arr)
        data['start'] = data['start'].apply(lambda t: t.isoformat())
        data['end'] = data['end'].apply(lambda t: t.isoformat())
        data['duration'] = data['duration'].apply(lambda d: d.total_seconds())
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        data.to_json(outpath, orient='records', indent=2)
        print('Contact data saved')
