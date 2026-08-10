#!/usr/bin/env python3
"""
Find when a SPECIFIC satellite (by NORAD ID) passes closest to the Moon,
as seen from a given site, over a time window.

Usage:
    python moon_pass.py 25544 3le/3le --start "2026-08-07 20:00" --hours 12

    25544        = NORAD ID to track (e.g. ISS)
    3le/3le      = your TLE catalogue file (must contain that object)

Needs: skyfield, numpy   ->   pip install skyfield numpy
First run downloads de421.bsp (~17MB) if not already cached - do this
somewhere with internet before you're off-grid.
"""
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
from skyfield.api import load, wgs84


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("norad_id", type=int, help="NORAD catalogue number to track")
    ap.add_argument("tle_file", help="Path to TLE/3LE catalogue file containing that object")
    ap.add_argument("--start", required=True, help="UTC start, e.g. '2026-08-07 20:00'")
    ap.add_argument("--hours", type=float, default=24.0, help="Window length in hours")
    ap.add_argument("--step", type=float, default=5.0, help="Coarse step in seconds")
    ap.add_argument("--lat", type=float, default=28.295965)
    ap.add_argument("--lon", type=float, default=-16.565346)
    ap.add_argument("--elev", type=float, default=2131.0)
    ap.add_argument("--min-alt", type=float, default=10.0, help="Ignore below this altitude (deg)")
    args = ap.parse_args()

    ts = load.timescale()
    eph = load("de421.bsp")
    moon = eph["moon"]
    earth = eph["earth"]
    site = wgs84.latlon(args.lat, args.lon, elevation_m=args.elev)

    sats = load.tle_file(args.tle_file)
    matches = [s for s in sats if s.model.satnum == args.norad_id]
    if not matches:
        print(f"NORAD {args.norad_id} not found in {args.tle_file}")
        return
    sat = matches[0]
    print(f"Tracking {sat.name.strip()} (NORAD {args.norad_id})")

    t0 = datetime.strptime(args.start, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    n_steps = int(args.hours * 3600 / args.step) + 1
    times = ts.utc([t0 + timedelta(seconds=i * args.step) for i in range(n_steps)])

    moon_topo = (earth + site).at(times).observe(moon).apparent()
    sat_topo = (sat - site).at(times)

    alt, az, _ = sat_topo.altaz()
    sep = sat_topo.separation_from(moon_topo).degrees

    visible = alt.degrees > args.min_alt
    if not visible.any():
        print("Object never rises above --min-alt in this window.")
        return

    sep_masked = np.where(visible, sep, np.inf)

    # find local minima (each separate close-approach event, not just the global min)
    events = []
    i = 0
    n = len(sep_masked)
    while i < n:
        if sep_masked[i] < 90:  # only look near reasonable candidate regions
            j = i
            while j < n - 1 and sep_masked[j + 1] <= sep_masked[j]:
                j += 1
            # j is now a local minimum (or edge)
            if j not in [e[0] for e in events]:
                events.append((j, sep_masked[j]))
            i = j + 1
        else:
            i += 1

    if not events:
        events = [(int(np.argmin(sep_masked)), float(np.min(sep_masked)))]

    print(f"\nClosest approach(es) to the Moon, {t0} + {args.hours}h window:\n")
    for idx, sepval in sorted(events, key=lambda e: e[1])[:5]:
        print(f"  {times[idx].utc_strftime('%Y-%m-%d %H:%M:%S')} UTC  "
              f"sep={sepval:6.3f} deg  alt={alt.degrees[idx]:5.1f}  az={az.degrees[idx]:5.1f}")

    best_idx, best_sep = min(events, key=lambda e: e[1])
    print(f"\nClosest overall: {times[best_idx].utc_strftime('%Y-%m-%d %H:%M:%S')} UTC, "
          f"{best_sep:.3f} deg separation.")
    print("(coarse step = {:.0f}s -> rerun with --step 0.5-1 and --hours ~0.05 "
          "centred on this time to pin down an actual transit to sub-second "
          "accuracy)".format(args.step))


if __name__ == "__main__":
    main()