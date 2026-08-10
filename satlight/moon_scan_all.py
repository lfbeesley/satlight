#!/usr/bin/env python3
"""
Scan your WHOLE TLE catalogue and report every object that passes close
to the Moon in a given window - not just one NORAD ID at a time.

Usage:
    python moon_scan_all.py 3le --start "2026-08-07 20:00" --hours 12 \
        --threshold 3.0 --heavens-above Daily_predictions_for_brighter_satellites.txt

Needs: skyfield, numpy   ->   pip install skyfield numpy
"""
import argparse
import re
from datetime import datetime, timedelta, timezone

import numpy as np
from skyfield.api import load, wgs84


def parse_heavens_above(path):
    """Parse a Heavens-Above 'brighter satellites' text export into a list
    of (norad_id, magnitude, start_time_str, end_time_str) - one entry per
    PASS, not deduped by NORAD ID, since the same object can have several
    passes (and magnitudes) in one export. Times in the file are LOCAL
    (the page header states UTC+01:00) - converted to UTC here assuming
    that fixed offset and the same calendar date as --start.
    NORAD ID is pulled from the satid= param in each entry's passdetails
    URL."""
    with open(path) as f:
        lines = f.readlines()

    entry_start = re.compile(r'^([A-Za-z0-9][\w /\-\.\(\)\+]*?)\t(\d\.\d)\t(\d\d:\d\d:\d\d)')

    passes = []
    current_mag = None
    buf = ""

    def flush(buf, mag):
        if mag is None:
            return
        sid = re.search(r"satid=(\d+)", buf)
        times = re.findall(r"\d\d:\d\d:\d\d", buf)
        if sid and len(times) >= 2:
            passes.append((int(sid.group(1)), mag, times[0], times[-1]))

    for line in lines:
        m = entry_start.match(line)
        if m:
            flush(buf, current_mag)
            current_mag = m.group(2)
            buf = line
        else:
            buf += line
    flush(buf, current_mag)

    return passes


def match_magnitude(passes, norad_id, hit_time_utc, local_offset_hours=1.0):
    """Return the magnitude for a Heavens-Above pass of this NORAD ID whose
    [start, end] window (converted from local to UTC) contains hit_time_utc,
    allowing a few minutes' slack for date/rounding mismatches. Returns
    '?' if no pass of this object's time window matches."""
    hit_date = hit_time_utc.date()
    for pid, mag, start_s, end_s in passes:
        if pid != norad_id:
            continue
        for base_date in (hit_date, hit_date - timedelta(days=1), hit_date + timedelta(days=1)):
            start_local = datetime.combine(base_date, datetime.strptime(start_s, "%H:%M:%S").time())
            end_local = datetime.combine(base_date, datetime.strptime(end_s, "%H:%M:%S").time())
            if end_local < start_local:
                end_local += timedelta(days=1)
            start_utc = (start_local - timedelta(hours=local_offset_hours)).replace(tzinfo=timezone.utc)
            end_utc = (end_local - timedelta(hours=local_offset_hours)).replace(tzinfo=timezone.utc)
            slack = timedelta(minutes=3)
            if start_utc - slack <= hit_time_utc <= end_utc + slack:
                return mag
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tle_file", help="Path to TLE/3LE catalogue file")
    ap.add_argument("--start", required=True, help="UTC start, e.g. '2026-08-07 20:00'")
    ap.add_argument("--hours", type=float, default=8.0, help="Window length in hours")
    ap.add_argument("--step", type=float, default=30.0, help="Coarse step in seconds")
    ap.add_argument("--lat", type=float, default=28.295965)
    ap.add_argument("--lon", type=float, default=-16.565346)
    ap.add_argument("--elev", type=float, default=2131.0)
    ap.add_argument("--threshold", type=float, default=2.0,
                     help="Report passes with separation below this many degrees")
    ap.add_argument("--min-alt", type=float, default=10.0,
                     help="Ignore points where the satellite is below this altitude (deg)")
    ap.add_argument("--max-slew", type=float, default=0.5,
                     help="Max apparent angular rate in deg/s at closest approach")
    ap.add_argument("--allow-shadow", action="store_true",
                     help="Don't require the object to be sunlit (default: sunlit required)")
    ap.add_argument("--heavens-above", default=None,
                     help="Heavens-Above 'brighter satellites' text export - annotates hits with magnitude")
    args = ap.parse_args()

    mag_lookup = parse_heavens_above(args.heavens_above) if args.heavens_above else []

    ts = load.timescale()
    eph = load("de421.bsp")
    earth, moon = eph["earth"], eph["moon"]
    site = wgs84.latlon(args.lat, args.lon, elevation_m=args.elev)

    sats = load.tle_file(args.tle_file)
    print(f"Loaded {len(sats)} objects from {args.tle_file}")

    t0 = datetime.strptime(args.start, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    n_steps = int(args.hours * 3600 / args.step) + 1
    times = ts.utc([t0 + timedelta(seconds=i * args.step) for i in range(n_steps)])

    moon_topo = (earth + site).at(times).observe(moon).apparent()
    _, _, moon_dist = moon_topo.altaz()
    moon_radius_km = 1737.4
    moon_ang_radius_deg = np.degrees(np.arcsin(moon_radius_km / moon_dist.km))

    hits = []
    for sat in sats:
        sat_topo = (sat - site).at(times)
        alt, az, _ = sat_topo.altaz()
        sep_centre = sat_topo.separation_from(moon_topo).degrees
        sep_limb = sep_centre - moon_ang_radius_deg

        visible = alt.degrees > args.min_alt
        if not visible.any():
            continue
        sep_masked = np.where(visible, sep_centre, np.inf)
        idx = int(np.argmin(sep_masked))
        min_sep = sep_masked[idx]

        if min_sep >= args.threshold:
            continue

        # apparent angular rate (deg/s) between consecutive coarse steps,
        # approximate - average rate over --step, not instantaneous
        r = sat_topo.position.km
        r_hat = r / np.linalg.norm(r, axis=0)
        cosang = np.clip(np.sum(r_hat[:, :-1] * r_hat[:, 1:], axis=0), -1, 1)
        rate = np.degrees(np.arccos(cosang)) / args.step
        rate = np.append(rate, rate[-1])  # pad to match length
        slew_at_idx = rate[idx]
        if slew_at_idx > args.max_slew:
            continue

        # sunlit check (skip if in Earth's shadow, unless --allow-shadow)
        if not args.allow_shadow:
            sunlit = sat.at(times[idx]).is_sunlit(eph)
            if not sunlit:
                continue

        hits.append({
            "name": sat.name.strip(),
            "norad": sat.model.satnum,
            "time_utc": times[idx].utc_strftime("%Y-%m-%d %H:%M:%S"),
            "separation_centre_deg": round(min_sep, 3),
            "separation_limb_deg": round(sep_limb[idx], 3),
            "alt_deg": round(alt.degrees[idx], 1),
            "az_deg": round(az.degrees[idx], 1),
            "slew_deg_s": round(float(slew_at_idx), 4),
            "magnitude": match_magnitude(mag_lookup, sat.model.satnum, times[idx].utc_datetime()),
        })

    hits.sort(key=lambda h: h["time_utc"])
    print(f"\n{len(hits)} object(s) within {args.threshold} deg of the Moon's CENTRE "
          f"(slew < {args.max_slew} deg/s, sunlit={'not required' if args.allow_shadow else 'required'}):\n")
    for h in hits:
        transit_flag = "  <-- POSSIBLE TRANSIT" if h["separation_limb_deg"] < 0 else ""
        mag_str = f"mag={h['magnitude']:>4} " if h["magnitude"] != "?" else ""
        print(f"{h['time_utc']}  NORAD {h['norad']:<6d} {h['name']:<25s} {mag_str}"
              f"centre={h['separation_centre_deg']:5.2f} deg  limb={h['separation_limb_deg']:6.3f} deg  "
              f"slew={h['slew_deg_s']:.4f} deg/s  alt={h['alt_deg']:5.1f}  az={h['az_deg']:5.1f}{transit_flag}")

    print("\nNote: coarse scan (step={:.0f}s). Rerun moon_pass.py on the winning NORAD ID "
          "with a fine step to pin down exact closest-approach timing.".format(args.step))


if __name__ == "__main__":
    main()