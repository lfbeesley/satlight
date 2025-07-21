import requests

def fetch_tle_by_norad_id(norad_id):
    """
    Fetches the TLE for a satellite using its NORAD ID via Celestrak.
    WARNING: For historical TLEs, pull from https://www.space-track.org/#gp
    """
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
    response = requests.get(url)
    content = response.text.strip()

    if response.status_code != 200:
        raise Exception(f"HTTP error {response.status_code} while contacting Celestrak.")

    if content == "No GP data found":
        raise Exception(f"No TLE found for NORAD ID {norad_id}. It may be incorrect or unavailable.")

    tle_lines = content.splitlines()
    if len(tle_lines) == 3:
        return tle_lines  # [name, line1, line2]
    else:
        raise Exception("Unexpected TLE format returned.")

if __name__ == "__main__":
    try:
        tle = fetch_tle_by_norad_id(25544)
        print("\n".join(tle))
    except Exception as e:
        print(e)
