import os
import time
import requests

TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
TLE_DIR = "./tle_cache"
TLE_FILE = os.path.join(TLE_DIR, "active.txt")

# ------------------------------
# 1. Download TLEs if needed
# ------------------------------
def download_tles():
    os.makedirs(TLE_DIR, exist_ok=True)
    if os.path.exists(TLE_FILE):
        last_modified = os.path.getmtime(TLE_FILE)
        age = time.time() - last_modified
        if age < 7200:  # 2 hours
            print("Using cached TLE file.")
            return
    print("Downloading latest TLE data from Celestrak...")
    response = requests.get(TLE_URL)
    lines = [line.strip() for line in response.text.splitlines() if line.strip()]
    with open(TLE_FILE, "w") as f:
        f.write("\n".join(lines))
    print("TLE file updated.")

# ------------------------------
# 2. Load TLEs and let user pick one
# ------------------------------
def load_tles_and_select():
    with open(TLE_FILE, "r") as f:
        lines = f.readlines()
    
    satellites = []
    for i in range(0, len(lines), 3):
        name = lines[i].strip()
        line1 = lines[i+1].strip()
        line2 = lines[i+2].strip()
        satellites.append((name, line1, line2))

    # Search satellite
    search = input("Enter satellite name or NORAD ID to search: ").strip().lower()
    matches = [
        sat for sat in satellites
        if search in sat[0].lower() or search in sat[1]
    ]
    
    if not matches:
        print("No matches found.")
        exit()
    
    print(f"Found {len(matches)} match(es):")
    for i, sat in enumerate(matches):
        print(f"{i}: {sat[0]}")
    
    selected_index = int(input("Select a satellite by number: "))
    return matches[selected_index]