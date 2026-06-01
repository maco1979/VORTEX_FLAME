import urllib.request
import os

CSV_URL = "https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv"
LOCAL = r"D:\VORTEX_FLAME\data\MetObjects.csv"

os.makedirs(os.path.dirname(LOCAL), exist_ok=True)

if os.path.exists(LOCAL):
    mb = os.path.getsize(LOCAL) / (1024 * 1024)
    print(f"Already exists: {mb:.1f}MB")
else:
    print("Downloading MetObjects.csv from GitHub...")
    urllib.request.urlretrieve(CSV_URL, LOCAL)
    mb = os.path.getsize(LOCAL) / (1024 * 1024)
    print(f"Downloaded: {mb:.1f}MB")
