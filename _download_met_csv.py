import urllib.request
import os

CSV_URL = "https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv"
LOCAL = os.path.join(os.getenv("AI_DATA", r"E:\AI_Data"), "MetObjects.csv")

os.makedirs(os.path.dirname(LOCAL), exist_ok=True)

if os.path.exists(LOCAL):
    mb = os.path.getsize(LOCAL) / (1024 * 1024)
    print(f"Already exists: {mb:.1f}MB")
else:
    print("Downloading MetObjects.csv from GitHub...")
    urllib.request.urlretrieve(CSV_URL, LOCAL)
    mb = os.path.getsize(LOCAL) / (1024 * 1024)
    print(f"Downloaded: {mb:.1f}MB")

print("Verifying CSV structure...")
import csv
with open(LOCAL, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames or []
    row1 = next(reader)
    total = 1
    for _ in reader:
        total += 1

print(f"Headers ({len(headers)}): {headers}")
print(f"Total rows: {total}")
print(f"Sample row keys: {list(row1.keys())[:10]}")
print(f"Is Public Domain: {row1.get('Is Public Domain', 'N/A')}")
print(f"Primary Image: {row1.get('Primary Image', 'N/A')[:80]}")
