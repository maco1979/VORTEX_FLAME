#!/usr/bin/env python3
"""Download datasets to E:\AI_Data\
1. CodeCapybara (from GitHub repo)
2. Capybara dataset (ldjnr/capybara)
3. cassanof/Capybara-code
"""
import os, sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

E_DRIVE = r"E:\AI_Data"
os.makedirs(E_DRIVE, exist_ok=True)

print("=" * 60, flush=True)
print("  Dataset Downloader for E:\\AI_Data\\", flush=True)
print("=" * 60, flush=True)

print("\n[1/3] Downloading ldjnr/capybara dataset...", flush=True)
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id="ldjnr/capybara",
        repo_type="dataset",
        local_dir=os.path.join(E_DRIVE, "Capybara"),
    )
    print(f"  Downloaded to: {path}", flush=True)
except Exception as e:
    print(f"  [ERROR] {e}", flush=True)

print("\n[2/3] Downloading cassanof/Capybara-code dataset...", flush=True)
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id="cassanof/Capybara-code",
        repo_type="dataset",
        local_dir=os.path.join(E_DRIVE, "Capybara-code"),
    )
    print(f"  Downloaded to: {path}", flush=True)
except Exception as e:
    print(f"  [ERROR] {e}", flush=True)

print("\n[3/3] Downloading CodeCapybara from GitHub...", flush=True)
try:
    import subprocess
    target = os.path.join(E_DRIVE, "CodeCapybara")
    if not os.path.exists(target):
        result = subprocess.run(
            ["git", "clone", "https://github.com/FSoft-AI4Code/CodeCapybara.git", target],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print(f"  Cloned to: {target}", flush=True)
        else:
            print(f"  [ERROR] git clone failed: {result.stderr[:200]}", flush=True)
    else:
        print(f"  Already exists: {target}", flush=True)
except Exception as e:
    print(f"  [ERROR] {e}", flush=True)

print("\n" + "=" * 60, flush=True)
print("  Download complete!", flush=True)
print("=" * 60, flush=True)
