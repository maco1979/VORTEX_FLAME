import shutil, os, urllib.request, zipfile, tempfile, glob, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

target = r'D:\VORTEX_FLAME\.trae\skills\huashu-design'

# Clean stale clone
if os.path.exists(target):
    os.system(f'rmdir /s /q "{target}"')
    import time; time.sleep(2)

# Try git first, fallback to zip
import subprocess
result = subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/alchaincyf/huashu-design.git', target],
    capture_output=True, text=True, timeout=60)
print(result.stdout[-200:] if result.stdout else "")
print(result.stderr[-200:] if result.stderr else "")

if os.path.exists(os.path.join(target, 'SKILL.md')) or os.path.exists(os.path.join(target, 'README.md')):
    print(f"OK: cloned successfully")
else:
    print(f"Clone may have issues, checking files...")

if os.path.exists(target):
    files = []
    for root, dirs, filenames in os.walk(target):
        for f in filenames:
            files.append(os.path.join(root, f))
    print(f"Total files: {len(files)}")
    for f in sorted(files)[:20]:
        rel = os.path.relpath(f, target)
        sz = os.path.getsize(f)
        print(f"  {rel} ({sz}B)")
else:
    print("ERROR: target directory does not exist")
