import subprocess, os, sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

target = r'D:\VORTEX_FLAME\.trae\skills\huashu-design'
tmp = r'D:\VORTEX_FLAME\.trae\skills\_huashu_tmp'

# Clean both
for d in [target, tmp]:
    if os.path.exists(d):
        try:
            import shutil
            os.system(f'rmdir /s /q "{d}"')
        except:
            pass

time.sleep(2)

# Clone into temp dir
print("Cloning...")
result = subprocess.run(
    ['git', 'clone', 'https://github.com/alchaincyf/huashu-design.git', tmp],
    capture_output=True, text=True, timeout=120, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
)
print(result.stdout[-500:])
print(result.stderr[-500:])

if os.path.exists(tmp):
    files = sum(1 for _ in os.walk(tmp) for __ in _[2])
    print(f"Cloned {files} files to temp")
    # Move to target
    os.system(f'rmdir /s /q "{target}"')
    time.sleep(1)
    os.rename(tmp, target)
    print(f"Moved to {target}")
else:
    print("FAILED: clone did not create directory")
    print(f"Returncode: {result.returncode}")
