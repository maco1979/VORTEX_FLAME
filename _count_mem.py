import sqlite3, os
db_dir = '.vf_memory'
souls = ['beethoven','cezanne','einstein','galileo','darwin','davinci','strategy','humboldt','yuanlongping','montesquieu','guizhu','herodotus','monet','vangogh']
total = 0
for s in souls:
    p = os.path.join(db_dir, s + '.db')
    if os.path.exists(p):
        c = sqlite3.connect(p)
        tables = [t[0] for t in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for t in tables:
            n = c.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
            if n > 0:
                total += n
                print(f"  {s}.{t}: {n}")
        c.close()
print(f"\nTOTAL: {total}")

wc_dir = '.vf_world_cache'
if os.path.exists(wc_dir):
    for f in os.listdir(wc_dir):
        if f.endswith('.db'):
            c = sqlite3.connect(os.path.join(wc_dir, f))
            tables = [t[0] for t in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            for t in tables:
                n = c.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
                print(f"  world_cache.{f}.{t}: {n}")
            c.close()
