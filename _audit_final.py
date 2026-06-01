import sqlite3, json, os

souls = [
    'einstein','cezanne','beethoven','galileo','darwin','davinci','strategy',
    'monet','vangogh','humboldt','montesquieu','yuanlongping','guizhu','herodotus'
]

print('COMPLETE SOUL KNOWLEDGE AUDIT — BEFORE v3')
print('='*65)

for soul in souls:
    db = f'.vf_memory/{soul}.db'
    if not os.path.exists(db):
        print(f'  {soul:>15}: NO DB')
        continue
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    total = cur.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
    kw = cur.execute("SELECT COUNT(*) FROM memories WHERE category='knowledge'").fetchone()[0]
    conv = cur.execute("SELECT COUNT(*) FROM memories WHERE category='conversation'").fetchone()[0]
    struct = cur.execute("SELECT COUNT(*) FROM memories WHERE category='knowledge' AND content LIKE '%KB]%'").fetchone()[0]
    conn.close()
    
    if struct <= 10:
        bar = chr(9608)*struct + chr(9617)*(10-struct)
    else:
        bar = chr(9608)*10
    status = 'RULES_OK' if struct > 0 else 'WIKI_ONLY'
    print(f'  {status:>10} {soul:>15}  total={total:>6}  kw={kw:>6}  rules=[{bar}] x{struct}  conv={conv}')

print()
print('SCIENCE COVERAGE vs UNESCO GB/T13745 7 Basic Sciences')
print('='*65)
print('  Math       — need structured rules via Einstein')
print('  Physics    — need structured rules via Einstein')
print('  Chemistry  — NEEDS ChemKB (user supplied, pending v3)')
print('  Logic      — NEEDS LogicKB (user supplied, pending v3)')
print('  Astronomy  — MISSING: needs Galielo AstroSpace rules')
print('  Earth Sci  — MISSING: needs Humboldt ClimateModel/GeoSpatial')
print('  Biology    — MISSING: needs Darwin BioLab/GenomeAnalysis')
print()
print('HUMANITIES COVERAGE')
print('='*65)
print('  Literature — MISSING: needs LitKB rules')
print('  History    — MISSING: needs Herodotus HistoricalMethod rules')
print('  Philosophy — MISSING: needs Guizhu/Montesquieu PhilKB rules')
print()
print('SOUL STRUCTURED RULES STATUS:')
print('  HAVE RULES: einstein(4), cezanne(9), beethoven(10), davinci(2)')
print('  ZERO RULES, WIKI ONLY: galileo, darwin, strategy, monet, vangogh,')
print('                          humboldt, montesquieu, yuanlongping, guizhu, herodotus')
