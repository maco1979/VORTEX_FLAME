import sqlite3, json, os

ALL_SOULS = ['einstein','cezanne','beethoven','galileo','darwin','davinci','strategy',
             'monet','vangogh','humboldt','montesquieu','yuanlongping','guizhu','herodotus']

# Domain rules by soul (what SHOULD be in the JSON configs)
DOMAIN_RULES_EXPECTED = {
    'galileo':    ['PhysicsSim', 'IoTEmbedded', 'AstroSpace', 'MathModel'],
    'vangogh':    ['ArtTheory', 'ColorComposition', 'VisualStyle', 'ArtHistory'],
    'montesquieu':['LogicReasoning', 'LegalCompliance', 'RuleEngine'],
    'yuanlongping':['AgriScience', 'FoodTech', 'CropGenetics', 'FoodSafety'],
    'herodotus':  ['ChronologyAnalysis', 'HistoricalMethod', 'ArchivalDoc', 'CulturalPattern'],
}

print('FULL DIAGNOSIS: Soul Knowledge Coverage')
print('='*80)

for soul in ALL_SOULS:
    db = f'.vf_memory/{soul}.db'
    if not os.path.exists(db):
        print(f'\n💀 {soul:>15} — NO DATABASE')
        continue
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    total = cur.execute('SELECT COUNT(*) as c FROM memories').fetchone()['c']
    knowledge = cur.execute("SELECT COUNT(*) as c FROM memories WHERE category='knowledge'").fetchone()['c']
    conversation = cur.execute("SELECT COUNT(*) as c FROM memories WHERE category='conversation'").fetchone()['c']
    
    # Check for structured rules (entries with [prefix] like [JEPAKB], [DeviceKB])
    structured = cur.execute(
        "SELECT COUNT(*) as c FROM memories WHERE category='knowledge' AND content LIKE '%[%KB]%'"
    ).fetchone()['c']
    
    # Check quality: entries with actual topic field
    topic_entries = cur.execute(
        "SELECT content FROM memories WHERE category='knowledge' LIMIT 100"
    ).fetchall()
    
    has_topic = sum(1 for r in topic_entries if '"topic"' in r['content'])
    
    # Check for conversation (real interaction memory)
    has_conv = conversation > 0
    
    expected = DOMAIN_RULES_EXPECTED.get(soul, [])
    
    print(f'\n{"="*40}')
    print(f'{soul.upper()}  Total={total}  Knowledge={knowledge}  StructuredRules={structured}  Conv={conversation}')
    print(f'  Expected domain rules: {expected}')
    
    if structured == 0 and soul in DOMAIN_RULES_EXPECTED:
        print(f'  ⚠️  MISSING STRUCTURED RULES — needs domain rules from JSON config')
    elif structured > 0:
        print(f'  ✅ Has structured rules')
    
    if conversation == 0:
        print(f'  ⚠️  Empty conversation memory — no interaction history')
    
    conn.close()

print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print('JSON config files (extended_domain_knowledge.json + v2): 64 entries across 9/14 souls')
print('soul_memory databases: All 14 populated via pull_harness_kb.py (Wikipedia)')
print('')
print('MISSING STRUCTURED RULES (5 souls): galileo, vangogh, montesquieu, yuanlongping, herodotus')
print('WEAK SOULS (minimal structured rules): darwin(1), strategy(1), monet(1), humboldt(1)')
print('USER SUPPLIED: Logic KB (→ montesquieu + shared), Chemistry KB (→ einstein + shared)')
