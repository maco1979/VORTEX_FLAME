import sqlite3, os, json

ALL_SOULS = ['einstein','cezanne','beethoven','galileo','darwin','davinci','strategy',
             'monet','vangogh','humboldt','montesquieu','yuanlongping','guizhu','herodotus']

print('='*75)
print('SOUL MEMORY DEEP AUDIT — Direct SQLite Query')
print('='*75)

for soul in ALL_SOULS:
    db_path = f'.vf_memory/{soul}.db'
    if not os.path.exists(db_path):
        print(f'\nNO DB: {soul}')
        continue
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    total = cur.execute('SELECT COUNT(*) as c FROM memories').fetchone()['c']
    knowledge = cur.execute("SELECT COUNT(*) as c FROM memories WHERE category='knowledge'").fetchone()['c']
    conversation = cur.execute("SELECT COUNT(*) as c FROM memories WHERE category='conversation'").fetchone()['c']
    todo = cur.execute("SELECT COUNT(*) as c FROM memories WHERE category='todo'").fetchone()['c']

    status = 'OK' if knowledge > 0 else 'EMPTY'
    print(f'\n{status} {soul:>15} Total={total}  Knowledge={knowledge}  Conv={conversation}  Todo={todo}')

    if knowledge > 0:
        rows = cur.execute("SELECT content FROM memories WHERE category='knowledge' LIMIT 20").fetchall()
        for r in rows:
            c = json.loads(r['content'])
            topic = c.get('topic', str(c)[:80])
            print(f'     {topic[:120]}')
    else:
        print(f'     *** ZERO KNOWLEDGE ***')

    conn.close()
