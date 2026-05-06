import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load graph
g = json.load(open(r'D:\VORTEX_FLAME\industry_knowledge_graph\industry_knowledge_graph.json','r',encoding='utf-8'))

# Load training data - strip BOM manually
f = open(r'D:\VORTEX_FLAME\industry_knowledge_graph\industry_training_data.json', 'rb')
raw = f.read(); f.close()
if raw[:3] == b'\xef\xbb\xbf':
    raw = raw[3:]
# Also try stripping byte 0xFEFF
if raw[:3] == b'\xef\xbb\xbf':
    raw = raw[3:]
# Remove any leading garbage
for i in range(0, 30):
    try:
        d = json.loads(raw[i:].decode('utf-8'))
        break
    except:
        continue

print('=== Cezanne in knowledge graph ===')
for layer_name, layer in g.get('layers', {}).items():
    for node in layer.get('nodes', []):
        sm = node.get('soul_mapping', '')
        bs = node.get('bound_souls', [])
        if sm == 'cezanne' or 'cezanne' in bs:
            name = node.get('name', '?')
            nid = node.get('id', '?')
            print(f"  [{nid}] {name}")
            if node.get('key_techs'):
                print(f"    tech: {', '.join(node['key_techs'])}")
            if node.get('sub_disciplines'):
                print(f"    sub: {', '.join(node['sub_disciplines'][:5])}")

print()
print(f"=== Training data for Cezanne ===")
cz = [x for x in d['data'] if x.get('soul') == 'cezanne']
print(f"Items: {len(cz)}")
for x in cz:
    iid = x.get('industry_id','?')
    txt = x['text'][:180]
    print(f"  [{iid}] {txt}")

# Check f-string for safety
s = "=" * 50
print()
print(s)
print("FIT: Cezanne + IoT/工控/汽车电子")
print(s)
print("""
Cezanne trained: discrete math, graph theory, logic, formal proof,
  data structures, OS, compilers, networking, databases, debugging

IoT/工控 needs: CAN, Modbus, PLC, SCADA, RTOS, MCU, low-power, OTA
  -> Cezanne's OS + networking = perfect base
  -> Domain protocols (CAN/Modbus) can be Stage4 supplements

Ver知识: IoT/工控 = Cezanne primary  [MATCH]
        汽车电子 = Cezanne+Davinci cross [partial MATCH]
""")
