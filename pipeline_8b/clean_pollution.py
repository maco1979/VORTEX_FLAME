import json

POLLUTION_KEYWORDS = ['DeepSeek', '深度思考（DeepSeek）', 'AI学习工具', 'Kimi则在文科', '豆包等AI工具']

def clean_file(filepath, label):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_count = len(data['data'])
    polluted = []
    clean = []

    for i, item in enumerate(data['data']):
        text = item.get('text', '')
        if any(kw in text for kw in POLLUTION_KEYWORDS):
            polluted.append(i)
        else:
            clean.append(item)

    removed = original_count - len(clean)
    print(f'\n=== {label} ===')
    print(f'原始条数: {original_count}')
    print(f'污染条数: {removed}')
    print(f'清理后条数: {len(clean)}')

    if removed > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'data': clean}, f, ensure_ascii=False, indent=2)
        print(f'已保存清理后文件')
        print(f'污染行索引: {polluted[:10]}{"..." if len(polluted) > 10 else ""}')
    else:
        print('无污染，跳过')

clean_file(r'd:\VORTEX_FLAME\soul_training_data\einstein\einstein_hq_10k_v3.json', 'Einstein v3')
clean_file(r'd:\VORTEX_FLAME\soul_training_data\einstein\einstein_hq_10k_v2_leveled.json', 'Einstein v2')
clean_file(r'd:\VORTEX_FLAME\soul_training_data\guizhu\guizhu_hq_10k.json', 'Guizhu')
