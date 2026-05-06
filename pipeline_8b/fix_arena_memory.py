import sys
f=open(r'D:\VORTEX_FLAME\pipeline_8b\arena_7b_vs_8b.py','r',encoding='utf-8')
c=f.read(); f.close()

reps = [
    ('generate(model_7b, tok_7b, prompt, max_new_tokens=1024)', "generate(model_7b, tok_7b, '7b', prompt, max_new_tokens=1024)"),
    ('generate(model_8b, tok_8b, prompt, max_new_tokens=1024)', "generate(model_8b, tok_8b, '8b', prompt, max_new_tokens=1024)"),
    ('generate(model_7b, tok_7b, attack_prompt, max_new_tokens=512)', "generate(model_7b, tok_7b, '7b', attack_prompt, max_new_tokens=512)"),
    ('generate(model_8b, tok_8b, attack_prompt, max_new_tokens=512)', "generate(model_8b, tok_8b, '8b', attack_prompt, max_new_tokens=512)"),
    ('generate(model_8b, tok_8b, defense_prompt_8b, max_new_tokens=512)', "generate(model_8b, tok_8b, '8b', defense_prompt_8b, max_new_tokens=512)"),
    ('generate(model_7b, tok_7b, defense_prompt_7b, max_new_tokens=512)', "generate(model_7b, tok_7b, '7b', defense_prompt_7b, max_new_tokens=512)"),
    ('generate(model_7b, tok_7b, q["q"], max_new_tokens=1024)', "generate(model_7b, tok_7b, '7b', q[\"q\"], max_new_tokens=1024)"),
    ('generate(model_8b, tok_8b, q["q"], max_new_tokens=1024)', "generate(model_8b, tok_8b, '8b', q[\"q\"], max_new_tokens=1024)"),
]

for old, new in reps:
    if old in c:
        c = c.replace(old, new)
        print(f'Fixed: {old[:60]}')
    else:
        print(f'NOT FOUND: {old[:60]}')

f=open(r'D:\VORTEX_FLAME\pipeline_8b\arena_7b_vs_8b.py','w',encoding='utf-8')
f.write(c); f.close()
print('Saved')
