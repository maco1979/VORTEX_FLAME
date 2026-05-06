import json

fp = r"D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage3_fusion_8k_v2.json"
with open(fp, "r", encoding="utf-8") as f:
    data = json.load(f)

AK = ["agent","terminal","shell","command","pipeline","tool","api call","workflow","automation","script","cron","daemon","service","socket","rpc","cli","bash","powershell","subprocess","exec","function calling","tool use","action","observe","plan","execute","dispatch","orchestrat","curl","wget","ssh","scp","rsync","docker","container","deploy","build","git","commit","branch","merge","awk","sed","grep","find","xargs","pipe","redirect","stdin","stdout","stderr","environment variable","config","flag","argument","makefile","cmake","npm","pip","webhook","http server","argparse","systemd"]

cats = {"LogicCode":0,"Debug":0,"LowLevel":0,"Security":0,"Agent":0,"Other":0}
for d in data:
    text = d.get("instruction","").lower() + d.get("output","").lower()
    if any(k in text for k in ["logic","reason","prove","algorithm","function","code","implement","program","loop","recursion","sort","search","data structure","proof","invariant","compiler","parser","interpreter","type system","induction","complexity","halting","monad","curry-howard","np-complete"]):
        cats["LogicCode"] += 1
    elif any(k in text for k in ["debug","error","bug","fix","trace","crash","exception","fault","gdb","valgrind","sanitizer","segfault","assert","race condition","memory leak"]):
        cats["Debug"] += 1
    elif any(k in text for k in ["kernel","os","memory","cpu","register","interrupt","syscall","process","thread","compile","assembly","pointer","stack","heap","linker","loader","virtual memory","page table","elf","mmap","fork","cache","scheduling","file system","device driver","mutex","semaphore","deadlock","context switch","system call","boot","endian","signal","shared library","ld_library"]):
        cats["LowLevel"] += 1
    elif any(k in text for k in ["security","vulnerability","cve","exploit","buffer overflow","injection","xss","csrf","sanitize","encrypt","auth","permission","privilege","sandbox","aslr"]):
        cats["Security"] += 1
    elif any(k in text for k in AK):
        cats["Agent"] += 1
    else:
        cats["Other"] += 1

total = len(data)
print(f"Total: {total}")
for k, v in cats.items():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")
print(f"---")
print(f"LowLevel:  {cats['LowLevel']/total*100:.1f}% (>=30%) {'PASS' if cats['LowLevel']/total>=0.30 else 'FAIL'}")
print(f"LogicCode: {cats['LogicCode']/total*100:.1f}% (>=60%) {'PASS' if cats['LogicCode']/total>=0.60 else 'FAIL'}")
print(f"Agent:     {cats['Agent']/total*100:.1f}% (>=5% paving) {'PASS' if cats['Agent']/total>=0.05 else 'FAIL'}")
