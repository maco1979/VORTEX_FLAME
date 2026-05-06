import json, random

fp = r"D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage3_fusion_8k_v2.json"
with open(fp, "r", encoding="utf-8") as f:
    data = json.load(f)

LOWLEVEL_KW = ["kernel","memory","cpu","register","interrupt","syscall","process","thread","compile","assembly","pointer","stack","heap","linker","loader","virtual memory","page table","elf","mmap","fork","cache","scheduling","file system","device driver","mutex","semaphore","deadlock","context switch","system call","boot","endian","signal","shared library","inode","ext4","cgroup","namespace","driver","ioctl","page fault","tlb","paging","swap","oom","seccomp","chroot","mount","blkid","fdisk","mkfs","fsck","iostat","vmstat","netlink","kernel module","memory allocation","address space","segment","protection","permission","privilege","ring ","user mode","kernel mode","context","scheduler","preempt","concurrent","parallel","synchron","atomic","lock","barrier","futex","spinlock","rcu","workqueue","softirq","hardirq","irq","poll","epoll","select","socket","tcp","udp","ip ","network","packet","routing","firewall","iptables","nftables","bridge","vlan","tunnel","vpn","dns","dhcp","arp","mac address","nic","eth","wifi","wireless","bluetooth","usb","pci","acpi","bios","uefi","grub","initramfs","systemd","journalctl","dmesg","kmsg","strace","ltrace","lsof","ss","netstat","top","htop","ps","kill","nice","renice","ulimit","prctl"]

AGENT_KW = ["agent","terminal","shell","command","pipeline","tool","api call","workflow","automation","script","cron","daemon","service","rpc","cli","bash","powershell","subprocess","exec","function calling","tool use","action","observe","dispatch","orchestrat","curl","wget","ssh","scp","rsync","docker","container","deploy","makefile","cmake","npm","pip","webhook","http server","argparse","systemd","git","commit","branch","merge","sed","awk","grep","find","xargs","pipe","redirect","stdin","stdout","stderr"]

DEBUG_KW = ["debug","error","bug","fix","trace","crash","exception","fault","gdb","valgrind","sanitizer","segfault","assert","race condition","memory leak","breakpoint","backtrace","core dump","diagnostic"]

SECURITY_KW = ["security","vulnerability","cve","exploit","buffer overflow","injection","xss","csrf","sanitize","encrypt","auth","permission","privilege","sandbox","aslr","harden","penetration","attack","threat","risk"]

LOGIC_CODE_KW = ["logic","reason","prove","algorithm","code","implement","program","loop","recursion","sort","search","data structure","proof","invariant","compiler","parser","interpreter","type system","induction","complexity","halting","monad","np-complete","function","binary search","recursion","dynamic programming","greedy","divide and conquer","hash","linked list","queue","tree","graph","formal verification","curry-howard","lambda","closure","higher-order","polymorphism","abstract","interface","design pattern","refactor","optimize","time complexity","space complexity","big o","asymptotic"]

def classify(item):
    text = item.get("instruction","").lower() + " " + item.get("output","").lower()
    if any(k in text for k in LOWLEVEL_KW):
        return "LowLevel"
    if any(k in text for k in AGENT_KW):
        return "Agent"
    if any(k in text for k in DEBUG_KW):
        return "Debug"
    if any(k in text for k in SECURITY_KW):
        return "Security"
    if any(k in text for k in LOGIC_CODE_KW):
        return "LogicCode"
    return "Other"

cats = {"LogicCode":0,"Debug":0,"LowLevel":0,"Security":0,"Agent":0,"Other":0}
other_samples = []
for d in data:
    c = classify(d)
    cats[c] += 1
    if c == "Other" and len(other_samples) < 5:
        other_samples.append(d.get("instruction","")[:100])

total = len(data)
print(f"Total: {total}")
for k, v in cats.items():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")
print(f"---")
print(f"LowLevel:  {cats['LowLevel']/total*100:.1f}% (>=30%) {'PASS' if cats['LowLevel']/total>=0.30 else 'FAIL'}")
print(f"LogicCode: {cats['LogicCode']/total*100:.1f}% (>=60%) {'PASS' if cats['LogicCode']/total>=0.60 else 'FAIL'}")
print(f"Agent:     {cats['Agent']/total*100:.1f}% (>=5% paving) {'PASS' if cats['Agent']/total>=0.05 else 'FAIL'}")
print(f"\nOther samples:")
for s in other_samples:
    print(f"  - {s}")
