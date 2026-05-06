import sys, os
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
print("Import torch OK", flush=True)

from transformers import Mistral3ForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
print("Import transformers OK", flush=True)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
print("BNB config OK", flush=True)

print("Loading model...", flush=True)
model = Mistral3ForConditionalGeneration.from_pretrained(
    r"D:\models\Ministral-8B-Reasoning",
    quantization_config=bnb,
    device_map="auto",
    dtype=torch.float16,
)
vram = torch.cuda.memory_reserved() / 1024**3
print("Model loaded! VRAM: %.1fGB" % vram, flush=True)

processor = AutoProcessor.from_pretrained(r"D:\models\Ministral-8B-Reasoning")
tokenizer = processor.tokenizer
print("Processor OK", flush=True)

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
print("Tokenize OK", flush=True)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20)
answer = tokenizer.decode(out[0], skip_special_tokens=True)
print("Generate OK: %s" % answer, flush=True)

del model
torch.cuda.empty_cache()
print("Done!", flush=True)
