import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

out_dir = "D:\\Phi-3-mini-4k-instruct"

model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)