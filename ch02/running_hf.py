import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device ='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

ckpt = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

prompt = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
How to create a bagel?<|end|>
<|assistant|>
"""
inputs = tokenizer(
    prompt, 
    return_tensors="pt", 
    return_attention_mask=False
)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)