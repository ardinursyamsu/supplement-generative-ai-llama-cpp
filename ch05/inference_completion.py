from llama_cpp import Llama

llm = Llama(
    model_path="D:\\books\\model\\Phi-3-mini-4k-instruct-q8_0.gguf", 
    n_gpu_layers=-1,
)

prompt = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
How to run for marathon?<|end|>
<|assistant|>
"""
    
output = llm(
    prompt, # Prompt
    max_tokens=1000
)

print(output['choices'][0]['text'])