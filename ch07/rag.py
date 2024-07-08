from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama

loader = PyPDFLoader("D:\\books\\document\\llm-ebook-part1.pdf")
pages = loader.load_and_split()

text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=10)
docs = text_splitter.split_documents(pages)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embedding_function)

question = "What is the best use of large language model (LLM)?"
context = ""

result = db.search(query=question, search_type="similarity")
for i in result:
    text = i.page_content.replace('-\n', '')
    text = text.replace('\n', ' ')
    text = text.replace('-', '')

    context += text


prompt = f"""<|system|>
You are a helpful AI assistant. Your task is to answer questions based on the provided context. 
Always approach the answer step-by-step, ensuring accuracy and clarity.<|end|>
<|user|>
Context:
{context}

Question: {question}<|end|>
<|assistant|>
 """

llm = Llama(
    model_path="D:\\books\\model\\Phi-3-mini-4k-instruct-q8_0.gguf", 
    n_gpu_layers=-1,
    n_ctx=4096
)
    
output = llm(
    prompt, 
    max_tokens=1000,  
    stream=True
)

for x in output: 
    print(x['choices'][0]['text'], end="", flush=True)