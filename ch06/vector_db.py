from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

loader = PyPDFLoader("D:\\books\\document\\llm-ebook-part1.pdf")
pages = loader.load_and_split()

text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=10)
docs = text_splitter.split_documents(pages)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embedding_function)

result = db.search(query="what is llm?", search_type="similarity")
for i in result:
    text = i.page_content.replace('-\n', '')
    text = text.replace('\n', ' ')
    text = text.replace('-', '')
    
    print(text)
    print()
