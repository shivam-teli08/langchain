from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import os
load_dotenv()

model = ChatMistralAI(model_name="codestral-2508")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pinecone = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

loadData = PyPDFLoader("data/GRU.pdf")

documents = loadData.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])


index_name = "rag_implementation"

if not pinecone.has_index(index_name):
    pinecone.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )

index = pinecone.get_index(index_name)

#store the embeddings in the index
for i, chunk in enumerate(chunks):
    index.upsert(
        vectors=[{
            "id": f"chunk_{i}",
            "values": embeddings[i],
            "metadata": {"chunk_text": chunk.page_content}
        }]
    )

# retrieve the relevant chunks from the index based on a query
query = "What is the main topic of the document?"
query_embedding = embedding_model.embed_query(query)

results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)
