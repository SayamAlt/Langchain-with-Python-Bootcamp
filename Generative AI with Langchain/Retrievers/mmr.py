# Maximal Marginal Relevance (MMR) information retrieval technique - A type of retriever based on search algorithm
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os, warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embedding_model = OpenAIEmbeddings()


vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# Enable MMR in the retriever
retriever = vectorstore.as_retriever(
    search_type="mmr", # This enables MMR search
    search_kwargs={"k": 3, 'lambda_mult': 0.5} # k = top results, lambda_mult = MMR parameter, relevance-diversity balance 1 -> behaves exactly as regular search technique, 0 -> returns highly distinct documents
)

query = "What is Langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)