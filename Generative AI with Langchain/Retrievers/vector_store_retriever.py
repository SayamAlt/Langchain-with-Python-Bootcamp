from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os, warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")


# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name='my_collection'
)

# Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is Chroma used for?"
results = retriever.invoke(query) 

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}") # Truncate the final results for display

# Main Benefit: Advanced retrievers can be connected with vectorstores for implementing advanced search strategies.
# results = vectorstore.similarity_search(query, k=2)

# for i, doc in enumerate(results):
#     print(f"\n--- Result {i+1} ---")
#     print(f"Content:\n{doc.page_content}")