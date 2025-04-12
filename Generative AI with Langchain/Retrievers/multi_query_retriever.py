from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import os, warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings()

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(
    documents=all_docs, 
    embedding=embedding_model
)

# Create retrievers
similarity_retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
    llm=ChatOpenAI(api_key=api_key)
)

# Query
query = "How to improve energy levels and maintain balance?"


# Retrieve results
similarity_results = similarity_retriever.invoke(query)
multi_query_results = multi_query_retriever.invoke(query)

# Print results
print("Similarity Search Results")
for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}")

print("*"*150)

print("Multi-Query Results")
for i, doc in enumerate(multi_query_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

