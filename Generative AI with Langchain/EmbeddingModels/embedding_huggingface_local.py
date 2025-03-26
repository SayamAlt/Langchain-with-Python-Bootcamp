from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# text = "Paris is the capital city of France."
documents = [
    "Delhi is capital of India.",
    "Kolkata is the capital of West Bengal.",
    "Mumbai is the financial capital of India.",
    "Paris is the capital of France."
]
# result = embedding.embed_query(text)
result = embedding.embed_documents(documents)
print(str(result)) # Returns a 384 dimensional dense vector space and can be used for tasks including clustering and semantic search.