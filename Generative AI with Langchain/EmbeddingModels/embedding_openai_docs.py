from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32) # The smaller the number of dimensions, the lesser semantic context is captured

documents = [
    "Delhi is capital of India.",
    "Kolkata is the capital of West Bengal.",
    "Mumbai is the financial capital of India.",
    "Paris is the capital of France."
]

result = embedding.embed_documents(documents)
print(str(result))