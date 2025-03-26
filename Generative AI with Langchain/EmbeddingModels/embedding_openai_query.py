from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32) # The smaller the number of dimensions, the lesser semantic context is captured
result = embedding.embed_query("Copenhagen is the capital of Denmark.")
print(str(result))