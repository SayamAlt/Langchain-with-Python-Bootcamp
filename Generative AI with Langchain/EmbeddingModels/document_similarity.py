from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "The sun rises in the east and sets in the west, marking the passage of time.",
    "A cat purrs when it is happy and content, often seeking warmth and affection.",
    "Machine learning algorithms analyze patterns in data to make predictions and decisions.",
    "The Eiffel Tower in Paris is an iconic landmark visited by millions of tourists annually.",
    "Photosynthesis allows plants to convert sunlight into energy, producing oxygen as a byproduct."
]

query = "Which famous landmark is located in Paris?"

doc_embeddings = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], doc_embeddings)[0] # Pass query embedding in 2D list

index, score = sorted(list(enumerate(similarity_scores)),key=lambda x: x[1])[-1]
score = float(score)

print("Query:", query)
print(f"Most similar document: {documents[index]}")
print(f"Similarity score: {score}")
