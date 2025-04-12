from langchain_community.retrievers import WikipediaRetriever

# Initialize the retriever (option: set language and top_k)
retriever = WikipediaRetriever(top_k_resuults=2, lang='en')

# Define the query
query = "The geopolitical history of India and Pakistan from the perspective of a Chinese."

# Get relevant Wikipedia documents
documents = retriever.invoke(query)

# Print retrieved content
for i, doc in enumerate(documents):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}") # Truncate the final results for display
    print(f"Metadata:\n{doc.metadata}")