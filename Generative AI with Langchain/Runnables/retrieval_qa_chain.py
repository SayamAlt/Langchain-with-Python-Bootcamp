from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the LLM
llm = OpenAI(openai_api_key=api_key, temperature=0.7)

# Initialize an OpenAI embedding model
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

pdf_loader = PyPDFLoader("adf.pdf")
documents = pdf_loader.load()

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Embed the documents
vectorstore = FAISS.from_documents(documents, embeddings)

# Create a retriever to perform semantic search
retriever = vectorstore.as_retriever()

query = "What are the key takeways from this document?"
# retrieved_docs = retriever.get_relevant_documents(query)

# Combine the retrieved documents into a single string
# combined_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Create a prompt template
template = "Based on the following text, answer the question: {query}\n{combined_text}\n"

prompt = PromptTemplate(
    template=template,
    input_variables=["query", "combined_text"]
)

qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=retriever)

# Execute the Retrieval chain to get a response to the query
response = qa_chain.run(query)

print(f"Answer: {response}")

# formatted_prompt = prompt.format(query=query,combined_text=combined_text)

# Generate a response using the LLM
# response = llm.predict(formatted_prompt)

# print(f"Answer: {response}")