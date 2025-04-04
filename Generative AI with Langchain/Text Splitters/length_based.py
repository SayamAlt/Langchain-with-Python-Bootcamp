from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF file and split it into chunks
loader = PyPDFLoader('dl-curriculum.pdf')
documents = loader.load()

# Initialize the text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result = text_splitter.split_documents(documents)
print(result[0].page_content)
print(result[3].metadata)