from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# documents = loader.load()
documents = loader.lazy_load()
# print(len(documents))

# print(documents[325].page_content)
# print(documents[325].metadata)

for doc in documents:
    print(doc.metadata)