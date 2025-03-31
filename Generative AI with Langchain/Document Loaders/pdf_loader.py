from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

documents = loader.load()

# print(documents)
# print(len(documents))

print(documents[16].metadata)
print(documents[0].page_content)
