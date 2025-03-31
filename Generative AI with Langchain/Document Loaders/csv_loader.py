from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('Social_Network_Ads.csv')

documents = loader.load()

# print(len(documents))
print(documents[12].page_content)