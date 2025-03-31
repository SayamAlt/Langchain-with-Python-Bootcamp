from langchain_community.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(api_key=api_key)

prompt = PromptTemplate(
    template='Write a summary for the following poem:\n{poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf-8')

documents = loader.load()

# print(documents)
# print(type(documents))
# print(len(documents))

# print(documents[0].metadata)
# print(documents[0].page_content)
# print(type(documents[0]))

chain = prompt | model | parser

result = chain.invoke({'poem': documents[0].page_content})
print(result)