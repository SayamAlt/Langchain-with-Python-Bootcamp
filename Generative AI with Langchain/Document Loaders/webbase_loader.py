from langchain_community.document_loaders import WebBaseLoader
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
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url = "https://www.bestbuy.ca/en-ca/product/apple-macbook-pro-14-2-fall-2024-space-black-apple-m4-16gb-ram-512gb-ssd-english/18619900?cmp=knc-s-20413581516&&x&utm_source=google&utm_id=240828all495&utm_medium=troas&utm_creativeformat=pmax&gad_source=4&gclid=Cj0KCQjw16O_BhDNARIsAC3i2GA9W5rQiECI5hVTWwPSMewnsZ7pBzBLKvgR-plHug_HmNt-ks0bEMYaAsKHEALw_wcB&gclsrc=aw.ds"

loader = WebBaseLoader(url) # Can pass in a list of URLs too under 'urls' parameter

documents = loader.load()

# print(len(documents))
# print(documents[0].page_content)

chain = prompt | model | parser

result = chain.invoke({'question': 'What is the product that we are talking about?', 'text': documents[0].page_content})
print(result)

# Product Idea: Chrome Plugin to load web page content and ask questions about it...