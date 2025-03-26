from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()
api_key = os.environ['OPENAI_API_KEY']

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary for the following text:\n{text}',
    input_variables=['text']
)

model = ChatOpenAI(api_key=api_key, temperature=0)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Economic Recession'})
print(result)

chain.get_graph().print_ascii()