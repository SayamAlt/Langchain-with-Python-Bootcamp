from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()
api_key = os.environ['OPENAI_API_KEY']

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI(api_key=api_key,temperature=0)

parser = StrOutputParser()

chain = prompt | model | parser # Langchain Expression Language - | -> Pipe operator to create pipelines

result = chain.invoke({'topic': 'Paris'})
print(result)

chain.get_graph().print_ascii()
