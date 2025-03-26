from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os, warnings
warnings.filterwarnings('ignore')
from langchain.schema.runnable import RunnableSequence

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize the chat model
chat_model = ChatOpenAI(api_key=openai_api_key, temperature=0.7)

# Define the prompts
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke: {text}',
    input_variables=['text']
)

# Define the output parser
output_parser = StrOutputParser()

chain = RunnableSequence(prompt1,chat_model,output_parser,prompt2,chat_model,output_parser)
print(chain.invoke({'topic': 'AI'}))