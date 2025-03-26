from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os, warnings
warnings.filterwarnings('ignore')
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize the chat model
chat_model = ChatOpenAI(api_key=openai_api_key)

# Define the prompt
prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text: \n{text}',
    input_variables=['text']
)

# Define the output parser
output_parser = StrOutputParser()

report_gen_chain = prompt1 | chat_model | output_parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | chat_model |output_parser),
    RunnablePassthrough()
)

final_chain = report_gen_chain | branch_chain

print(final_chain.invoke({'topic': 'Russia vs. Ukraine'}))