from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from typing import Literal
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()
api_key = os.environ['OPENAI_API_KEY']

model = ChatOpenAI(api_key=api_key)
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative:\n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback:\n{feedback}',
    input_variables=['feedback']
)
chain2 = prompt2 | model | parser
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback:\n{feedback}',
    input_variables=['feedback']
)
chain3 = prompt3 | model | parser

branch_chain = RunnableBranch( # Essentially like if-else statements
    (lambda x: x.sentiment == 'positive', chain2),
    (lambda x: x.sentiment == 'negative', chain3),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a beautiful smartphone.'}))

chain.get_graph().print_ascii()

