from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()

HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18,description='Age of the person')
    city: str = Field(description='City of the person')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person.\n{format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

prompt = template.format(place='German')
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)
# chain = template | model | parser
# result = chain.invoke({'place': 'Paris'})
# print(result)