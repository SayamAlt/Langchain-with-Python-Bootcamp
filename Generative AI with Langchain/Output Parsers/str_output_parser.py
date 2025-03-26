from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate # type: ignore
# from langchain_core.output_parsers import StrOutputParser
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()

HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')
# api_key = os.environ['OPENAI_API_KEY']

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN
)

model = ChatHuggingFace(llm=llm)

# model = ChatOpenAI(api_key=api_key)

# 1st prompt -> detailed report
template1 = PromptTemplate(template='Write a detailed report on {topic}',
                           input_variables=['topic'])

# 2nd prompt -> summary
template2 = PromptTemplate(template='Write a 5 line summary on the following text. \n {text}',
                           input_variables=['text'])

prompt1 = template1.invoke({'topic': 'black hole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})
summary = model.invoke(prompt2)

print(f"Summary: {summary.content}")
# parser = StrOutputParser()

# chain = template1 | model | parser | template2 | model | parser
# result = chain.invoke({'topic': 'black hole'})
# print(result)