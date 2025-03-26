from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the LLM
llm = OpenAI(openai_api_key=api_key,temperature=0.7)

# Create a prompt template
template = "Suggest an appealing blog title for the topic {topic}"
prompt = PromptTemplate(
    template=template,
    input_variables=["topic"]
)

# Define user input
topic = input('Enter a topic: ')

# # Format the prompt with the user input
# formatted_prompt = prompt.format(topic=topic)

# Create a LLMChain using the prompt and LLM
chain = LLMChain(llm=llm,prompt=prompt)

response = chain.run(topic)
print(f"Suggested blog title: {response}")
# # Generate a response using the LLM
# response = llm.predict(formatted_prompt)

# print(f"Suggested blog title: {response}")

