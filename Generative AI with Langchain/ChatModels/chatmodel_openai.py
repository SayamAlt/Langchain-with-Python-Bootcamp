from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4o', temperature=1.5, max_completion_tokens=10)

result = model.invoke("Write a five line poem on football.")
print(result.content)