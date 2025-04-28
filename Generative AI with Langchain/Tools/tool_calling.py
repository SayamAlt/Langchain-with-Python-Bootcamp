from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import requests, os, warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
# Create a tool
@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers a and b, return their product."""
    return a * b

# print(multiply.invoke({'a': 6, 'b': 3}))

# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

# Tool binding
llm = ChatOpenAI(api_key=api_key, temperature=0.5)
llm_with_tools = llm.bind_tools([multiply])

# print(llm_with_tools.invoke("Hi, how are you?"))

query = HumanMessage(content='Can you multiply 3 with 1000?')

messages = [query]

# Tool calling
result = llm_with_tools.invoke(messages)

messages.append(result)
# print(messages)
# print(result.tool_calls[0]['args'])

# print(multiply.invoke(result.tool_calls[0]['args']))

# Tool execution
tool_result = multiply.invoke(result.tool_calls[0])
messages.append(tool_result) # Tool message

# print(messages)

print(llm_with_tools.invoke(messages).content)


