from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage
import os, warnings, requests, json
warnings.filterwarnings("ignore")
from typing import Annotated

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
exchange_rate_api_key = os.getenv('EXCHANGE_RATE_API_KEY')

# Create tools
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a base currency and a target currency.
    """
    url = f"https://v6.exchangerate-api.com/v6/{exchange_rate_api_key}/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    return response.json()

# print(get_conversion_factor.invoke({'base_currency': 'USD', 'target_currency': 'EUR'}))

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    Given a currency conversion rate, this function calculates the target currency value from the base currency value.
    """
    return base_currency_value * conversion_rate

# print(convert.invoke({'base_currency_value': 10, 'conversion_rate': 0.8803}))

# Tool binding
llm = ChatOpenAI(api_key=api_key, temperature=0.5)
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage(content="What is the conversion factor between CAD and SGD? And then, based on that, can you convert 10 CAD to SGD?")]
# print(messages)

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)
# print(ai_message.tool_calls)

for tool_call in ai_message.tool_calls:
    # print(tool_call)
    # execute the first tool and get the value of conversion rate
    # execute the second tool using the conversion rate from the first tool
    if tool_call['name'] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        # print(tool_message1)
        # fetch this conversion rate
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']
        # print(conversion_rate)
        # append this tool message to error rate
        messages.append(tool_message1)
    elif tool_call['name'] == 'convert':
        # fetch the current arguments
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)
        
# print(messages)
print(llm_with_tools.invoke(messages).content)