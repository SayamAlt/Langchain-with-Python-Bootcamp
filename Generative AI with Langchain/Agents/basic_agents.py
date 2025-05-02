from langchain.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import os, warnings, requests
warnings.filterwarnings('ignore')

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(api_key=openai_api_key)

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke("top news in football today")
# print(results)

# Pull the ReAct prompt from the LangChain Hub
prompt = hub.pull("hwchase17/react") # pulls the standard react agent prompt

# ReAct = Reasoning + Acting It's a design pattern
# Create a ReAct agent with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

# Wrap the agent with an executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)

query = "Give a detailed overview of the results of today's IPL match between MI and RR"
result = agent_executor.invoke({"input": query})
print(result['output'])
