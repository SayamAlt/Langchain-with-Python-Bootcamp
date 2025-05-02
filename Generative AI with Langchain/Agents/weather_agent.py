import os, warnings, requests
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
weatherstack_api_key = os.getenv('WEATHERSTACK_API_KEY')

llm = ChatOpenAI(api_key=openai_api_key)
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
        This function fetches the current weather data for a given city.
    """
    response = requests.get(f"https://api.weatherstack.com/current?access_key={weatherstack_api_key}&query={city}")
    return response.json()

# Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create a ReAct agent with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Wrap it with AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)

query = "Find the capital of Scotland and find its current weather condition."
response = agent_executor.invoke({"input": query})
print(response['output'])