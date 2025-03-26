from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os, warnings
warnings.filterwarnings('ignore')
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a LinkedIn post about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI(api_key=openai_api_key)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin_post': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic': 'AI'})
print("Tweet:", result['tweet'])
print("\nLinkedIn Post:", result['linkedin_post'])