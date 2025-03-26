from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os, warnings
warnings.filterwarnings('ignore')
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize the chat model
chat_model = ChatOpenAI(api_key=openai_api_key)

# Define the prompt
prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

def word_count(text):
    return len(text.split())

# Define the output parser
output_parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt,chat_model,output_parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic': 'AI'})

final_result = """{} \n Word Count: {}""".format(result['joke'], result['word_count'])
print(final_result)














# from langchain.schema.runnable import RunnableLambda

# def word_counter(text):
#     return len(text.split())

# runnable_word_counter = RunnableLambda(word_counter)
# print(runnable_word_counter.invoke('Hi there, how are you?'))