from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert.'),
    ('human', 'Explain in simple terms, what is {topic}')
    # SystemMessage(content='You are a helpful {domain} expert.'),
    # HumanMessage(content='Explain in simple terms, what is {topic}')
])

# chat_template = ChatPromptTemplate.from_messages([
#     ('system', 'You are a helpful {domain} expert.'),
#     ('human', 'Explain in simple terms, what is {topic}')
#     # SystemMessage(content='You are a helpful {domain} expert.'),
#     # HumanMessage(content='Explain in simple terms, what is {topic}')
# ])

prompt = chat_template.invoke({
    'domain': 'football',
    'topic': 'var refereeing'
})

print(prompt)