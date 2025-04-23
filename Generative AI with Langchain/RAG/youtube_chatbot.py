import os, warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get('OPENAI_API_KEY')

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

video_id = "zuGFdImjwEs"

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    # Flatten transcript into a single string
    transcript = " ".join([d["text"] for d in transcript_list])
    # print(transcript)
except TranscriptsDisabled:
    print("No captions available for this video.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

embedding_model = OpenAIEmbeddings(api_key=api_key)
vector_store = FAISS.from_documents(chunks, embedding_model)

# print(vector_store.index_to_docstore_id)
# print(vector_store.get_by_ids(['9e229ec7-c05b-4c70-9693-a3cc51bae8e9']))

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# print(retriever.invoke('What is a knowledge graph?'))

llm = ChatOpenAI(api_key=api_key, temperature=0.5)

prompt = PromptTemplate(
    template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context: {context}
        Question: {question}
    """,
    input_variables=['context', 'question']
)

question = "Which metrics were used to evaluate the regression models?"
retrieved_docs = retriever.invoke(question)

context = "\n".join([doc.page_content for doc in retrieved_docs])

formatted_prompt = prompt.format(context=context, question=question)

response = llm.predict(formatted_prompt)

# print(response)

def format_documents(retrieved_docs):
    return "\n".join([doc.page_content for doc in retrieved_docs])

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_documents),
    'question': RunnablePassthrough()
})

# print(parallel_chain.invoke("Why is regression useful?"))

parser = StrOutputParser()

chain = parallel_chain | prompt | llm | parser

result = chain.invoke("Summarize the video")
print(result)