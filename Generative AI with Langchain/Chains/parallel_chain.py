from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']
google_api_key = os.environ['GOOGLE_API_KEY']

model1 = ChatOpenAI(api_key=openai_api_key,temperature=0)
model2 = ChatGoogleGenerativeAI(api_key=google_api_key,model='gemini-1.5-pro')

prompt1 = PromptTemplate(
    template='Generate short and concise notes for the following text:\n{text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='Generate 5 short question answers based on the following text:\n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document:\n{notes}\n{quiz}',
    input_variables=['notes', 'quiz']
)

chain1 = prompt1 | model1 | parser
chain2 = prompt2 | model2 | parser

parallel_chain = RunnableParallel({
    'notes': chain1,
    'quiz': chain2
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
    Support Vector Machines (SVM) are powerful supervised learning algorithms used for classification and regression tasks. 
    They work by finding an optimal hyperplane that best separates data points in a high-dimensional space. 
    The goal of SVM is to maximize the margin between different classes while minimizing misclassification errors. 
    For linearly separable data, SVM identifies the hyperplane that maximizes the distance between the closest points (support vectors) of each class. 
    However, real-world datasets are often not linearly separable, which is where SVM uses kernel functions to project data into a higher-dimensional space where it becomes separable.
    SVM supports several kernel functions, such as linear, polynomial, radial basis function (RBF), and sigmoid kernels, allowing it to adapt to different types of data distributions. 
    The RBF kernel is particularly effective in handling complex relationships by mapping data into an infinite-dimensional space. 
    The algorithm relies on regularization (C parameter) to control the trade-off between achieving a low error and maintaining a large margin. 
    A higher C focuses more on correct classification, potentially leading to overfitting, whereas a lower C prioritizes a wider margin and better generalization.
    For regression tasks, Support Vector Regression (SVR) follows a similar principle but aims to fit data within an ε-tube, where predictions are penalized only if they fall outside this margin. 
    Despite its strength in handling high-dimensional data and avoiding overfitting, SVM can be computationally expensive, especially with large datasets, as training complexity grows significantly with sample size. 
    Nonetheless, it remains a widely used algorithm in fields like image recognition, bioinformatics, and finance due to its robustness and versatility.
"""
result = chain.invoke({'text': text})
print(result)

chain.get_graph().print_ascii()