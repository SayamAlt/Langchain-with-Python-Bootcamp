from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.title("Research Assistant Interface")

model = ChatOpenAI()

paper_input = st.selectbox("Select Research Paper Name", ["Select...", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# Prompt Template
# template = PromptTemplate(template=f"""
#     Please summarize the research paper titled "{paper_input}" with the following specifications:
#     - Explanation Style: {style_input}
#     - Explanation Length: {length_input}
#     1. Mathematical Details:
#        - Include relevant mathematical equations if present in the paper.
#        - Explain complex mathematical concepts with simple, intuitive code snippets where applicable. 
#     2. Analogies:
#        - Use relatable analogies to simplify complex ideas.
#     If certain information is not available in the paper, respond with "Insufficient information available" instead of guessing. 
#     Ensure the summary is clear, accurate, and aligned with the provided style and length.
# """,
# input_variables=['paper input','style input','length input'],
# validate_template=True)

template = load_prompt('prompt_template.json')

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)