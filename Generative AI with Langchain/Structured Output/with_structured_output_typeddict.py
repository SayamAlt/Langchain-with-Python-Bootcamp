from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Optional, Literal
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()
api_key = os.environ['OPENAI_API_KEY']

chat_model = ChatOpenAI(api_key=api_key,temperature=0)
# TypedDict doesn't support data validation
# Create a schema for data format
class Review(TypedDict): 
    key_topics: Annotated[List[str], "Extract all key topics discussed in the review in a list"]
    pros: Annotated[Optional[List[str]], "Fetch all pros from the review in a list"]
    cons: Annotated[Optional[List[str]], "Fetch all cons from the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos","neg","neu"], "Return sentiment of the review either positive, negative, or neutral"]
    name: Annotated[Optional[str], "Mention reviewer name"]

structured_model = chat_model.with_structured_output(Review)

review = """
I recently purchased the EcoPro Blender, and I’m incredibly impressed with its performance. 
The motor is powerful, blending smoothies in seconds, even with frozen fruit. 
It’s also surprisingly quiet compared to other blenders I’ve used. 
The design is sleek, and cleaning is a breeze with its removable parts. 
The only downside is that it’s a bit bulky, so storage can be tricky in smaller kitchens. 
Overall, I’m really happy with my purchase and would definitely recommend it to anyone looking for a reliable, efficient blender!
Review by Sayam Kumar
"""

result = structured_model.invoke(review)
print(result)
print(result['summary'])
print(result['sentiment'])
