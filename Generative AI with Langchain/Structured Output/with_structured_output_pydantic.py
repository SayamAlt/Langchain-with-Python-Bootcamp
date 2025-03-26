from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, List
from pydantic import BaseModel, Field
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()
api_key = os.environ['OPENAI_API_KEY']

chat_model = ChatOpenAI(api_key=api_key,temperature=0)

# Create a schema
class Review(BaseModel):
    key_topics: List[str] = Field(description="Extract all key topics discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: str = Field(description="Return sentiment of the review either positive, negative, or neutral", choices=["pos", "neg", "neu"])
    pros: Optional[List[str]] = Field(default=None, description="Fetch all pros from the review in a list")
    cons: Optional[List[str]] = Field(default=None, description="Fetch all cons from the review in a list")
    name: Optional[str] = Field(default=None, description="Mention reviewer name")

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
print("Review summary:", result.summary)
print("Sentiment:", result.sentiment)
