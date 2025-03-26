from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, List
from pydantic import BaseModel, Field
import os, warnings
warnings.filterwarnings('ignore')

load_dotenv()
api_key = os.environ['OPENAI_API_KEY']

chat_model = ChatOpenAI(api_key=api_key,temperature=0)

# Create a JSON schema
review_schema = {
    "title": "Review",
    "description": "Review for a product",
    "type": "object",
    "properties": {
        "key_topics": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Extract all key topics discussed in the review in a list"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos","neg","neu"], # Instead of literal
            "description": "Return sentiment of the review either positive, negative, or neutral"
        },
        "pros": {
            "type": ["array","null"],
            "items": {
                "type": "string"
            },
            "description": "Fetch all pros from the review in a list"
        },
        "cons": {
            "type": ["array","null"],
            "items": {
                "type": "string"
            },
            "description": "Fetch all cons from the review in a list"
        },
        "name": {
            "type": ["string","null"],
            "description": "Write the name of the reviewer"
        }
    },
    "required": ["key_topics", "summary", "sentiment"] 
}

structured_model = chat_model.with_structured_output(review_schema)

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

