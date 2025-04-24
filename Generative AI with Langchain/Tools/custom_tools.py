from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Custom Tools
# Step 1 - Create a function
def multiply(a, b):
    """Multiply two numbers."""
    return a * b

# Step 2 - Add type hints
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Step 3 - add tool decorator
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Step 4 - invoke the tool
result = multiply.invoke({'a': 5, 'b': 4})
print(result)

# Tool attributes
print(multiply.name)
print(multiply.description)
print(multiply.args)

print(multiply.args_schema.model_json_schema()) # Passed to the LLM - tools are not sent, its schema is sent
