from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# Custom Tools - Using StructuredTool
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="First number to multiply")
    b: int = Field(required=True, description="Second number to multiply")
    
def multiply_two_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_two_numbers,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a': 6, 'b': 9})
print(result)

print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)