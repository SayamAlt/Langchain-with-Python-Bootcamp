from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import asyncio

class MultiplyInput(BaseModel):
    a: int = Field(..., description="First number to multiply")
    b: int = Field(..., description="Second number to multiply")
    
# On inheriting from BaseTool class, we can create a custom tool
class MultiplyTool(BaseTool):
    """Multiply two numbers."""

    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput
    
    def _run(self, a: int, b: int) -> int:
        return a * b
    
    async def _arun(self, a: int, b: int) -> int: # async method can be used to ensure concurrency
        return a * b
    
# multiply_tool = MultiplyTool()
# result = multiply_tool.invoke({'a': 2, 'b': 3})
# print(result)

# print(multiply_tool.name)
# print(multiply_tool.description)
# print(multiply_tool.args)

async def _():
    multiply_tool = MultiplyTool()
    result = await multiply_tool.ainvoke({'a': 6, 'b': 7})
    print("Async Result:", result)
    print("Tool name:", multiply_tool.name)
    print("Description:", multiply_tool.description)
    print("Args Schema:", multiply_tool.args_schema.schema())

asyncio.run(_())