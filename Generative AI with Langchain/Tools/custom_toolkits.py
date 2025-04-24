from langchain_core.tools import tool

# Custom tools
@tool
def add(x: int, y: int) -> int:
    """Adds two numbers."""
    return x + y

@tool
def subtract(x: int, y: int) -> int:
    """Subtracts two numbers."""
    return x - y

@tool
def multiply(x: int, y: int) -> int:
    """Multiplies two numbers."""
    return x * y

@tool
def divide(x: int, y: int) -> float:
    """Divides two numbers."""
    return x / y

@tool
def power(x: int, y: int) -> int:
    """Raises x to the power of y."""
    return x ** y

def calculate_factorial(n: int) -> int:
    """Calculates the factorial of n."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0 or n == 1:
        return 1
    return n * calculate_factorial(n - 1)
@tool
def factorial(x: int) -> int:
    """Calculates the factorial of x."""
    return calculate_factorial(x)

class MathToolkit:
    """A toolkit for performing various mathematical operations."""

    def __init__(self):
        self.tools = [add, subtract, multiply, divide, power, factorial]

    def get_tools(self):
        """Returns the available tools."""
        return self.tools
    
    def execute_tool(self, tool_name: str, *args):
        """Executes a tool by its name with the given arguments."""
        for tool in self.tools:
            if tool.name == tool_name:
                input_keys = tool.args
                if len(input_keys) != len(args):
                    raise ValueError(f"{tool_name} expects {len(input_keys)} arguments, got {len(args)}.")
                input_dict = dict(zip(input_keys, args))
                return tool.invoke(input_dict)
        raise ValueError(f"Tool {tool_name} not found in toolkit.")
    
toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name,"-->",tool.description)
    
print("Addition result:", toolkit.execute_tool("add", 5, 3))
print("Subtraction result:", toolkit.execute_tool("subtract", 5, 3))
print("Multiplication result:", toolkit.execute_tool("multiply", 5, 3))
print("Division result:", toolkit.execute_tool("divide", 5, 3))
print("Power result:", toolkit.execute_tool("power", 5, 3))
print("Factorial result:", toolkit.execute_tool("factorial", 5))
    

    
