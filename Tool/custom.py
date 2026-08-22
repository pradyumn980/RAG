from langchain_core.tools import tool


def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

@tool
def multiply_tool(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return multiply(a, b)

result = multiply_tool.invoke({"a":3, "b":4})

print(multiply_tool.name)
print(multiply_tool.description)
print(result)