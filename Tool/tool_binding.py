import os
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

print("Token loaded:", bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")))
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

tools = [multiply, add]

#tool binding

model_with_tools = model.bind_tools(tools)

#using the model with tools
response = model_with_tools.invoke(
    "What is 15 multiplied by 20?"
)

print(response.content)