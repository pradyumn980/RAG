import os
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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

query=HumanMessage(content="can you multiply 3 and 4, and then add 5 to the result?")

messages=[query]
#using the model with tools
response = model_with_tools.invoke(messages)


messages.append(response)


#tool execution
response.tool_calls[0]
multiply_result = multiply.invoke({'name': 'multiply', 'args': {'a': 3, 'b': 4}, 'id': 'call_Y63eVkLU4FMy3DfQ1UciBpT3', 'type': 'tool_call'})
#print(ans)

messages.append(multiply_result)
print(messages)

#print(f"Multiply Result: {multiply_result}")

#response back to LLM
