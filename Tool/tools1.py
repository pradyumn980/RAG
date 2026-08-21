from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke("What is the capital of France?")
print(result)