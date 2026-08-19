from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7,
)

model=ChatHuggingFace(llm=llm)
prompt=PromptTemplate(
    template='What is {topic} from the following text: {text} in 2 lines',  
    input_variables=['topic', 'text']
)
parser=StrOutputParser()

loader=WebBaseLoader("https://www.geeksforgeeks.org/python-programming-language/")
docs=loader.load()
chain=prompt|model|parser

result=chain.invoke({'topic':'pandas','text':docs[0].page_content})

print(result)