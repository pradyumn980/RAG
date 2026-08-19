from langchain_community.document_loaders import TextLoader
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
    template='Write a sumary on the poem {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()
chain=prompt|model|parser


loader = TextLoader("file.txt")
docs = loader.load()
ans=chain.invoke({'topic':docs[0].page_content})

print(ans)