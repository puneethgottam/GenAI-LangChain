from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_API_KEY"]= os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"]= 'true'
os.environ["LANGSMITH_ENDPOINT"]= os.getenv('LANGSMITH_ENDPOINT')

app = FastAPI(
    title = "Langchain Server",
    version = '1.0',
    description = 'A simple API Server for Langchain'
)

llm = OllamaLLM(model="llama3.2")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 50 words for a 5 year old child")

add_routes(
    app,
    prompt1|llm,
    path = "/essay",
)

add_routes(
    app,
    prompt2|llm,
    path = "/poem",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)