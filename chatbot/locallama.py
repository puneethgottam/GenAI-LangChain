from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain 
# from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"]= os.getenv('HUGGINGFACE_API_KEY')
os.environ["LANGSMITH_API_KEY"]= os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"]= 'true'
os.environ["LANGSMITH_ENDPOINT"]= os.getenv('LANGSMITH_ENDPOINT')

## PromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an assistant. Please respond to user queries"),
        ("user","Question:{question}")
    ]
)

##streamlit

st.title('Langchain chatbot with LLAMA 3.2 API')
input_text = st.text_input("Ask anything you want.")

## Ollama : llama3.2 
## Install and download model first
llm = OllamaLLM(model = "llama3.2")
output = StrOutputParser()
chain = prompt|llm|output

if input_text:
    st.write(chain.invoke({"question":input_text}))