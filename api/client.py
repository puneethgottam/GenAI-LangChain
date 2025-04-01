# To interact with api
import requests
import streamlit as st

def get_essay_response(input_text):
    # API endpoint URL
    url = "http://localhost:8000/essay/invoke"
    reponse = requests.post(url, json={"input":  {"topic" : input_text}})
    
    return reponse.json()['output']

def get_poem_response(input_text):
    # API endpoint URL
    url = "http://localhost:8000/poem/invoke"
    reponse = requests.post(url, json={"input":  {"topic" : input_text}})
    
    return reponse.json()['output']

st.title('Langchian AI Writer with LLAMA3.2')

input_text1 = st.text_input('Enter a topic to write an essay')
input_text2 = st.text_input('Enter a topic to write a poem')

if input_text1:
    st.write(get_essay_response(input_text1))

if input_text2:
    st.write(get_poem_response(input_text2))