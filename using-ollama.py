from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
import streamlit
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

#Prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked. Give the response in a json"),
        ("user","Question:{question}")
    ]
)

#streamlit framwork
streamlit.title("Langchain Demo with LLama3.2")
input_text = streamlit.text_input("What question you have in mind ?")

## Ollama LLama Model

llm = OllamaLLM(model="llama3.2")
output_parser = StrOutputParser()
json_parser = JsonOutputParser()
chain = prompt|llm|json_parser # First prompt is set and then it is passed to llm to get the output and then after it gives the answer and it is passed to json parser to send it as an json output
if input_text: 
    streamlit.write(chain.invoke({"question":input_text}))