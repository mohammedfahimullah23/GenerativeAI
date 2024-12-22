import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful massistant . Please  repsonse to the user queries",
        ),
        ("user", "Question:{question}"),
    ]
)


def generate_response(question, llm):
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer


## #Title of the app
st.title("Enhanced Q&A Chatbot With OpenAI")


## Select the OpenAI model
llm = st.sidebar.selectbox("Select Open Source model", ["llama3.2:latest"])

## MAin interface for user input
st.write("Goe ahead and ask any question")
user_input = st.text_input("You:")


if user_input:
    response = generate_response(user_input, llm)
    st.write(response)
else:
    st.write("Please provide the user input")
