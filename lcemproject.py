from fastapi import FastAPI
#FastAPI automatically generates interactive API documentation using Swagger UI and ReDoc, based on your code
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)
system_template="Translate the following into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system",system_template),
     ("user","{text}")]
)

parser = StrOutputParser()

#create chain
chain = prompt_template | model | parser

# App definition
app = FastAPI(title="langchain server",version="1.0",
              description="A simple API server using runnable interfaces")

# Adding chain routes
add_routes(
    app,
    chain,
    path='/chain'
)

if __name__== "__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)

# To execute this run python filename
# swagger page go to localhost:8000/docs
# You can execute it like an API call.
# Whatver we have created as project those API will come

