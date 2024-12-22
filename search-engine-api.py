from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
from langserve import add_routes
from langchain_core.runnables import Runnable
import os

# Load environment variables
load_dotenv()

# FastAPI instance
app = FastAPI(title="LangChain Chatbot API", version="1.0.0")

# Pydantic model for incoming messages
class ChatRequest(BaseModel):
    api_key: str
    messages: list[dict]  # Example: [{"role": "user", "content": "What is AI?"}]

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Function to initialize the LangChain agent as a Runnable
def get_chain(api_key: str) -> Runnable:
    """
    Initializes and returns the LangChain agent as a Runnable.
    """
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )
    return agent


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    API endpoint to process chat messages.
    """
    api_key = request.api_key
    messages = request.messages

    if not api_key:
        raise HTTPException(status_code=400, detail="Missing Groq API Key.")

    # Initialize the chain
    agent = get_chain(api_key)

    # Run the agent
    try:
        response = agent.run(messages)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
