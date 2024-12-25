from crewai import Agent, LLM
from tools import yt_tool
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")
my_llm = LLM( # there is some issue here
    model="groq/llama3-8b-8192",
    temperature=0.3,
    max_tokens=4096,
)

blog_researcher = Agent(
    role="Blog Researcher from youtube videos",
    goal=" get the relevant video content for the topic {topic} from youtube channel",
    verbose=True,
    memory=True,
    backstory="Expert in understanding videos in AI Data Sciene and Machine Learning",
    tools=[yt_tool],
    allow_delegation=True,
    llm=my_llm,
)


# Create a senior blog writter agent with YT tool

blog_writer = Agent(
    role="Blog Writer",
    goal="Narrate compelling tech stories about the video {topic} from youtube channel",
    verbose=True,
    memory=True,
    backstory="With a flair for simplifying complex topics, you crag engagin narractives that captivate and educate, bringing new discoverties to light in an accessible mananer.",
    tools=[yt_tool],
    allow_delegation=False,
    llm=my_llm,
)
