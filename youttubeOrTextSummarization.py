import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_core.output_parsers import StrOutputParser
import os
import os
from dotenv import load_dotenv

load_dotenv()

# streamlit run youttubeOrTextSummarization.py 

## sstreamlit APP
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜"
)
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

groq_api_key = os.getenv("GROQ_API_KEY")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value=groq_api_key, type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")


llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

prompt_template = """
I will provide you with a YouTube or website link, and I need you to summarize the content using the context provided.Summarize the video or the website link in point so that the users
can understand clearly.Please do not add extra details. Only add whatever is mentioned in the video or the website.
Context: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
parser = StrOutputParser()

if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error(
            "Please enter a valid Url. It can may be a YT video utl or website url"
        )

    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        youtube_url=generic_url, add_video_info=False
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.205  Safari/537.36"
                        },
                    )
                docs = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
