import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
import os
from dotenv import load_dotenv

load_dotenv()


## sstreamlit APP
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜"
)
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

groq_api_key = os.getenv("GROQ_API_KEY")

## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value=groq_api_key, type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Gemma Model USsing Groq API
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

prompt_template = """
I am an investor and I will be applying for ipo. in that context given below i will be giving you the 
link of the website where you need to tell whether to apply for the ipo or not.
If the company is good and has good profits and has a good future then you can tell me to apply for the ipo.
Give me an answer with yes or no. And then give the reason for the same in points
Content:{text}`

"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error(
            "Please enter a valid Url. It can may be a YT video utl or website url"
        )

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url, add_video_info=True
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

                ## Chain For Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")