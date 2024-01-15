from Gemini_RecursiveChunking_TikTokenV3 import Summarizer
import streamlit as st


import streamlit as st


@st.cache_resource
def get_summarizer():
    return Summarizer()


@st.cache_data
def summarize(txt):
    return client.summarize(txt)


if __name__ == "__main__":
    client = get_summarizer()
    st.title("Long Document Adaptive Summarizer")

    input = st.text_area(
        "Enter text of any length here, I shall summarize it for you :)"
    ).strip()
    if st.button("Summarize"):
        with st.spinner("Summarizing ..."):
            st.markdown("Summary:  " + summarize(input))
