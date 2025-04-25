import streamlit as st

st.set_page_config(
  page_title="ChatBot",
  page_icon="🤖",
  layout="wide"
)

with st.sidebar:
  st.title("ChatBot")
  st.write("")


  st.markdown(
    """
      ### About
      #### This is a chatbot that you can upload csv files for use as a context for the chatbot.
    """ )
  
question = st.chat_input('Type your question here')
