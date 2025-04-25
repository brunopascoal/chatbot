import streamlit as st
from ai import create_chunks, load_existing_vector_store, add_to_vector_store, ask_question

st.set_page_config(
  page_title="ChatBot",
  page_icon="🤖",
  layout="wide"
)

with st.sidebar:
  st.title("ChatBot")
  st.write("")

  vector_store = load_existing_vector_store()

  api_key = st.text_input("OpenAI API Key", type="password")
  upload_files = st.file_uploader(
    label="Put your csv files here",
    type=[".csv"],
    accept_multiple_files=True,
  )

  if upload_files:
    with st.spinner("Loading files..."):
      all_chunks = []
      for file in upload_files:
        chunks = create_chunks(file)
        all_chunks.extend(chunks)
      vector_store = add_to_vector_store(all_chunks, vector_store)

  st.markdown(
    """
      ### About
      #### This is a chatbot that you can upload csv files for use as a context for the chatbot.
    """ )

if 'messages' not in st.session_state:
  st.session_state['messages'] = []

question = st.chat_input('Type your question here')

if vector_store and question:
  for message in st.session_state.messages:
    st.chat_message(message.get('role')).write(message.get('content'))

  st.chat_message('user').write(question)
  st.session_state.messages.append({'role': 'user', 'content': question})

  response = ask_question(
    api_key=api_key,
    query=question,
    vector_store=vector_store,
  )

  st.chat_message('ai').write(response)
  st.session_state.messages.append({'role': 'ai', 'content': response})
