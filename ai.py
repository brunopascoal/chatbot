from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import tempfile

persist_directory = "db/db"
def create_chunks(file):
  
  with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file.write(file.read())
    temp_file_path = temp_file.name

  loader = CSVLoader(file_path=temp_file_path)
  docs = loader.load()
  os.remove(temp_file_path)
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400,
  )

  chunks = text_splitter.split_documents(
    documents=docs,
  )

  return chunks

def load_existing_vector_store():
  if os.path.exists(os.path.join(persist_directory)):
    embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
    vector_store = Chroma(
      persist_directory=persist_directory,
      embedding_function=embedding_function,
    )
    return vector_store
  return None

def add_to_vector_store(chunks, vector_store):
  if vector_store:
    vector_store.add_documents(chunks)
  else:
    embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')

    vector_store = Chroma.from_documents(
      documents=chunks,
      embedding_function=embedding_function,
      persist_directory=persist_directory,
    )
    return vector_store
  
