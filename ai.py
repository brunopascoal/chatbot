import tempfile
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
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