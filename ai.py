from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
import streamlit as st
import os
import tempfile

persist_directory = "db/db"

def ask_question(api_key, query, vector_store):

  llm = OpenAI(
      api_key=api_key
  )
  retriever = vector_store.as_retriever()

  system_prompt = '''
  Você é um assistente financeiro inteligente. Use o contexto abaixo para responder perguntas relacionadas às despesas registradas.

  Responda em formato **markdown**, utilizando visualizações **elaboradas e interativas** sempre que possível (como tabelas ou gráficos).

  **Instruções Importantes:**
  - Desconsidere linhas onde a coluna `Tipo` contenha o texto **"Total mês"**.
  - Considere apenas valores válidos e numéricos da coluna `Valor`.
  - Caso a coluna `Data` esteja em branco, utilize a coluna `mês` para entender o período da despesa.
  - Linhas com parcelas devem ser tratadas como **somente a parcela atual**, e **não o valor total da compra**.
  - Despesas com categoria **"Fixo"** devem ser consideradas recorrentes.

  Você pode ser solicitado a:
  - Calcular totais por mês, por categoria ou por banco.
  - Listar parcelas ativas no mês.
  - Gerar resumos interativos das despesas.
  - Fazer comparações entre meses (considere os nomes dos meses normalizados, como "Março", "março", etc).

  Contexto: {context}
  '''

  messages = [('system', system_prompt)]
  for message in st.session_state.messages:
    messages.append((message.get('role'), message.get('content')))
  messages.append(('human', '{input}'))

  prompt = ChatPromptTemplate.from_messages(messages)

  question_answer_chain = create_stuff_documents_chain(
      llm=llm,
      prompt=prompt,
  )

  chain = create_retrieval_chain(
      retriever=retriever,
      combine_docs_chain=question_answer_chain
  )

  response = chain.invoke({'input': query})

  return response.get('answer')
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
  
