from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

persist_directory = "db/db"

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def ask_question(query, vector_store):
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    retriever = vector_store.as_retriever()

    system_prompt = """
    You are an AI assistant for answering questions about the content of the PDF files.
    If the question is not related to the content of the PDF files, please answer that you do not know.
    If the question is related to the content of the PDF files, please answer the question.
    If the question is a greeting, please respond with a friendly greeting. 
    Context: {context}
    """

    messages = [("system", system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get("role"), message.get("content")))
    messages.append(("human", "{input}"))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=question_answer_chain
    )

    response = chain.invoke({"input": query})

    return response.get("answer")


def create_chunks(file):

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(
        file_path=temp_file_path,
    )
    docs = loader.load()

    os.remove(temp_file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=60,
    )
    chunks = text_splitter.split_documents(documents=docs)

    chunks = text_splitter.split_documents(
        documents=docs,
    )

    return chunks


def load_existing_vector_store():
    if os.path.isdir(persist_directory) and os.listdir(persist_directory):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        return vector_store
    return None


def add_to_vector_store(chunks, vector_store):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    if vector_store:
        vector_store.add_documents(chunks)
        return vector_store
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        return vector_store
