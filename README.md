# What is this?

This is a simple chatbot application built using Streamlit and LangChain, creating a RAG (Retrieval-Augmented Generation) chatbot. The chatbot can process multiple PDF files, extract text from them, and generate responses based on the content of the uploaded documents.

## Features

- Upload and process multiple PDF files
- Extract and chunk text from PDFs
- Store document embeddings in a vector database
- Ask questions about the uploaded documents
- Get AI-powered responses based on document content
- Markdown-formatted responses

## Technology Stack

- Python 3.13
- Streamlit for the web interface
- LangChain for document processing and AI chain operations
- OpenAI for embeddings and language model
- ChromaDB for vector storage
- PyPDF for PDF processing

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd chatbot
```

2. Setup a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY=<your-api-key>
```

Or add it to a .env file.

## Usage

1. Run the Streamlit app:

```bash
streamlit run main.py
```

2. Open the provided URL in your web browser.
3. Upload one or more PDF files.
4. Ask questions about the uploaded documents.
5. The chatbot will process the PDFs, generate embeddings, and provide responses.

## Customization

You can customize the chatbot's behavior by modifying the main.py file. Here are some potential areas to explore:

- Adjust the chunk size for text extraction.
- Change the vector database used (e.g., Pinecone).
- Experiment with different language models.
- Implement additional features like document summarization.
