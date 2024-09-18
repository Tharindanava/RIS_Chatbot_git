import os
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from bs4 import BeautifulSoup
import requests

# Constants
DATA_PATH = "data/"
DB_PATH = "vectorstores/db_chroma"
BATCH_SIZE = 5000  # Adjust this as per your system's capacity

# Function to scrape the content of a given URL
def scrape_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text content from the page
        text = soup.get_text()
        return Document(page_content=text, metadata={"url": url})
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return None

# Split a list into chunks of a given size
def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


# Single method to load documents from local files and URLs, and add them to vector DB
def create_vectorstore(urls=None):
    # Load documents from the local directory
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    local_documents = loader.load()

    # Scrape documents from URLs
    url_documents = [
        "https://www.medscape.com/radiology",
        "https://pubmed.ncbi.nlm.nih.gov/",
        "https://radiopaedia.org/",
        "https://www.myesr.org/",
        "https://www.bmj.com/specialties/radiology"
    ]

    if urls:
        url_documents = [scrape_content(url) for url in urls if scrape_content(url) is not None]

    # Combine local and URL documents
    documents = local_documents + url_documents

    # Ensure all elements are Document objects
    documents = [doc if isinstance(doc, Document) else Document(page_content=doc) for doc in documents]

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
    texts = text_splitter.split_documents(documents)

    # Initialize GPT4AllEmbeddings with CUDA
    embeddings = GPT4AllEmbeddings(model_kwargs={'device': 'cuda'})  # Enable CUDA for embeddings

    # Check if Chroma DB already exists
    if os.path.exists(DB_PATH):
        print("Loading existing vector database...")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)  # Use embeddings directly
    else:
        print("Creating new vector database...")
        # Create a new vector database if none exists

        # Fix: Pass `embeddings` as a separate argument
        vectorstore = Chroma.from_documents(documents=texts, collection_name="rag-chroma", persist_directory=DB_PATH, embedding_function=embeddings)

    # Add documents in batches to avoid exceeding the maximum batch size
    for batch in chunked(texts, BATCH_SIZE):
        vectorstore.add_documents(batch)

    # Persist changes (if necessary)
    if not os.path.exists(DB_PATH):
        vectorstore.persist()

    retriever = vectorstore.as_retriever()

    return retriever