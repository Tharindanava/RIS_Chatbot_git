from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import requests
from bs4 import BeautifulSoup

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

def create_vectorstore():
    urls = [
        "https://www.medscape.com/radiology",
        "https://pubmed.ncbi.nlm.nih.gov/",
        "https://radiopaedia.org/",
        "https://www.myesr.org/",
        "https://www.bmj.com/specialties/radiology"
    ]
    
    # Scrape documents
    docs = [scrape_content(url) for url in urls if scrape_content(url) is not None]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    docs_list = text_splitter.split_documents(docs)

    # Filter metadata
    filtered_docs = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in docs_list]

    # Create vector database
    vectorstore = Chroma.from_documents(documents=filtered_docs, 
                                        collection_name="rag-chroma", 
                                        embedding=GPT4AllEmbeddings())
    retriever = vectorstore.as_retriever()

    return retriever
