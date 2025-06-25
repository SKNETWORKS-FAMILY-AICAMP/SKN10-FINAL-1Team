import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone and embeddings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sk-chatbot-index")

if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set in the environment.")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

@tool
def search_documents(query: str) -> str:
    """Searches for relevant documents in Pinecone and returns the content of the most relevant document."""
    logger.info(f"Searching for documents with query: {query}")
    try:
        docs = vectorstore.similarity_search(query, k=1) # Retrieve the most relevant document
        if docs:
            # Return the content of the first document
            content = docs[0].page_content
            logger.info(f"Found relevant document content.")
            return content
        else:
            logger.warning("No relevant documents found.")
            return "No relevant information found."
    except Exception as e:
        logger.error(f"Error searching documents in Pinecone: {e}", exc_info=True)
        return "Error searching for information."
