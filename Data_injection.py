import logging
import wikipedia
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- logger setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),                # prints to console
        logging.FileHandler("Data_injection_logs.log")  # <-- Writes to a file
    ]
)
logger = logging.getLogger("Data_injection_logs")

wikipedia.set_user_agent("MyRAGProjectApp/1.0 (contact: omkar.kolekar@viit.ac.in)")

query = "Paris"
logger.info(f"Loading Wikipedia page for query: '{query}'")
page_loader = WikipediaLoader(query=query, lang='en', load_max_docs=1)

document = page_loader.load()

if not document:
    logger.error(f"No Wikipedia page found for query: '{query}'")
    raise ValueError("No document loaded")

logger.info(f"Loaded {len(document)} document(s). Title: {document[0].metadata.get('title')}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
splitted_documents = text_splitter.split_documents(document)

logger.info(f"Split into {len(splitted_documents)} chunks")

logger.info("Initializing embedder (qwen3-embedding via Ollama)")
embedder = (
    OllamaEmbeddings(model="qwen3-embedding")
)

logger.info(f"Initializing FAISS database, and Embedding {len(splitted_documents)} chunks...")
database_name = "Tourism_Cities"
try:
  cities_db = FAISS.from_documents(
    splitted_documents,
    embedder
  )
except Exception as e:
    logger.exception(f"Embedding failed: {e}")
    raise

logger.info(f"Saving FAISS database, under the name {database_name}")
try:
  cities_db.save_local(database_name)
except Exception as e:
  logger.exception(f"Saving database failed: {e}")
  raise

logger.info("*** All data injection operations completed! ***")
