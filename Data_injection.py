import os
import logging
import wikipedia
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_DATABASE_NAME = "Tourism_Cities"
DEFAULT_EMBEDDER = "qwen3-embedding"

# --- logger setup ---
script_name = os.path.splitext(os.path.basename(__file__))[0]

script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "..", "logs")
os.makedirs(log_dir, exist_ok=True)  # create the folder if it doesn't exist yet

log_path = os.path.join(log_dir, f"{script_name}.log")

logger = logging.getLogger(script_name)
logger.setLevel(logging.INFO)

if not logger.handlers:  # prevents duplicate log lines if main() is called multiple times
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def main(
  query: str, embedding_model: str = DEFAULT_EMBEDDER, 
  database_name: str = DEFAULT_DATABASE_NAME):

  wikipedia.set_user_agent("MyRAGProjectApp/1.0 (contact: omkar.kolekar@viit.ac.in)")

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

  logger.info(f"Initializing embedder ({embedding_model} via Ollama)")
  embedder = (
      OllamaEmbeddings(model=embedding_model)
  )

  logger.info(f"Initializing FAISS database, and Embedding {len(splitted_documents)} chunks...")

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


if ("__main__" == __name__):

  query = "Paris"

  main(query)
  
