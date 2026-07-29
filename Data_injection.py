import os
import time
import logging
import requests
import wikipedia
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_DATABASE_NAME = "Tourism_Cities"
DEFAULT_EMBEDDER = "qwen3-embedding"
OLLAMA_HOST = "http://localhost:11434"

# --- logger setup ---
script_name = os.path.splitext(os.path.basename(__file__))[0]
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"{script_name}.log")

logger = logging.getLogger(script_name)
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# --- 1: pre-flight check ---
def check_ollama_available(host: str = OLLAMA_HOST) -> None:
    """Fail fast if Ollama isn't reachable — no point doing Wikipedia/chunking work first."""
    try:
        requests.get(host, timeout=2)
    except requests.exceptions.ConnectionError:
        logger.error(f"Ollama is not reachable at {host}. Is the server running?")
        raise ConnectionError(f"Ollama is not reachable at {host}")


# --- 2: retry helper — used ONLY for the Wikipedia fetch, not the embedding call ---
def retry_with_backoff(func, max_attempts: int = 3, initial_delay: int = 2):
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts:
                logger.error(f"All {max_attempts} attempts failed: {e}")
                raise
            logger.warning(f"Attempt {attempt}/{max_attempts} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # exponential backoff


def main(
  query: str,
  embedding_model: str = DEFAULT_EMBEDDER,
  database_name: str = DEFAULT_DATABASE_NAME):

  wikipedia.set_user_agent("MyRAGProjectApp/1.0 (contact: omkar.kolekar@viit.ac.in)")

  # pre-flight: fail fast, no retry — Ollama being down isn't a "try again" situation
  check_ollama_available()

  logger.info(f"Loading Wikipedia page for query: '{query}'")
  page_loader = WikipediaLoader(query=query, lang='en', load_max_docs=1)
  document = retry_with_backoff(lambda: page_loader.load())

  if not document:
      logger.error(f"No Wikipedia page found for query: '{query}'")
      raise ValueError("No document loaded")

  logger.info(f"Loaded {len(document)} document(s). Title: {document[0].metadata.get('title')}")

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
  splitted_documents = text_splitter.split_documents(document)
  logger.info(f"Split into {len(splitted_documents)} chunks")

  logger.info(f"Initializing embedder ({embedding_model} via Ollama)")
  embedder = OllamaEmbeddings(model=embedding_model)

  # --- 3: no silent overwrite — branch on whether the index already exists ---
  if os.path.exists(database_name):
      logger.info(f"Existing index '{database_name}' found — loading and appending new chunks.")
      try:
          cities_db = FAISS.load_local(database_name, embedder, allow_dangerous_deserialization=True)
          cities_db.add_documents(splitted_documents)
      except Exception as e:
          logger.exception(f"Failed to load/append to existing database: {e}")
          raise
  else:
      logger.info(f"No existing index found — creating new database '{database_name}'.")
      try:
          cities_db = FAISS.from_documents(splitted_documents, embedder)
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