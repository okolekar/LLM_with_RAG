import os
import logging
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

DEFAULT_DATABASE_NAME = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Tourism_Cities"
)
DEFAULT_EMBEDDER = "qwen3-embedding"
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
DEFAULT_OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")

# --- logger setup: shares data_ingestion.log, but with its own logger name ---
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "data_ingestion.log")  # same file as data_ingestion.py, on purpose

logger = logging.getLogger("city_registry")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def city_exists(
    query: str,
    embedder: OllamaEmbeddings,
    database_name: str = DEFAULT_DATABASE_NAME
) -> bool:
    """
    Check whether a city's Wikipedia page has already been ingested into the FAISS index.

    Args:
        query: the city name to check (e.g. "Paris")
        embedder: an OllamaEmbeddings instance — required to load the FAISS index,
                   even though we're only reading, not embedding anything new
        database_name: path to the FAISS index directory

    Returns:
        True if the city already exists in the index, False otherwise
    """
    normalized_query = query.strip().lower()

    if not os.path.exists(database_name):
        logger.info(f"No database found at '{database_name}' — '{query}' does not exist.")
        return False

    try:
        db = FAISS.load_local(database_name, embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.exception(f"Failed to load database '{database_name}': {e}")
        raise

    existing_titles = {
        doc.metadata.get("title", "").strip().lower()
        for doc in db.docstore._dict.values()
    }

    exists = normalized_query in existing_titles
    logger.info(f"'{query}' {'already exists' if exists else 'not found'} in '{database_name}'.")
    return exists


def ollama_model_available(
    model_name: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    port: int = DEFAULT_OLLAMA_PORT,
    timeout: int = 5,
) -> bool:
    """
    Verify that the Ollama server is reachable and that the specified model is available.
    """
    endpoint = f"http://{host}:{port}/v1/models/{urllib.parse.quote(model_name, safe='')}"
    request = urllib.request.Request(endpoint, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status == 200:
                logger.info(f"Ollama model '{model_name}' is available at {host}:{port}.")
                return True
            logger.warning(
                "Ollama model '%s' responded with status %s.", model_name, response.status
            )
            return False
    except HTTPError as exc:
        logger.warning(
            "Ollama model '%s' check failed with HTTP status %s: %s",
            model_name,
            exc.code,
            exc.reason,
        )
    except URLError as exc:
        logger.warning(
            "Ollama model '%s' check failed because the server is unreachable: %s",
            model_name,
            exc,
        )
    except Exception as exc:
        logger.exception("Unexpected error while checking Ollama model '%s': %s", model_name, exc)

    return False


if __name__ == "__main__":
    # quick manual test — run `python city_registry.py` directly to check a city by hand
    test_embedder = OllamaEmbeddings(model=DEFAULT_EMBEDDER)
    test_query = "Paris"
    result = city_exists(test_query, test_embedder)
    print(f"'{test_query}' exists in database: {result}")