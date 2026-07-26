import logging
import wikipedia
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- logger setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
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

texts = [doc.page_content for doc in splitted_documents]
metadatas = [doc.metadata for doc in splitted_documents] 


logger.info("Initializing embedder (qwen3-embedding via Ollama)")
embedder = (
    OllamaEmbeddings(model="qwen3-embedding")
)

logger.info(f"Embedding {len(texts)} chunks...")
try:
    r1 = embedder.embed_documents(texts)
except Exception as e:
    logger.exception(f"Embedding failed: {e}")
    raise

logger.info(f"Done. {len(r1)} embeddings generated, dimension = {len(r1[0])}")
