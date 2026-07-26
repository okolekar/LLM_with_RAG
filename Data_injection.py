import wikipedia
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

wikipedia.set_user_agent("MyRAGProjectApp/1.0 (contact: omkar.kolekar@viit.ac.in)")

page_loader = WikipediaLoader(query="Paris",lang='en',load_max_docs=1)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

document = page_loader.load()

for doc in page_loader.lazy_load():
    print(type(doc))

# for temp in document:
#     print(temp)

splitted_documents = text_splitter.split_documents(document)


