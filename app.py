import os
from functools import lru_cache

import gradio as gr
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# Loading environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Terminal-Gradio-App")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

DEFAULT_DATABASE_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tourism_Cities")
DEFAULT_EMBEDDER = "qwen3-embedding"

# Core LangChain Logic
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert travel guide assistant. Use the retrieved context below whenever possible. "
            "Answer with a concise travel guide that includes top attractions, local food to try, and a practical tip. "
            "If the context is insufficient, say so clearly instead of guessing.",
        ),
        ("user", "City to explore: {query}\n\nRelevant context:\n{context}"),
    ]
)

llm = OllamaLLM(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


@lru_cache(maxsize=1)
def load_vectorstore():
    embedder = OllamaEmbeddings(model=DEFAULT_EMBEDDER)
    return FAISS.load_local(DEFAULT_DATABASE_NAME, embedder, allow_dangerous_deserialization=True)


def build_context_text(documents):
    if not documents:
        return ""

    context_parts = []
    for index, document in enumerate(documents, start=1):
        title = document.metadata.get("title", "Unknown source")
        content = document.page_content.strip()
        context_parts.append(f"[{index}] Source: {title}\n{content}")
    return "\n\n".join(context_parts)


# Function that the UI will call
def get_travel_guide(city_name):
    if not city_name.strip():
        return "Please enter a valid city name!"

    try:
        vectorstore = load_vectorstore()
        documents = vectorstore.similarity_search(city_name, k=3)
        context_text = build_context_text(documents)

        if not context_text:
            return "I could not find relevant information in the local knowledge base for that city."

        return chain.invoke({"query": city_name, "context": context_text})
    except Exception as e:
        return f"Error: {e}\n\nMake sure Ollama server is running and the FAISS database exists."


# ==========================================
# GRADIO FANCY UI CONFIGURATION
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗺️ Travel Assistant with Gemma 2b")
    gr.Markdown("Enter the city you wish to visit below to generate a tailored travel guide.")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Destination City",
                placeholder="e.g., Tokyo, Paris, New York...",
                lines=1,
            )
            submit_btn = gr.Button("Explore Destination", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Assistant Response", lines=10, interactive=False)

    # Triggering logic
    submit_btn.click(fn=get_travel_guide, inputs=input_text, outputs=output_text)
    input_text.submit(fn=get_travel_guide, inputs=input_text, outputs=output_text)


# Launching with share=True to generate a public link out of Colab
if __name__ == "__main__":
    demo.launch(share=True)