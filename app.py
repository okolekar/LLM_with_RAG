import os
import gradio as gr
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Loading environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Terminal-Gradio-App")

# Core LangChain Logic
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a travel guide assistant. Please respond to the tourist query"),
        ("user", "Query:{query}")
    ]
)

# UPGRADED: OllamaLLM is the modern replacement for Ollama inside LangChain
llm = OllamaLLM(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Function that the UI will call
def get_travel_guide(city_name):
    if not city_name.strip():
        return "Please enter a valid city name!"
    try:
        return chain.invoke({"query": city_name})
    except Exception as e:
        return f"Error: {e}\n\nMake sure Ollama server is running in your Colab terminal background."

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
                lines=1
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