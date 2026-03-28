import os
import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# --- Configuration ---
# This folder must be in your GitHub repo for the Space to see it
LOCAL_MODEL_PATH = "./summarizer_model" 
# Fallback to a public model if your local folder is missing or too large to push
HF_FALLBACK_MODEL = "t5-small" 

# 1. Environment Detection
IS_SPACES = "SPACE_ID" in os.environ

# 2. Model Source Logic
if os.path.exists(LOCAL_MODEL_PATH):
    model_source = LOCAL_MODEL_PATH
    print(f"Loading model from local folder: {model_source}")
else:
    model_source = HF_FALLBACK_MODEL
    print(f"Local folder not found. Falling back to Hub: {model_source}")

# --- Load Model & Tokenizer ---
# legacy=False avoids SentencePiece errors in different environments
tokenizer = T5Tokenizer.from_pretrained(model_source, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_source)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Inference Logic ---
def summarize(text, max_len, min_len, beam_size):
    input_text = "summarize: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=int(max_len),
        min_length=int(min_len),
        num_beams=int(beam_size),
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- Interface ---
demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(label="Article", placeholder="Paste text here...", lines=5),
        gr.Slider(20, 200, value=80, step=5, label="Max Length"),
        gr.Slider(10, 100, value=20, step=5, label="Min Length"),
        gr.Slider(1, 10, value=4, step=1, label="Beam Size")
    ],
    outputs="text",
    title="Summarizer Demo"
)

# --- Compatibility Layer ---
def launch_app():
    """
    Universal launch function.
    server_name="0.0.0.0" allows access from outside Docker.
    server_port=7860 is the standard port for Gradio/Hugging Face.
    """
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    # This runs automatically on Hugging Face Spaces and Docker
    launch_app()