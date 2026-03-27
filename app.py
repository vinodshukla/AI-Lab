import os
import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import mlflow
import dagshub

# --- 1. MLOps (Replace with your actual DagsHub username) ---


def init_tracking():
    # Force DagsHub to use your Hugging Face Secrets
    if "MLFLOW_TRACKING_PASSWORD" in os.environ:
        os.environ["DAGSHUB_USER_TOKEN"] = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    
    try:
        # This will now log in SILENTLY on Hugging Face
        dagshub.init(repo_owner='vinodshukla', repo_name='AI-Lab', mlflow=True)
        print("✅ MLflow Tracking Active (Headless Login)")
    except Exception as e:
        print(f"⚠️ Tracking skipped: {e}")

init_tracking()

# --- 2. Load Model ---
MODEL_NAME = "./summarizer_model" if os.path.exists("./summarizer_model") else "t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to("cpu")

# --- 3. Summarize Function ---
def summarize(text, max_len, min_len, beam_size):
    with mlflow.start_run(run_name="Gradio-Inference", nested=True):
        mlflow.log_params({"max_len": max_len, "min_len": min_len, "beams": beam_size})
        inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs["input_ids"], max_length=int(max_len), min_length=int(min_len), num_beams=int(beam_size))
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        mlflow.log_metric("summary_len", len(summary))
        return summary

# --- 4. Define the Demo ---
demo = gr.Interface(
    fn=summarize,
    inputs=[gr.Textbox(lines=5), gr.Slider(20, 200, 80), gr.Slider(10, 100, 20), gr.Slider(1, 10, 4)],
    outputs="text",
    title="AI-Lab"
)

# --- 5. THE MISSING FUNCTION ---
def launch_app():
    """Starts the Gradio app"""
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_app()
