import os
import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import mlflow
import dagshub

# --- 1. PLACE THE TRACKING CODE HERE ---
def init_tracking():
    # Get credentials from Hugging Face Secrets
    user = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    if user and password:
        # Set them as environment variables so MLflow sees them
        os.environ['MLFLOW_TRACKING_USERNAME'] = user
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password
        
        try:
            # bootstrap=True helps DagsHub set up everything automatically
            dagshub.init(repo_owner="vinodshukla", repo_name="AI-Lab", bootstrap=True)
            mlflow.set_experiment("AI-Lab-Summarizer")
            print("✅ DagsHub & MLflow Tracking Active")
        except Exception as e:
            print(f"⚠️ Tracking skipped: {e}")
    else:
        print("⚠️ No credentials found. Tracking skipped to prevent hang.")

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
