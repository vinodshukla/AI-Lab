# Use a lightweight Python base
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Optimize Python settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME="0.0.0.0"

# 1. FIX: Use the correct CPU-only index-url (Must be exactly this)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Copy and install other requirements
# Ensure 'torch' is REMOVED from your requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir protobuf==3.20.3

# 3. Copy your app code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]