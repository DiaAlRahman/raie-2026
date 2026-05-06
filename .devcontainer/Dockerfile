# 1. Start with a lightweight, official Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /workspace

# 3. Install basic system tools (ChromaDB sometimes needs C++ compilers to run)
RUN apt-get update && apt-get install -y build-essential curl git zstd

# 4. Copy the grocery list into the container
COPY requirements.txt .

# 5. Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Install the Ollama Engine
RUN curl -fsSL https://ollama.com/install.sh | sh