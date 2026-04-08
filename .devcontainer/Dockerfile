# 1. Start with a lightweight, official Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /workspace

# 3. Install basic system tools (ChromaDB sometimes needs C++ compilers to run)
RUN apt-get update && apt-get install -y build-essential curl

# 4. Copy the grocery list into the container
COPY requirements.txt .

# 5. Install the Python libraries and playwright dependencies and browser
RUN pip install --no-cache-dir -r requirements.txt && playwright install --with-deps chromium

# 6. Install Xvfb to run the graphical application through Docker
RUN apt-get update && apt-get install -y xvfb xauth

# (You do NOT install Ollama here. Ollama runs on your host machine to access your GPU, not inside the container).