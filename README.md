# Beyond Blue NLP Triage Pipeline

Welcome to the Beyond Blue Risk-Aware Extraction project. This repository contains our Python pipeline, but because we are working with Large Language Models and heavy vector databases, we need to do a quick local setup first.

## Getting Started (Read This First)

### Step 1: Install the AI Engine (Host Machine)
We run the AI models natively on your computer to ensure we have hardware/GPU acceleration. **Do not run these commands inside the Docker container.**

1. Download and install [Ollama](https://ollama.com/).
2. Open your standard Mac/Windows terminal and run these two commands to download our models:
   `ollama pull llama3`
   `ollama pull mxbai-embed-large`
3. Ensure the Ollama app is running in the background.

### Step 2: Download the Data
Our scraped JSON data and populated ChromaDB vector database are too large for GitHub. 

1. Go to our team [One Drive Link Here].
2. Download the `data.zip` and `chroma_db.zip` files.
3. Unzip them and place the `data/` and `chroma_db/` folders directly into the root of this project folder. *(Note: Our `.gitignore` is set up so you won't accidentally push these back to GitHub).*

### Step 3: Build and open the Dev Container
Now that your models and data are ready, let's boot up the code.

1. Open this project folder in **VS Code**.
2. If prompted, install the "Dev Containers" extension.
3. A popup will appear in the bottom right saying "Folder contains a Dev Container configuration file." Click **Reopen in Container**.
4. VS Code will build the Python environment and connect automatically to your local Ollama instance.