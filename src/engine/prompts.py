"""
Implement prompt to turn raw post data into a JSON output with boolean switches

Task 2 (LLM and Prompting)
         
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ollama import Client

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = str(REPO_ROOT / "chroma_db")

COLLECTION_NAME = "beyondblue_posts"
LLM_MODEL = "mistral" # Attempted models: llama3.2, llama3.1, mistral
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
OLLAMA_TIMEOUT = 120.0

client = Client(
    host=OLLAMA_HOST,
)

def generate_output(model: str, prompt: str, options: dict = {"temperature": 0}):
    response_stream = client.generate(model, prompt, options=options, stream=True) 
    
    full_response = ""
    spinner = ['|', '/', '-', '\\']
    step = 0
    
    for chunk in response_stream:
        text_chunk = chunk['response']
        full_response += text_chunk
        
        sys.stdout.write(f"\rThinking... {spinner[step % 4]}")
        sys.stdout.flush()
        step += 1
        
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()
    
    return full_response