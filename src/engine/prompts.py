"""
Implement prompt to turn raw post data into a JSON output with boolean switches

Task 2 (LLM and Prompting)

Collection structure:
    post         -- the original post text (str)
    in_crisis    -- the human risk label (bool)
    explanation  -- the human reasoning behind the label (str)

The text fed to the embedding model is truncated to MAX_CHARS_FOR_EMBEDDING
(mxbai-embed-large has a 512-token context window). The full original text
is preserved in metadata['post'].

Run:
    python src/database/build_db.py             # full run
    python src/database/build_db.py --reset     # wipe collection and start over
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict

import pandas as pd
import chromadb
import httpx
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from ollama import Client
from ollama import chat
from ollama import ChatResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = str(REPO_ROOT / "chroma_db")

COLLECTION_NAME = "beyondblue_posts"
LLM_MODEL = "mistral" # Attempted models: llama3.2, llama3.1, mistral
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
OLLAMA_TIMEOUT = 120.0

with open("src/engine/bool_prompt.txt", "r") as f:
    PROMPT_BOOL = f.read()




#PROMPT_BOOL = pd.read_csv("src/engine/bool_prompt.txt", sep='\r', header=None)

test_post = "I’ve been diagnosed with ADHD, anxiety and depression, but I’m really struggling. I try to talk to peers about it but I most in similar situations have some huge trauma they went through in their life, but my childhood and everything was fairly normal. I don’t know why I’m struggling and it feels like I’m overreacting or doing it to myself. Maybe I am, idk. I just don’t understand why I feel the way I do as most seem to have some kind of clear trauma that lead them to feel the way they do, I don’t mean to sound self absorbed as I understand that trauma like they go through must be absolutely horrific and I sympathise a lot. But I feel like I’m drowning but I don’t know how to swim."
PROMPT_BOOL = f"{PROMPT_BOOL}\n{test_post}"
print(PROMPT_BOOL)

client = Client(
    host=OLLAMA_HOST,

)

response = client.generate(model=LLM_MODEL, prompt=PROMPT_BOOL) 
print(response['response'])