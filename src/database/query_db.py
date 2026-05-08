import os
import chromadb
from chromadb.utils import embedding_functions

# 1. Setup Paths (Same as build script)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
CHROMA_DIR = os.path.join(PROJECT_ROOT, 'chroma_db')

def test_database():
    print("Connecting to ChromaDB...")
    
    # 2. Setup the exact same Ollama embedding function
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://127.0.0.1:11434/api/embeddings",
        model_name="mxbai-embed-large",
    )

    # 3. Connect to the existing database
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Notice we use 'get_collection' instead of 'get_or_create' this time
    collection = client.get_collection(
        name="triage_posts", 
        embedding_function=ollama_ef
    )

    # --- PROOF 1: DID IT WORK? ---
    total_posts = collection.count()
    print(f"\n✅ SUCCESS! Found {total_posts} embedded posts in the database.\n")
    if total_posts == 0:
        return

    # --- PROOF 2: IS IT GOOD ENOUGH? ---
    # Let's test it with a mock crisis post
    # test_post = "I am feeling completely overwhelmed and hopeless. I can't sleep and I just want the pain to stop."
    # test_post = 'I am happy! I will never kill myself. I have a great support system and I am grateful for my life.'
    test_post = """I will kill myself tonight. It's the only way out. I can't take this pain anymore. No one understands me and I feel so alone. It's insanity. I am depressed and anxious and all i can think about is how i have no friends no girlfriend no life. always sad and moping, overeating and rewatching the same shows again and again. i can't get out of my head at all."""
    
    print(f"🔍 Searching for similar posts to: '{test_post}'")
    print("=" * 60)

    # Run the similarity search
    results = collection.query(
        query_texts=[test_post],
        n_results=3 # Ask for the Top 3 matches
    )

    # Print the results nicely so we can read them
    for i in range(len(results['ids'][0])):
        # We used Cosine distance: closer to 0.0 means highly similar
        distance = results['distances'][0][i] 
        metadata = results['metadatas'][0][i]
        text = results['documents'][0][i]

        print(f"🏆 MATCH {i+1} | Distance Score: {distance:.4f}")
        print(f"Label: {metadata['in_crisis']}")
        print(f"Human Explanation: {metadata['explanation']}")
        # Print the first 200 characters of the matched post
        print(f"Post Text: {text[:200]}...") 
        print("-" * 60)

if __name__ == "__main__":
    test_database()