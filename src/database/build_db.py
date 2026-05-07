import os
import glob
import csv
import chromadb
from chromadb.utils import embedding_functions

# 1. Setup Paths
# Since this script is in src/database/, we navigate up two levels to the root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
CHROMA_DIR = os.path.join(PROJECT_ROOT, 'chroma_db')

def build_database():
    print("Initializing ChromaDB...")
    
    total_successful = 0
    total_failed = 0
    
    # 2. Setup Ollama Embedding Function
    # This automatically routes text through Ollama when added to the DB
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="mxbai-embed-large",
    )

    # Initialize persistent Chroma client
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Create or load the collection
    collection = client.get_or_create_collection(
        name="triage_posts",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"} # Cosine similarity is best practice for text embeddings
    )

    # 3. Read and Process CSV Files
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"\nProcessing {file_name}...")

        # utf-8-sig handles the byte order mark (BOM) common in Google Sheets exports
        with open(file_path, mode='r', encoding='utf-8-sig') as f: 
            reader = csv.DictReader(f)
            
            ids = []
            documents = []
            metadatas = []

            for row in reader:
                # Extracting based on your screenshot's headers
                post_id = row.get('post_id')
                text = row.get('post')
                in_crisis = row.get('in_crisis')
                explanation = row.get('explanation')

                if not post_id or not text:
                    continue # Skip empty or invalid rows
                  
                # Split the text if it's too long
                text_chunks = chunk_text(text)
                
                # Add each chunk to our lists individually
                for index, chunk in enumerate(text_chunks):
                    # We make a unique ID for Chroma (e.g., "post123_chunk0")
                    ids.append(f"{post_id}_chunk{index}") 
                    documents.append(chunk)
                    
                    metadatas.append({
                        "original_post_id": str(post_id), # Keep the original ID for reference
                        "in_crisis": str(in_crisis),
                        "explanation": str(explanation),
                        "source_file": file_name
                    })
                    
                ids.append(str(post_id))
                documents.append(text)
                
                # Store the labels as metadata so we can retrieve them later
                metadatas.append({
                    "in_crisis": str(in_crisis),
                    "explanation": str(explanation),
                    "source_file": file_name
                })

            # 4. Add to Chroma in Batches
            BATCH_SIZE = 1
            total_posts = len(ids)

            for i in range(0, total_posts, BATCH_SIZE):
                batch_ids = ids[i:i+BATCH_SIZE]
                batch_docs = documents[i:i+BATCH_SIZE]
                batch_metas = metadatas[i:i+BATCH_SIZE]

                print(f"  Embedding and adding batch {i} to {min(i + BATCH_SIZE, total_posts)} of {total_posts}...")
                
                try:
                  # The embedding function is called automatically here
                  collection.add(
                      documents=batch_docs,
                      metadatas=batch_metas,
                      ids=batch_ids
                  )
                  total_successful += len(batch_ids)
                except Exception as e:
                    # If it fails, add to the failure counter and print a quiet warning
                    total_failed += len(batch_ids)
                    error_msg = str(e).lower()
                    if "context length" in error_msg or "input length" in error_msg:
                        print(f"    ⚠️ Skipped {batch_ids[0]}: Text formatting caused a memory overflow.")
                    else:
                        print(f"    ⚠️ Skipped {batch_ids[0]}: Unexpected error - {e}")

        print(f"Finished adding {file_name}")

    print("\nDatabase build complete! Embeddings are stored in /chroma_db")
    print(f"Successful embeddings: {total_successful}")
    print(f"Failed embeddings: {total_failed}")

def chunk_text(text, max_chars=1500, overlap=200):
    """Splits text into overlapping chunks by characters to strictly prevent memory crashes."""
    # If the text is short enough, just return it as one piece
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    # Step through the text by character count, ensuring overlap
    for i in range(0, len(text), max_chars - overlap):
        chunk = text[i:i + max_chars]
        chunks.append(chunk)
        
    return chunks

if __name__ == "__main__":
    build_database()