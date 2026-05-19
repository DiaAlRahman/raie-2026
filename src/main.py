import argparse
import csv
import sys
import os
from pathlib import Path
from engine.triage_scorer import run_pipeline

# ==========================================
# PATH RESOLUTION (Bulletproof Output Folder)
# ==========================================
# Get the directory this file is in (src/)
CURRENT_DIR = Path(__file__).resolve().parent
# Get the root of the repo (ProjectRaie/)
REPO_ROOT = CURRENT_DIR.parent

# Define absolute paths for the output folder and file
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_FILE = str(OUTPUT_DIR / "output.csv")

HEADERS = ["post_id", "in_crisis", "risk_score", "severity", "confidence_score", "human_review_choice", "post", "explanation"]

def init_output_csv():
    """Creates the outputs directory and initializes the CSV with headers (overwrites on each run)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)

def write_result_to_csv(post_id: str, query: str, result: dict):
    """Appends a single result row to the CSV."""
    with open(OUTPUT_FILE, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow({
            "post_id": post_id,
            "post": query,
            "in_crisis": result.get("in_crisis"),
            "explanation": result.get("explanation"),
            "risk_score": round(result.get("risk_score", 0), 2),
            "severity": result.get("severity"),
            "confidence_score": round(result.get("confidence_score", 0), 4),
            "human_review_choice": result.get("human_review_choice"),
        })

def process_csv(filepath: str, is_verbose: bool, ignore_human_review: bool):
    """Reads a CSV and runs the pipeline on each post."""
    try:
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if not reader.fieldnames or 'post' not in [field.lower() for field in reader.fieldnames]:
                print("Error: Your CSV must contain a column named 'post'")
                sys.exit(1)
            
            for row_num, row in enumerate(reader, 1):
                query = row.get('post', '')
                # Safely grab an ID if the CSV has one, otherwise generate a generic one
                post_id = row.get('post_id', row.get('id', f"csv_row_{row_num}"))
                
                if not query.strip():
                    continue
                    
                print(f"\n{'='*50}")
                print(f"Processing Post #{row_num} from CSV...")
                print(f"{'='*50}")
                try:
                    result = run_pipeline(query, is_verbose=is_verbose, ignore_human_review=ignore_human_review)
                    if result:
                        write_result_to_csv(post_id, query, result)
                except Exception as e:
                    print(f"Failed on Post #{row_num} (ID: {post_id}): {e}")
                    print("Skipping to next post...")
                    continue
    except FileNotFoundError:
        print(f"Error: Could not find the file at '{filepath}'")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Triage AI Pipeline: Classify risk in forum posts.")
    
    # Setup for capturing multiple loose words as a single query string without quotes
    parser.add_argument(
        "query", 
        nargs="*", 
        default=[], 
        help="The text string to classify."
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose output to see top 3 similar posts from the database and LLM boolean output."
    )
    
    parser.add_argument(
        "-f", "--file", 
        type=str, 
        help="Path to a CSV file to process multiple posts automatically."
    )
    
    parser.add_argument(
        "-i", "--ignore", 
        action="store_true", 
        help="Ignore human review prompts and process straight into the CSV."
    )

    args = parser.parse_args()

    # Routing logic
    if args.file:
        init_output_csv()
        process_csv(args.file, args.verbose, args.ignore)
        print(f"\nProcessing complete. Results saved to {OUTPUT_FILE}")
    elif args.query:
        init_output_csv()
        full_query = " ".join(args.query)
        result = run_pipeline(full_query, is_verbose=args.verbose, ignore_human_review=args.ignore)
        if result:
            write_result_to_csv("single_cli_query", full_query, result)
            print(f"\nProcessing complete. Result saved to {OUTPUT_FILE}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()