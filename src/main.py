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

HEADERS = [
    "post_id", "post", "in_crisis", "explanation", 
    "risk_score", "severity", "confidence_score", "requires_human_review"
]

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
            "requires_human_review": result.get("requires_human_review")
        })

def process_csv(filepath: str, is_verbose: bool):
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
                
                result = run_pipeline(query, is_verbose=is_verbose)
                if result:
                    write_result_to_csv(post_id, query, result)
                    
    except FileNotFoundError:
        print(f"Error: Could not find the file at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while processing the CSV: {e}")
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
        help="Enable verbose output to see top 3 similar posts from the database."
    )
    
    parser.add_argument(
        "-f", "--file", 
        type=str, 
        help="Path to a CSV file to process multiple posts automatically."
    )

    args = parser.parse_args()

    # Routing logic
    if args.file:
        init_output_csv()
        process_csv(args.file, args.verbose)
        print(f"\n✅ Processing complete. Results saved to {OUTPUT_FILE}")
    elif args.query:
        init_output_csv()
        full_query = " ".join(args.query)
        result = run_pipeline(full_query, is_verbose=args.verbose)
        if result:
            write_result_to_csv("single_cli_query", full_query, result)
            print(f"\n✅ Processing complete. Result saved to {OUTPUT_FILE}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()