import argparse
import csv
import sys
from engine.triage_scorer import run_pipeline

def process_csv(filepath: str, is_verbose: bool):
    """Reads a CSV and runs the pipeline on each post."""
    try:
        with open(filepath, mode='r', encoding='utf-8') as f:
            # We assume your CSV has a column header named 'post'
            reader = csv.DictReader(f)
            
            if not reader.fieldnames or 'post' not in [field.lower() for field in reader.fieldnames]:
                print("Error: Your CSV must contain a column named 'post'")
                sys.exit(1)
            
            for row_num, row in enumerate(reader, 1):
                query = row.get('post', '')
                if not query.strip():
                    continue
                    
                print(f"\n{'='*50}")
                print(f"Processing Post #{row_num} from CSV...")
                print(f"{'='*50}")
                
                run_pipeline(query, is_verbose=is_verbose)
                
    except FileNotFoundError:
        print(f"Error: Could not find the file at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while processing the CSV: {e}")
        sys.exit(1)

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Triage AI Pipeline: Classify risk in forum posts.")
    
    # Optional positional argument for a single string query
    parser.add_argument(
        "query", 
        nargs="?", 
        default="", 
        help="A single text string to classify (wrap in quotes if it has spaces)."
    )
    
    # Flags
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

    # Routing logic based on what the user typed
    if args.file:
        process_csv(args.file, args.verbose)
    elif args.query:
        run_pipeline(args.query, is_verbose=args.verbose)
    else:
        # If they just run `python3 src/main.py` with no arguments, show the help menu
        parser.print_help()

if __name__ == "__main__":
    main()