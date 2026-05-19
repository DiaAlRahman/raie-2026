# Triage AI Pipeline

This project is an AI-powered pipeline designed to analyze and classify the risk level of mental health forum posts. It uses a combination of a local Large Language Model (Mistral) for classification and reasoning, and a local vector database (ChromaDB) with an embedding model (`mxbai-embed-large`) for retrieving contextually similar historical posts.

## Getting Started

Follow these steps to set up and run the pipeline on your machine.

### Step 0: Install Prerequisites

Before setting up the project, you need to install [Ollama](https://ollama.com/), which runs the local AI models.

**1. Install Ollama**
* **Mac:** Download the installer from the [Ollama website](https://ollama.com/download).
* **Windows:** Download the installer from the [Ollama website](https://ollama.com/download).
* **Linux:** Run the following command in your terminal:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh

    ```

**2. Pull the Required Models**
Once Ollama is installed, open your computer's native terminal (not the VS Code terminal yet) and run these two commands to download the necessary AI models:

```bash
ollama run mistral
ollama run mxbai-embed-large

```

*(Note: These are large files and may take a few minutes to download depending on your internet connection).*

### Step 1: Open in VS Code (DevContainer)

This project is configured to run inside a Docker DevContainer, ensuring you have all the correct Python dependencies without cluttering your local machine.

1. Open the `ProjectRaie` folder in **Visual Studio Code**.
2. VS Code should prompt you to **"Reopen in Container"** in the bottom right corner. Click it.
* *If you don't see the prompt:* Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac), type "Dev Containers: Reopen in Container", and press Enter.


3. Wait a moment for VS Code to build the container and install the requirements.

### Step 2: Running the Pipeline

Once the container is running, open the integrated terminal in VS Code (`Ctrl+` `or`Terminal > New Terminal`).

The entry point for the application is `src/main.py`. You can interact with the pipeline in two main ways: testing single posts or processing batches via CSV.

#### Method A: Testing a Single Post

You can pass a single sentence or paragraph directly into the terminal. You do need to wrap it in quotes.

```bash
python3 src/main.py "I am feeling really lost today"

```

#### Method B: Processing a Batch (CSV)

To process a list of posts, you can pass a CSV file using the `-f` or `--file` flag.

There is a sample test file provided in the repository. Try running:

```bash
python3 src/main.py -f data/test/simple_tests.csv

```

#### Additional Options & Flags

You can customize the pipeline's behavior by chaining different flags:

* **`-v` or `--verbose`:** Enables verbose output. This will print the raw JSON output from the LLM, as well as the top 3 similar posts retrieved from the database and their distance scores.
* *Example:* `python3 src/main.py "I feel hopeless" -v`


* **`-i` or `--ignore`:** Ignores the manual human-review prompt. By default, if the AI detects a high-risk post or has low confidence, the pipeline will pause and ask you to confirm the severity. The `-i` flag suppresses this pause and auto-logs the post, which is essential for processing large CSV batches unattended.
* *Example:* `python3 src/main.py -f data/test/simple_tests.csv -v -i`


* **`-h` or `--help`:** Displays the help menu with all available commands.

### Outputs

Regardless of whether you run a single query or a batch CSV, the pipeline automatically generates (or updates) an `output.csv` file.

You can find the results, including the AI's explanation, risk score, and severity classification, located at:
`outputs/output.csv`