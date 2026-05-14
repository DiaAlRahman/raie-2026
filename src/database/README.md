# Database — Tasks 1-3

Documentation for the data aggregation, ChromaDB initialisation, and
embedding pipeline that feed the rest of the project.

## Task 1 — Data Aggregation

### Topic mapping

| File                  | Topic                              | Annotator |
| --------------------- | ---------------------------------- | --------- |
| `data/raw_data/sheet_1.xlsx` | `anxiety`                          | Cooper    |
| `data/raw_data/sheet_2.xlsx` | `suicidal-thoughts-and-self-harm`  | Rhett     |
| `data/raw_data/sheet_3.xlsx` | `ptsd-and-trauma`                  | Omar      |
| `data/raw_data/sheet_4.xlsx` | `depression`                       | Yukang    |
| `data/raw_data/sheet_5.xlsx` | `positive-content`                 | —         |

`positive-content` is the non-crisis baseline (pets, hobbies, daily positive
moments) — every row is `in_crisis=False`. Without it the retrieval system
has no clear "safe" reference posts, so Top-3 neighbour lookups would tilt
toward false-positive crisis classifications.

### Cleaning rules

* `in_crisis` normalised to a real boolean. The source sheets disagree —
  Cooper and Omar used `0` / `1`, Rhett used native booleans, Yukang used
  `TRUE` / `True\n` strings. All flavours are coerced via
  `normalize_in_crisis()`:

    ```python
    def normalize_in_crisis(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return None
            return bool(int(value))
        s = str(value).strip().lower()
        if s in {"true", "t", "1", "yes"}:
            return True
        if s in {"false", "f", "0", "no"}:
            return False
        if s in {"", "nan", "none"}:
            return None
    ```

* Rows with empty `post` are dropped. There are 3 in **sheet_3** (post ids
  3, 25, 235) and 1 in **sheet_5** (post id 201) that were left blank.
* Rows with a `post` but no `in_crisis` label are dropped. There are 2
  (**sheet_1** id 137, **sheet_4** id 212).
* Whitespace is stripped from text fields. Smart quotes are kept (they
  carry meaning in the explanations).
* A globally-unique `master_post_id` (`<topic>_<original_id>`) is added.

### Output

`data/master_dataset.csv` — 1244 rows × 6 columns
(`master_post_id`, `topic`, `original_post_id`, `post`, `in_crisis`, `explanation`).

```
Per-topic counts (in_crisis True / False):
                                in_crisis_true  in_crisis_false  total
topic
anxiety                                     56              193    249
depression                                 149              100    249
positive-content                             0              249    249
ptsd-and-trauma                             37              210    247
suicidal-thoughts-and-self-harm            185               65    250

Grand total: 1244 rows
```

## Task 2 — ChromaDB Initialisation

Collection `beyondblue_posts` on a persistent ChromaDB store at
`chroma_db/` (repo root). Embedding function: `mxbai-embed-large` via
Ollama, cosine space, 1024-dim.

### Per-row layout

| Field                | Where it lives                |
| -------------------- | ----------------------------- |
| post text (embedded) | `documents` argument          |
| `post`               | metadata (full original text) |
| `in_crisis`          | metadata (bool)               |
| `explanation`        | metadata (str)                |
| row id               | `master_post_id` from the CSV |

The post text is truncated to 1800 chars **before embedding** (mxbai's
512-token context window won't accept longer inputs — 153 of 1244 rows
exceed this). The full original text is kept verbatim in `metadata['post']`
so downstream LLM scoring sees complete content; only the vector is built
from the truncated version.

## Task 3 — Embedding Pipeline

`build_db.py` reads `master_dataset.csv`, runs each post through
`mxbai-embed-large` via Ollama, and upserts the 1024-dim vectors into the
Task 2 collection.

### Prerequisites

1. Ollama is running on the host with `mxbai-embed-large` pulled.
2. `data/master_dataset.csv` exists (download from team OneDrive).

### Run

```bash
python src/database/build_db.py             # full run, resume-safe
python src/database/build_db.py --limit 10  # smoke test on first 10 rows
python src/database/build_db.py --reset     # wipe collection and start over
```

Resume is automatic — re-running after a crash skips any `master_post_id`
already in the collection. CPU-only Ollama runs around 2-4 rows/s for
mxbai-embed-large; on a GPU expect 20+ rows/s. A clean run of 1244 rows
takes ~6 minutes on CPU.

### Connecting downstream

`query_db.py` (Task 4) should import the collection setup directly:

```python
from src.database.build_db import init_collection

collection = init_collection()
results = collection.query(query_texts=[new_post], n_results=3)
# results["distances"][0] are the cosine distances Team 3 needs for scoring
```
