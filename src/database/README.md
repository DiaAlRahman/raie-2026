# Database — Tasks 1-3

Documentation for the data aggregation, ChromaDB initialisation, and embedding pipeline.

## Task 1 — Data Aggregation

### Topic mapping

| File                  | Topic                              | Annotator |
| --------------------- | ---------------------------------- | --------- |
| `data/raw_data/sheet_1.xlsx` | `anxiety`                          | Cooper    |
| `data/raw_data/sheet_2.xlsx` | `suicidal-thoughts-and-self-harm`  | Rhett     |
| `data/raw_data/sheet_3.xlsx` | `ptsd-and-trauma`                  | Omar      |
| `data/raw_data/sheet_4.xlsx` | `depression`                       | Yukang    |
| `data/raw_data/sheet_5.xlsx` | `positive-content`                 | —         |

`positive-content` is the non-crisis baseline (pets, hobbies, daily positive moments)

### Cleaning rules

* `in_crisis` normalised to a real boolean. All flavours are coerced via `normalize_in_crisis()`:

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
* Whitespace is stripped from text fields.
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

Collection `beyondblue_posts` on a persistent ChromaDB store at `chroma_db/`. 
Embedding function: `mxbai-embed-large` via Ollama, cosine space, 1024-dim.

### Per-row layout

| Field                | Where it lives                |
| -------------------- | ----------------------------- |
| post text (embedded) | `documents` argument          |
| `post`               | metadata (full original text) |
| `in_crisis`          | metadata (bool)               |
| `explanation`        | metadata (str)                |
| row id               | `master_post_id` from the CSV |

The post text is truncated to 1800 chars **before embedding**. The full original text is kept verbatim in `metadata['post']`

## Task 3 — Embedding Pipeline

`build_db.py` reads `master_dataset.csv`, runs each post through
`mxbai-embed-large` via Ollama, and upserts the 1024-dim vectors into the
database collection.

### Prerequisites

1. Ollama is running on the host with `mxbai-embed-large` pulled.
2. `data/master_dataset.csv` exists.

### Run

```bash
python src/database/build_db.py             # full run
python src/database/build_db.py --reset     # wipe collection and start over
```

Resume is automatic, re-running after a crash skips any `master_post_id`
already in the collection.
