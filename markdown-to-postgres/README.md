# Markdown to PGVector Loader

This project reads Markdown files, generates text embeddings using Sentence Transformers, and stores them in a PostgreSQL database with the PGVector extension for similarity searching.

## Prerequisites

1.  **Python:** Version 3.8 or higher recommended.
2.  **PostgreSQL:** A running PostgreSQL instance (version 11+).
3.  **PGVector Extension:** The `vector` extension must be installed in your PostgreSQL database.
    *   Follow the installation instructions: https://github.com/pgvector/pgvector#installation
    *   After installing the extension binaries/packages, connect to your target database using `psql` and run: `CREATE EXTENSION IF NOT EXISTS vector;`
4.  **NLTK Data:** The `punkt` tokenizer data is needed for chunking. It will be downloaded automatically on the first run if missing, or you can download it manually:
    ```bash
    python -m nltk.downloader punkt
    ```

## Setup

1.  **Clone the repository (or create the files as described):**
    ```bash
    # git clone <your-repo-url> # If you put this in git
    cd markdown_to_pgvector
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    *   Copy the `.env.example` to `.env` (or create `.env` manually).
    *   Edit the `.env` file and fill in your actual PostgreSQL database connection details (`DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`).
    *   You can optionally change the `EMBEDDING_MODEL`, `MARKDOWN_DIR`, `CHUNK_SIZE`, and `CHUNK_OVERLAP`.

5.  **Add Markdown Files:**
    *   Place your `.md` files inside the `sample_docs/` directory (or the directory specified by `MARKDOWN_DIR` in `.env`).

6.  **Enable PG Extension**
    ```
    psql db postgres
    CREATE EXTENSION IF NOT EXISTS vector;
    exit
    ```
## Usage

### 1. Load Data

Run the `load_data.py` script to process the Markdown files and load them into the database. The script will:
*   Connect to the database.
*   Ensure the `vector` extension exists.
*   Create the `documents` table if it doesn't exist.
*   Scan the `MARKDOWN_DIR` for `.md` files.
*   Read, parse, and chunk the text content.
*   Generate embeddings for each chunk.
*   Insert the source, chunk content, and embedding into the `documents` table.

```bash
python -m src.load_data
