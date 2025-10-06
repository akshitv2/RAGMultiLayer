# RAGMultiLayer

# Wikipedia Hierarchical RAG (Retrieval-Augmented Generation)

## Project Overview

This project implements a **Hierarchical Retrieval-Augmented Generation (RAG)** system using the large-scale *
*Wikimedia/Wikipedia** dataset.

The core innovation is a **two-layer chunking strategy** designed to maximize retrieval precision (using small chunks
for search) while ensuring context completeness (using large chunks for the LLM). This architecture is optimized for
low-resource environments using a highly efficient open-source embedding model.

### Key Features

* **Two-Layer Chunking:** Implements the "Small-to-Large" or "Parent Document" retrieval pattern.
* **Vector Database:** Uses **ChromaDB** with two distinct collections to manage the hierarchical data.
* **Efficient Embeddings:** Leverages the fast and compact **`all-MiniLM-L6-v2`** model for rapid indexing and low
  computational cost.

---

## 🏗Architecture: Single Client, Two Collections

The system relies on a single ChromaDB instance containing two separate collections, linked by a shared `parent_id` in
the metadata.

| Layer               | Collection Name     | Purpose                                                                                                   | Vectorized?           | Chunk Size (Tokens)   |
|:--------------------|:--------------------|:----------------------------------------------------------------------------------------------------------|:----------------------|:----------------------|
| **Layer 1: Child**  | `wiki_small_chunks` | **Search Target.** Stores fine-grained, highly relevant text snippets for semantic search.                | **Yes** (Embedded)    | $\approx 300 - 500$   |
| **Layer 2: Parent** | `wiki_large_chunks` | **LLM Context.** Stores the broader, context-rich sections to be passed to the LLM for answer generation. | **No** (ID Retrieval) | $\approx 1500 - 2000$ |

### 🔍 Retrieval Flow

1. **Search:** The user query is embedded and searched against the **Small Chunks** collection (`wiki_small_chunks`).
2. **Retrieve ID:** The system retrieves the metadata of the most relevant small chunk, extracting its unique *
   *`parent_id`**.
3. **Fetch Context:** The `parent_id` is used to fetch the full, corresponding **Large Chunk** from the
   `wiki_large_chunks` collection.
4. **Generate:** The Large Chunk (the full context) is passed to the LLM (not included in this setup) for final answer
   generation.

---

## 🛠️ Setup and Installation

### Prerequisites

* Python 3.8+
* The project is configured for **local CPU execution** for maximum accessibility.

### Installation

Clone the repository and install the required dependencies:

```bash
# Clone your project repository (assuming one exists)
git clone [your-repo-link]
cd wikipedia-hierarchical-rag

# Install all necessary libraries
pip install chromadb datasets sentence-transformers langchain-text-splitters uuid

python3 -m venv RAGmult
RAGmult/bin/activate
```

### Data Loading

The project uses the `wikimedia/wikipedia` dataset (specifically the 20231101.en configuration). The provided script
will automatically load a small sample for demonstration.

How to Run the Ingestion Script
The core logic is handled by the ingestion script (which can be saved as, e.g., ingest.py). This script performs the chunking, embedding, and storage steps.

1. The Ingestion Command
Execute the script to process the data and populate the ChromaDB:
`python ingest.py`
2. Output and Persistence
Upon successful execution, a local directory named ./chroma_wikipedia_db will be created, containing the persistent storage files for both the small and large chunk collections.
3. 