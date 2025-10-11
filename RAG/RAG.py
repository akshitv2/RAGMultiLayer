import os
import re
import uuid
from itertools import islice
from multiprocessing import Pool

import spacy
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from DB.qDrant import create_collection

def get_splitter(use_large: bool = False):
    if use_large:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
    return splitter


def process_document(document_and_index):
    """
    Processes a single document: cleans, splits, and prepares data for Chroma.
    This function will be run in parallel.

    Returns: A tuple (small_data, large_data) for this document.
    """
    doc_index, document = document_and_index

    # Initialize components within the worker process (important for multiprocessing)
    # The clean_document_spacy function implicitly handles spacy loading.
    large_splitter = get_splitter(use_large=True)
    small_splitter = get_splitter(use_large=False)

    doc_small_data = []
    doc_large_data = []

    text_to_split = document['text']
    doc_title = document['title']

    # 2. Large chunks (Parents)
    large_chunks = large_splitter.create_documents([text_to_split])
    for l_chunk in large_chunks:
        parent_id = str(uuid.uuid4())
        # doc_large_data["metadatas"].append(l_chunk.metadata)
        doc_large_data.append(PointStruct(id = parent_id, vector=[0.0] * 384, payload={"text": l_chunk.page_content, "title": doc_title}))
        # 3. Small chunks (Children)
        small_chunks = small_splitter.create_documents([l_chunk.page_content])
        for s_chunk in small_chunks:
            child_id = str(uuid.uuid4())

            doc_small_data.append(PointStruct(id=child_id, vector=[0.0] *384,
                                              payload={"text": s_chunk.page_content, "parent_id": parent_id}))
            # s_chunk.metadata["doc_index"] = doc_index  # Add original doc index for context
            # doc_small_data["ids"].append(child_id)
            # doc_small_data["documents"].append(s_chunk.page_content)
            # doc_small_data["metadatas"].append(s_chunk.metadata)
    return doc_small_data, doc_large_data


def rag_parallel(config_yaml, qdrant_client: QdrantClient):
    # Load dataset in streaming mode
    max_articles = 500
    dataset = load_dataset(path="wikimedia/wikipedia", name="20231101.en", cache_dir=config_yaml["dataset"]["dir"],split=f"train[:{max_articles}]")
                           # streaming=True)
    # Get iterable for the training split
    data_iterable = dataset

    small_collection_name = config_yaml['db']['collections']['collection_small']['name']
    large_collection_name = config_yaml['db']['collections']['collection_large']['name']

    # Initialize collections
    small_collection = create_collection(qdrant_client, small_collection_name)
    large_collection = create_collection(qdrant_client, large_collection_name)

    # Configuration

    batch_size = config_yaml['db']['insertion']['batch_size']  # Size of the batch to add to Chroma
    num_processes = os.cpu_count() - 1 if os.cpu_count() > 1 else 1  # Use N-1 cores

    print(f"Using {num_processes} processes for document processing.")

    # islice limits the number of articles processed
    limited_iterable = islice(enumerate(data_iterable), max_articles)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

    # Using multiprocessing pool to process documents in parallel
    with Pool(processes=num_processes) as pool:
        # map() applies the function to all elements and returns results in order
        results_iterator = pool.imap(process_document, limited_iterable, chunksize=50)  # Use imap for memory efficiency

        current_doc_small_data = []
        current_doc_large_data = []
        doc_count = 0

        for small_data, large_data in tqdm(results_iterator, total=max_articles, desc="Processing Articles"):
            doc_count += 1

            current_doc_small_data+=small_data
            current_doc_large_data+=large_data

            if doc_count % batch_size == 0:
                qdrant_client.upsert(
                    collection_name=small_collection_name,
                    points=current_doc_small_data
                )
                qdrant_client.upsert(
                    collection_name=large_collection_name,
                    points=current_doc_large_data
                )
                current_doc_small_data = []
                current_doc_large_data = []

    print("RAG data population complete.")
