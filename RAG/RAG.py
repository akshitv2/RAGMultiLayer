import asyncio
import os
import uuid
from itertools import islice
from multiprocessing import Pool

from datasets import load_dataset
from fastembed import SparseTextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from DB.qDrant import create_collection_with_embeddings, create_collection_without_embeddings, get_async_client, \
    QdrantConfig

import re
import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
#
# # Download required NLTK data
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
#
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
#
#
# def clean_wiki_text(text):
#     # Lowercase
#     text = text.lower()
#
#     # Remove URLs
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text)
#
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
#
#     # Remove numbers
#     text = re.sub(r'\d+', '', text)
#
#     # Remove special characters (anything not letters or spaces)
#     text = re.sub(r'[^a-z\s]', '', text)
#
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
#
#     # Tokenize, remove stopwords, and lemmatize
#     tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
#
#     # Join back into string
#     return ' '.join(tokens)


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

    doc_small_data = {"ids":[], "parent_ids":[], "texts":[], "titles":[]}
    doc_large_data = []

    text_to_split = document['text']
    doc_title = document['title']

    # 2. Large chunks (Parents)
    large_chunks = large_splitter.create_documents([text_to_split])
    for l_chunk in large_chunks:
        parent_id = str(uuid.uuid4())
        doc_large_data.append(
            PointStruct(id=parent_id,
                        vector=[0.0] * 2,
                        payload={"text": l_chunk.page_content, "title": doc_title}))
        small_chunks = small_splitter.create_documents([l_chunk.page_content])
        for s_chunk in small_chunks:
            child_id = str(uuid.uuid4())
            doc_small_data["ids"].append(child_id)
            doc_small_data["texts"].append(s_chunk.page_content)
            doc_small_data["parent_ids"].append(parent_id)
            doc_small_data["titles"].append(doc_title)
    return doc_small_data, doc_large_data

def point_construct_from_dict(entries:dict, dense_embedding_model: SentenceTransformer, sparse_model):
    result_array = []
    dense_embeddings = dense_embedding_model.encode(entries["texts"], batch_size=128)
    sparse_embeddings = sparse_model.embed(entries["texts"])
    for child_id, parent_id, text,doc_title, dense_emb, sparse_emb in zip(entries["ids"], entries["parent_ids"], entries["texts"],entries["titles"], dense_embeddings, sparse_embeddings):
        result_array.append(
            PointStruct(
                id=child_id,
                vector={
                    "dense": dense_emb.tolist(),  # Dense vector
                    "sparse": models.SparseVector(
                        indices=sparse_emb.indices.tolist(),  # Term indices
                        values=sparse_emb.values.tolist()  # Term weights
                    )
                },
                payload={
                    "parent_id": parent_id,
                    "text": text,
                    "title":doc_title
                }
            )
        )
    return result_array

async def insert_into_db(async_client, small_data, large_data):
    await async_client.upsert(
        collection_name="wiki_small_chunks",
        points=small_data
    )
    await async_client.upsert(
        collection_name="wiki_large_chunks",
        points=large_data
    )
def rag_parallel(config_yaml, qdrant_client: QdrantClient):
    small_collection_name = config_yaml['db']['collections']['collection_small']['name']
    large_collection_name = config_yaml['db']['collections']['collection_large']['name']
    # Initialize collections
    create_collection_with_embeddings(qdrant_client, small_collection_name)
    create_collection_without_embeddings(qdrant_client, large_collection_name)
    # Load dataset in streaming mode
    max_articles = config_yaml["dataset"]["max_articles"]
    dataset = load_dataset(path="wikimedia/wikipedia", name="20231101.en", cache_dir=config_yaml["dataset"]["dir"],
                           split=f"train[:{max_articles}]")
    # Get iterable for the training split
    data_iterable = dataset
    # Configuration
    batch_size = config_yaml['db']['insertion']['batch_size']  # Size of the batch to add to Chroma
    num_processes = os.cpu_count() - 1 if os.cpu_count() > 1 else 1  # Use N-1 cores

    print(f"Using {num_processes} processes for document processing.")

    # islice limits the number of articles processed
    limited_iterable = islice(enumerate(data_iterable), max_articles)
    dense_embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    async_client = get_async_client(QdrantConfig())
    # Using multiprocessing pool to process documents in parallel
    with Pool(processes=num_processes) as pool:
        # map() applies the function to all elements and returns results in order
        results_iterator = pool.imap(process_document, limited_iterable, chunksize=500)  # Use imap for memory efficiency

        current_doc_small_data = []
        current_doc_large_data = []
        doc_count = 0

        for small_data, large_data in tqdm(results_iterator, total=max_articles, desc="Processing Articles"):
            doc_count += 1

            current_doc_small_data.extend(point_construct_from_dict(small_data, dense_embedding_model=dense_embedding_model, sparse_model=sparse_model))
            current_doc_large_data.extend(large_data)

            if doc_count % batch_size == 0:
                asyncio.run(insert_into_db(async_client, current_doc_small_data, current_doc_large_data))
                current_doc_small_data = []
                current_doc_large_data = []

    print("RAG data population complete.")
