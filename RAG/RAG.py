import re
import uuid

from importlib_metadata import metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import spacy

from DB.Chroma import create_collection, create_collectionx

from datasets import load_dataset
from itertools import islice


def fetch_doc_by_index(dataset_path, name, split, index):
    """
    Fetch one document from a Hugging Face dataset without loading it fully into memory.
    Uses streaming mode for lazy iteration.
    """
    ds = load_dataset(dataset_path, name=name, split=split, streaming=True)
    doc = next(islice(ds, index, index + 1))
    return doc


def get_spacy():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading en_core_web_sm model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp


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


def clean_document_spacy(text: str) -> str:
    nlp = get_spacy()
    cleaned_text = re.sub(r'\\n|\\t|\\r', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    doc = nlp(cleaned_text)
    cleaned_tokens = []
    for token in doc:
        if not (token.is_punct or token.is_space):
            cleaned_tokens.append(token.text)
    return " ".join(cleaned_tokens)


def rag(config_yaml, chroma_client):
    # dataset = load_dataset(path="wikimedia/wikipedia", name="20231101.en", cache_dir=config_yaml["dataset"]["dir"],
    #                        split="train[:5000]")
    dataset = load_dataset(path="wikimedia/wikipedia", name="20231101.en", cache_dir=config_yaml["dataset"]["dir"],
                           streaming=True)

    large_splitter = get_splitter(use_large=True)
    small_splitter = get_splitter(use_large=False)
    small_collection = create_collectionx(chroma_client, config_yaml['db']['collections']['collection_small']['name'])
    large_collection = create_collectionx(chroma_client, config_yaml['db']['collections']['collection_large']['name'])

    batch_size = 25
    current_doc_small_data = {"ids": [], "documents": [], "metadatas": []}
    current_doc_large_data = {"ids": [], "documents": [], "metadatas": []}

    max_articles = 5000

    # tqdm_iter = tqdm(enumerate(dataset['train']), total=max_articles)
    for doc_index, document in enumerate(dataset['train']):
        print(doc_index, end="")
        if doc_index > max_articles:
            break
        if doc_index % batch_size == 1:
            small_collection.add(
                ids=current_doc_small_data['ids'],
                documents=current_doc_small_data['documents'],
                metadatas=current_doc_small_data['metadatas'])

            large_collection.add(
                ids=current_doc_large_data['ids'],
                documents=current_doc_large_data['documents'],
                metadatas=current_doc_large_data['metadatas'])
            del current_doc_small_data
            del current_doc_large_data

            current_doc_small_data = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
            current_doc_large_data = {"ids": [], "documents": [], "metadatas": []}

        large_chunks = large_splitter.create_documents([document['text']])
        len_large_chunks = len(large_chunks)
        for l_chunk_id, l_chunk in enumerate(large_chunks):
            parent_id = str(uuid.uuid4())
            current_doc_large_data["ids"].append(parent_id)
            current_doc_large_data["documents"].append(l_chunk.page_content)
            current_doc_large_data["metadatas"].append(l_chunk.metadata)

            l_chunk.metadata["doc_index"] = doc_index
            small_chunks = small_splitter.create_documents([l_chunk.page_content])
            len_small_chunks = len(small_chunks)
            for s_chunk_id, s_chunk in enumerate(small_chunks):
                # tqdm_iter.set_postfix(Status=f"L :{l_chunk_id}/{len_large_chunks} S :{s_chunk_id}/{len_small_chunks}")
                child_id = str(uuid.uuid4())
                s_chunk.metadata["parent_id"] = parent_id

                current_doc_small_data["ids"].append(child_id)
                current_doc_small_data["documents"].append(s_chunk.page_content)
                current_doc_small_data["metadatas"].append(s_chunk.metadata)
