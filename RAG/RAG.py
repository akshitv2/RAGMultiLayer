import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm

from DB.Chroma import create_collection, create_collectionx


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


def rag(config_yaml, chroma_client):
    dataset = load_dataset(path="wikimedia/wikipedia", name="20231101.en", cache_dir=config_yaml["dataset"]["dir"],
                           split="train[:500]")
    max_articles = min(len(dataset), config_yaml["dataset"]["max_articles"])
    dataset = dataset.select(range(max_articles))
    print(f"Number of articles: {max_articles}")

    COLLECTION_SMALL = "wiki_small_chunks"
    COLLECTION_LARGE = "wiki_large_chunks"

    # Initialize the Embedding Model
    documents_to_process = [doc['text'] for doc in dataset]

    large_splitter = get_splitter(use_large=True)
    small_splitter = get_splitter(use_large=False)

    small_collection = create_collectionx(chroma_client, COLLECTION_SMALL)
    large_collection = create_collectionx(chroma_client, COLLECTION_LARGE)

    tqdm_iter =tqdm(enumerate(documents_to_process), total=len(documents_to_process))

    # Process each document
    for doc_index, document in tqdm_iter:
        large_chunks = large_splitter.create_documents([document])
        len_large_chunks = len(large_chunks)
        for l_chunk_id, l_chunk in enumerate(large_chunks):
            l_chunk.metadata["doc_index"] = doc_index
            parent_id = str(uuid.uuid4())
            large_collection.add(
                ids=[parent_id],
                documents=[l_chunk.page_content],
                metadatas=[l_chunk.metadata],
            )

            small_chunks = small_splitter.create_documents([l_chunk.page_content])
            len_small_chunks = len(small_chunks)
            for s_chunk_id, s_chunk in enumerate(small_chunks):
                tqdm_iter.set_postfix(Status=f"L :{l_chunk_id}/{len_large_chunks} S :{s_chunk_id}/{len_small_chunks}")
                child_id = str(uuid.uuid4())
                s_chunk.metadata["parent_id"] = parent_id
                small_collection.add(
                    ids=[child_id],
                    documents=[s_chunk.page_content],
                    metadatas=[s_chunk.metadata],
                    # embeddings=small_chunks_data["embeddings"]  # Insert the pre-computed embeddings
                )


