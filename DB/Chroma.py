import os

import chromadb


def run_db():
    db_path = os.path.join(os.getcwd(), "vector_db")
    client = chromadb.PersistentClient(path=db_path)
    return client


def create_collection(client, collection_name, embedding_function):
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function  # Crucial for search!
    )
    return collection


def create_collectionx(client, collection_name):
    collection = client.get_or_create_collection(
        name=collection_name
    )
    return collection

def get_collection(client, collection_name):
    collection = client.get_collection(name = collection_name)
    return collection

