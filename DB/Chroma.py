import os
import shutil

import chromadb


def run_db():
    db_path = os.path.join(os.getcwd(), "vector_db")
    client = chromadb.PersistentClient(path=db_path)
    return client

def delete_existing_collections():
    db_path = os.path.join(os.getcwd(), "vector_db")

    for filename in os.listdir(db_path):
        file_path = os.path.join(db_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

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

