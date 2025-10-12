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

from DB.qDrant import create_collection_with_embeddings
class RagQuery:

    def __init__(self, config:dict, ):
