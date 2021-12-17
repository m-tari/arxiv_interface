from sentence_transformers import SentenceTransformer, util
import pandas as pd
import pickle
import os, io
import s3fs
import torch
# custom libraries
from . import config_set

# create connection object for AWS
fs = s3fs.S3FileSystem(anon=False)

# unpack a binary files saved on a gpu machine in a cpu machine
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

# load binary file using CPU_Unpickler
def read_file(filename):
	with fs.open(filename, 'rb') as file_bin:
		return CPU_Unpickler(file_bin).load()

corpus_embeddings = read_file("arxivinterface/sample_df_2021_embeddings.pkl")

# load csv dataset from AWS
def read_dataset(filename):
	with fs.open(filename, 'rb') as file_bin:
		return pd.read_csv(file_bin)

df = read_dataset("arxivinterface/sample_df_2021.csv")

# load model from SentenceTransformer
def load_model(model_name):
	return SentenceTransformer(model_name)

model = load_model('multi-qa-MiniLM-L6-cos-v1')

def search_papers(title, abstract):
	query_embedding = model.encode(title+'[SEP]'+abstract, convert_to_tensor=True)

	search_hits = util.semantic_search(query_embedding, corpus_embeddings)
	search_hits = search_hits[0]  #Get the hits for the first query

	indices = []
	for hit in search_hits:
		indices.append(hit['corpus_id'])
	related_papers = df.iloc[indices]
	
	return	related_papers
