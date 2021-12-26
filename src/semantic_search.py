from sentence_transformers import util
import pandas as pd

# custom libraries
from . import config_set


def search_papers(abstract, df, model, corpus_embeddings):
	query_embedding = model.encode(abstract, convert_to_tensor=True)

	search_hits = util.semantic_search(query_embedding, corpus_embeddings)
	search_hits = search_hits[0]  #Get the hits for the first query

	indices = []
	for hit in search_hits:
		indices.append(hit['corpus_id'])
	related_papers = df.iloc[indices]
	
	return	related_papers
