#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/m-tari/arxiv_interface/blob/master/notebooks/04_semantic_search_publications.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Semantic Search in Publications
# 
# This notebook demonstrates how [sentence-transformers](https://www.sbert.net) library can be used to find similar publications ([source](https://colab.research.google.com/drive/12cn5Oo0v3HfQQ8Tv6-ukgxXSmT3zl35A?usp=sharing)).
# 
# As corpus, we use 100k articles from the arXiv dataset that are published after 2021.
# 

# In[2]:


get_ipython().system('pip install sentence-transformers')


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


import json
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd

sample_df_2021 = pd.read_csv('/content/drive/MyDrive/ML/sample_df_2021.csv')

print(len(sample_df_2021), "papers loaded")


# In[5]:


sample_df_2021.head()


# In[6]:


sample_df_2021_papers = sample_df_2021.loc[:, ['title', 'abstract']]


# In[12]:


#We then load the model with SentenceTransformers
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


# In[13]:


#To encode the papers, we must combine the title and the abstracts to a single string
paper_texts_concat = sample_df_2021_papers['title'] + '[SEP]' + sample_df_2021_papers['abstract']


# In[14]:


paper_texts_concat.to_list()[:10]


# In[15]:


paper_texts = paper_texts_concat.to_list()

#Compute embeddings for all papers
corpus_embeddings = model.encode(paper_texts, convert_to_tensor=True)


# In[17]:


import pickle
#Saving corpus embeddings
with open('/content/drive/MyDrive/ML/sample_df_2021_embeddings.pkl', "wb") as fOut:
    pickle.dump(corpus_embeddings, fOut, protocol=pickle.HIGHEST_PROTOCOL)


# In[18]:


sample_df_2021_papers.iloc[0]['title']


# In[19]:


#We define a function, given title & abstract, searches our corpus for relevant (similar) papers
def search_papers(title, abstract):
  query_embedding = model.encode(title+'[SEP]'+abstract, convert_to_tensor=True)

  search_hits = util.semantic_search(query_embedding, corpus_embeddings)
  search_hits = search_hits[0]  #Get the hits for the first query

  print("Paper:", title)
  print("Most similar papers:")
  for hit in search_hits:
    related_paper = sample_df_2021_papers.iloc[hit['corpus_id']]
    print("{:.2f}\t{}".format(hit['score'], related_paper['title']))


# ## Search

# In[22]:


sample_title = '''
 Holomorphy of normalized intertwining operators for certain induced representations I: a toy example 
'''
sample_abstract = '''
The theory of intertwining operators plays an important role in the development of the
Langlands program. This, in some sense, is a very sophisticated theory, but the basic question of
its singularity, in general, is quite unknown. Motivated by its deep connection with the longstand-
ing pursuit of constructing automorphic L-functions via the method of integral representations,
we prove the holomorphy of normalized local intertwining operators, normalized in the sense of
Casselman–Shahidi, for a family of induced representations of quasi-split classical groups as an
exercise. Our argument is the outcome of an observation of an intrinsic non-symmetry property
of normalization factors appearing in different reduced decompositions of intertwining operators.
Such an approach bears the potential to work in general.
'''
search_papers(title=sample_title, abstract=sample_abstract)


# In[ ]:




