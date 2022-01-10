import streamlit as st
import pandas as pd
import pickle
import os, io
import s3fs
import torch
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from src import semantic_search, summarizer

#############################
# loading models and datasets
#############################

# create connection object
fs = s3fs.S3FileSystem(anon=False)

# unpack a binary file that has been saved on a gpu machine, on a cpu machine
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

# load binary a file using CPU_Unpickler
@st.cache(ttl=600)
def read_file(filename):
	with fs.open(filename, 'rb') as file_bin:
		return CPU_Unpickler(file_bin).load()

# load a csv dataset from AWS
@st.cache(ttl=600)
def read_dataset(filename):
	with fs.open(filename, 'rb') as file_bin:
		return pd.read_csv(file_bin)

# load semantic search model from SentenceTransformer
@st.cache(ttl=600, allow_output_mutation=True)
def load_semantic_search_model(model_name):
	return SentenceTransformer(model_name)

# load title generation model
@st.cache(ttl=600, allow_output_mutation=True)
def load_summarizer_model(model_name):
	return TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

# load tokenizer for title generation model
@st.cache(ttl=600, allow_output_mutation=True)
def load_tokenizer(tokernizer_name):
	return AutoTokenizer.from_pretrained(tokernizer_name)

#############################
# main functionalities of the app
#############################

def get_category(input_abstract):
	st.session_state.category = 'Computer Science'
	return st.session_state.category

def suggest_title(input_abstract, min_length, max_length):
	tokenizer = load_tokenizer("t5-small")
	summarizer_model = load_summarizer_model('mohammadtari/arxivinterface')
	st.session_state.title  = summarizer.title_generator(input_abstract, min_length, max_length, summarizer_model, tokenizer)
	return st.session_state.title

def suggest_articles(input_abstract):
	df = read_dataset("arxivinterface/sample_df_2021.csv")
	semantic_search_model = load_semantic_search_model('multi-qa-MiniLM-L6-cos-v1')
	corpus_embeddings = read_file("arxivinterface/sample_df_2021_embeddings.pkl")
	st.session_state.articles = semantic_search.search_papers(input_abstract, df, semantic_search_model, corpus_embeddings)
	return st.session_state.articles	


def main(): 

	if 'category' not in st.session_state:
		st.session_state.category = ""

	if 'title' not in st.session_state:
		st.session_state.title = ""

	if 'articles' not in st.session_state:
		st.session_state.articles = ""

	st.write(
	'''
	# Viresa: an AI-powered virtual assistant for scientists

	'''
	)
	st.write(
	'''
	A common task for scientists is to extract knowledge from scientific articles: 
	What area of research does it belong to? What is the best one-line summary of the context?
	What are the relevant articles to the new information? We built a tool to answer these questions!

	To find the answer to these questions, we need the abstract of the article:

	'''
	)
	# image = Image.open("./images/tight@1920x_transparent.png") # ArXive image
	# st.sidebar.image(image, use_column_width=True)
	st.sidebar.markdown(
		"Check out the package on [Github](https://github.com/m-tari/arxiv_interface)!"
	)

	models = {
		"n_bayes_on_arXiv": "Simple Naive Bayes model trained on ArXiv artilces",
		"distilbert-base-uncased-finetuned-XXX": "DistilBERT model finetuned on X task. Predicts category of STEM articles."
	}
	model_name = st.sidebar.selectbox(
		"Choose a classification model", list(models.keys())
	)

	input_abstract = st.text_area('Abstract to analyze:', 
		height=500,
		max_chars=2000, 
		value="Common sense has always been of interest in AI, but has rarely taken center stage. "
		"Despite its mention in one of John McCarthy's earliest papers and years of work by dedicated "
		"researchers, arguably no AI system with a serious amount of general common sense has ever emerged. "
		"Why is that? What's missing? Examples of AI systems' failures of common sense abound, and "
		"they point to AI's frequent focus on expertise as the cause. Those attempting to break "
		"the brittleness barrier, even in the context of modern deep learning, have tended to invest "
		"their energy in large numbers of small bits of commonsense knowledge. But all the commonsense "
		"knowledge fragments in the world don't add up to a system that actually demonstrates common sense "
		"in a human-like way. We advocate examining common sense from a broader perspective than in the past. "
		"Common sense is more complex than it has been taken to be and is worthy of its own scientific exploration. "
		)

	# categorize the abstract
	st.subheader("Find the categories!")
	get_categories_but = st.button("Categorize the abstract", on_click=get_category, args=(input_abstract, ))
	st.write('Categories:', st.session_state.category)	

	# suggest a title for the abstract
	st.subheader("Suggest a title!")
	min_length, max_length = st.slider("Choose minimum and maximum length of the title!", value=[10,50], min_value=1, max_value=100)
	get_title_but = st.button("Suggest a title for the abstract", on_click=suggest_title, args=(input_abstract, min_length, max_length))
	st.write('Title:', st.session_state.title)

	# show most related articles in the dataset
	st.subheader("Show similar articles!")	
	get_similar_articles_but = st.button("Suggest 10 most similar articles", on_click=suggest_articles, args=(input_abstract, ))
	st.write('Most related articles:')
	
	if get_similar_articles_but:
		
		article_number = 1
		for index, article in st.session_state.articles.iterrows():

			link = f"https://arxiv.org/abs/{article['id']}"
			st.write(article_number, "**Title**: ",article['title'])
			st.markdown("**Abstract**:")
			st.write(article['abstract'][:600], "...")
			st.markdown(f"**Link**: [{link}]({link})")	
			st.markdown("***")		
			article_number += 1

	# TODO: add lime for explainability, etc		

if __name__=='__main__': 
	main()	
