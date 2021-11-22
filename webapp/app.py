import streamlit as st
import joblib
import os
# import config

# loading the trained model
model_bin = open('../models/n_bayes_score_0.32.bin', 'rb') 
classifier = joblib.load(model_bin)


def get_category(txt):
	category = 'Computer Science'
	st.session_state.category = category
	return st.session_state.category

def suggest_title(txt):
	title = "A thought-provoking title"
	st.session_state.title = title	
	return st.session_state.title

def main(): 

	if 'category' not in st.session_state:
	    st.session_state.category = ""

	if 'title' not in st.session_state:
	    st.session_state.title = ""

	st.write(
	'''
	# A Web Interface for ArXiv Articles

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
	# model, tokenizer = load_model(model_name)

	txt = st.text_area('Abstract to analyze:', 
		height=400,
		max_chars=850, 
		value="We derive a new fully implicit formulation for the "
		)
	print(txt)

	st.button("Categorize the abstract", on_click=get_category, args=(txt, ))
	st.write('Categories:', st.session_state.category)	

	st.button("Suggest a title for the abstract", on_click=suggest_title, args=(txt, ))
	st.write('Title:', st.session_state.title)

	# TODO: add lime for explainability, etc		

if __name__=='__main__': 
	main()	