import streamlit as st

from src import config_set, semantic_search


def get_category(txt):
	st.session_state.category = 'Computer Science'
	return st.session_state.category

def suggest_title(txt):
	st.session_state.title  = "A thought-provoking title"
	return st.session_state.title

# @st.cache(ttl=600)
def suggest_articles(title, input_abstract):
	st.session_state.articles = semantic_search.search_papers('title', input_abstract)
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

	input_abstract = st.text_area('Abstract to analyze:', 
		height=400,
		max_chars=850, 
		value="We derive a new fully implicit formulation for the ..."
		)

	get_categories_but = st.button("Categorize the abstract", on_click=get_category, args=(input_abstract, ))
	st.write('Categories:', st.session_state.category)	

	get_title_but = st.button("Suggest a title for the abstract", on_click=suggest_title, args=(input_abstract, ))
	st.write('Title:', st.session_state.title)

	get_similar_articles_but = st.button("Suggest 10 most similar articles", on_click=suggest_articles, args=('title', input_abstract, ))
	st.write('Most related articles:')
	
	# Show most related articles in the dataset
	if get_similar_articles_but:
		
		article_number = 1
		for index, article in st.session_state.articles.iterrows():
			st.write(article_number, "**Title**: ",article['title'])
			st.markdown("**Abstract**:")
			st.write(article['abstract'][:600], "...")
			link = "https://arxiv.org/abs/"+str(article['id'])
			st.markdown("**Link**: "+"["+link+"]"+"("+link+")")	
			st.markdown("***")		
			article_number += 1

	# TODO: add lime for explainability, etc		

if __name__=='__main__': 
	main()	