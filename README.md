### status: under development

# Viresa: an AI-powered virtual assistant for scientists
Web-app : https://share.streamlit.io/m-tari/arxiv_interface [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/m-tari/arxiv_interface)

*(note: article classification part is not implemented in the web-app yet.)*
## Overview
A common task for scientists is to extract knowledge from scientific articles: What area of research does it belong to? What is the best one-line summary of the context? What are the relevant articles to the new information? We built a tool to answer these questions!

## Background and Motivation
ArXiv is a collaboratively funded, community-supported resource founded by Paul Ginsparg in 1991 and maintained and operated by Cornell University [1]. They promote open scientific collaboration and progress by providing tools and contents for scientists all over the world. Our hope is that by using the rich dataset of scholary articles and powerfull machine learning techniques we discover insights about scientific works, and we try to build simple tools to explore the dataset for trend analysis, paper recommender engines, category prediction, and more.

## Goals
To build a multi-label multiclass classification model capable of automatic tagging of the summaries of articles, generating titles for the summaries, and reccommending similar articles to the user.

## Datasets
Dataset used in this project is the [metadata file of the arXiv dataset](https://www.kaggle.com/Cornell-University/arxiv) provided for the Kaggle classification and title genertion challenges. This dataset contains 1.7M+ scholarly papers across STEM, with relevant features such as article titles, authors, categories, abstracts, and more.

## Milestones

- ~Build a classical machine learning classifer for automatic tagging of articles~
- ~Use transformers for title generation~
- ~Build a recommender system for similar articles~
- Apply RNNs for title generation and compare the performance with transformers
- Try different word embeddings (using Word2Vec, GloVe, etc)
## References
[1] https://arxiv.org/
