#!/usr/bin/env python
# coding: utf-8

# # Exploretory data analysis (EDA)

# For the first step, we look at the arXiv dataset to find the basic trends, and to clean the dataset to make it ready for the machine learning algorithm.

# Let's import necessary libraries.

# In[105]:


import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from sklearn.preprocessing import MultiLabelBinarizer
import altair as alt


# In[106]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[107]:


DATA_PATH = '../input/archive/arxiv-metadata-oai-snapshot.json'


# This dataset is huge (It has 1.9M+ rows). To use only limited memory, we open just the first 20000 rows, and save that on the disk.

# In[108]:


df = pd.read_json(DATA_PATH, lines=True, nrows=20000)


# In[109]:


df.head()


# In[110]:


df.to_csv('../input/arxiv_20krows.csv', index=False)


# In[111]:


df_20k = pd.read_csv('../input/arxiv_20krows.csv')


# In[112]:


print("Number of rows in data =",df_20k.shape[0])
print("Number of columns in data =",df_20k.shape[1])


# In[113]:


## check for Null values
df_20k.isnull().sum()


# Many rows of `comments`, `journal-ref`, `doi`, `report-no` and `license` columns are empty. But we need in our project is the `title`, `abstract` and `categories` columns, there are no null values in those columns.

# Let's look at a sample `abstract`:

# In[9]:


df_20k['abstract'][0]


# The category of articles are in the `categories` column, and they are seprated with a space character. So to get each category for each sample in our dataset, we need to split the values in the `categories` column by space character:

# In[115]:


df_20k['cats_split'] = df_20k['categories'].str.split()


# How many categories exist in the dataset?

# In[118]:


cats = df_20k['cats_split'].sum()
unique_cats = set(cats)
len(unique_cats)


# 144! That's a lot of categories! Let's look at the distribution of them:

# In[119]:


all_cats = nltk.FreqDist(cats) 
all_cats_df = pd.DataFrame({'categories': list(all_cats.keys()), 
                              'Count': list(all_cats.values())})


# In[120]:


# source: https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/
most_freq_cats = all_cats_df.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(10,12)) 
ax = sns.barplot(data=most_freq_cats, x= "Count", y = "categories")
plt.show()


# So the most frquent categories are **astro-ph** (astro-physics), **hep-th** (high-energy physics - theory), and **hep-ph** (high-energy physics - phenomenology). The complete list of abbreviations exits at https://arxiv.org/category_taxonomy.

# Now we are interested in distribution of the number of categories per article.

# In[18]:


num_cats = [len(df_20k['cats_split'][row]) for row in range(len(df_20k['cats_split']))]


# In[19]:


np.mean(num_cats)


# In[20]:


plt.plot(num_cats)
plt.xlabel('document Id')
plt.ylabel('number of categories')


# So the maximum number of categories for an article is 10, and on average each article is tagged by 1.5 categories.

# In[122]:


plt.hist(num_cats)
plt.xlabel('number of categories')
plt.ylabel('count')


# What about the abstracts? Let's see.

# In[22]:


df_20k['split_abstract'] = df_20k['abstract'].str.split()


# In[24]:


len_of_abstracts = df_20k['split_abstract'].str.len()


# In[25]:


len_of_abstracts.hist(bins=50)


# In[26]:


max(len_of_abstracts)


# Therefore most articles have an abstract with a length of around 100 words, and the maximum number of words for an abstract is 342 words.

# ## Preparation of the dataset for the classification model

# In order to have less number of labels in our dataset, we decide to only consider the general category of each article, and remove the subcategory labels. In the dataset, the subcategories are specified with a "." notation.

# For instance, 'cs.AI' means this article belongs to AI subcategory of cs (computer science) field.

# Let's prepare the general categoty column for the classification model:

# In[124]:


df_20k['general_category'] = df_20k['cats_split'].apply(lambda x:[a.split('.')[0] for a in x])


# In[125]:


df_20k.head()


# In[31]:


unique_gen_cat = df_20k['general_category'].map(set)
gen_cats = df_20k['general_category'].sum()
unique_gen_cats = set(gen_cats)
len(unique_gen_cats)


# Now the number of categories is reduced to 19 general category. Let's see how they are distibuted:

# In[126]:


all_gen_cats = nltk.FreqDist(gen_cats) 
all_gen_cats_df = pd.DataFrame({'categories': list(all_gen_cats.keys()), 
                              'Count': list(all_gen_cats.values())})


# In[127]:


most_freq_gen_cats = all_gen_cats_df.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(10,12)) 
ax = sns.barplot(data=most_freq_gen_cats, x= "Count", y = "categories")
plt.show()


# In[35]:


all_gen_cats_df


# Now there are more sample for each category. But still some categoties have just a few hundared samples. The **econ** category has only three samples in the dataset which is not enough. So we decide to load more data. In the next section we use Dask, which is libarary to efficiently load large datasets and scales well for large data machine learning tasks

# Let's save the processed dataset for now.

# In[128]:


df_20k.to_csv('../input/arxiv_20krows_train.csv', index=False)


# ## loading data using Dask

# In[37]:


# many parts are inspired by: https://www.kaggle.com/kobakhit/eda-and-multi-label-classification-for-arxiv
import dask.bag as db
import json

docs = db.read_text('../input/archive/arxiv-metadata-oai-snapshot.json').map(json.loads)


# In[38]:


docs


# How many articles are in the dataset?

# In[39]:


docs.count().compute()


# Let's look at the first sample:

# In[40]:


docs.take(1)


# In[41]:


# Submissions by datetime
get_latest_version = lambda x: x['versions'][-1]['created']


# Now we select only the articles submitted after 2021.

# In[42]:


# get only necessary fields
trim = lambda x: {'id': x['id'],
                  'title': x['title'],
                  'category':x['categories'].split(' '),
                  'abstract':x['abstract']}
# filter for papers published on or after 2020-01-01
columns = ['id','category','abstract']
docs_df = (docs
             .filter(lambda x: int(get_latest_version(x).split(' ')[3]) > 2020)
             .map(trim)
             .compute())

# convert to pandas
docs_df = pd.DataFrame(docs_df)

# add general category. we are going to use as our target variable
docs_df['general_category'] = docs_df.category.apply(lambda x:[a.split('.')[0] for a in x])


# In[60]:


docs_df


# About 200k articles are selected. Now let's plot the number of samples per category.

# In[84]:


# plot paper distribution by category
source = pd.DataFrame(sample_df.iloc[:,2:].apply(sum)).reset_index().rename(columns = {0:'count'})
ptotalText = alt.Text('PercentOfTotal:Q', format = '.2%')

chart = alt.Chart(source).transform_joinaggregate(
    TotalPapers='sum(count)',
).transform_calculate(
    PercentOfTotal="datum.count / datum.TotalPapers"
).mark_bar().encode(
    x = 'index',
    y = 'count',
    tooltip = ['index','count', ptotalText]
).properties(
    title='arXiv Papers by Category, after 2021-01-01',
    width = 800
)

# add percentage labels
chart = chart + chart.mark_text(
    align='center',
    baseline='middle',
    dx= 3,  # Nudges text to right so it doesn't appear on top of the bar,
    dy = -5
).encode(
    text = ptotalText
) 

chart


# So we have at least 500 samples for each category in this dataset. We save the dataset for modelling in the next step.

# In[77]:


sample_df.to_csv('../input/sample_df_2021.csv', index=False)

