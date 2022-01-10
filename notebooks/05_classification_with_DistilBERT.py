#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/m-tari/arxiv_interface/blob/master/notebooks/05_classification_with_DistilBERT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


get_ipython().system('pip install transformers datasets pandas')


# ## Load dataset

# In[ ]:


from datasets import Dataset, load_metric, load_dataset
import pandas as pd


# In[ ]:


sample_df = pd.read_csv("/content/drive/MyDrive/ML/sample_df_2021.csv", nrows=10000, usecols=['abstract', 'general_category'])


# In[ ]:


sample_df


# In[ ]:


sample_df['labels'] = sample_df.general_category.str.replace("[\[\]\']", "",).str.split(pat=", ")


# In[ ]:


sample_df['text'] = sample_df['abstract']


# In[ ]:


sample_df


# In[ ]:


sample_df = sample_df[['text', 'labels']][:10000]


# In[ ]:


sample_df.labels = sample_df.labels.apply(set).apply(list)


# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer

multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(sample_df.labels)
y_train = pd.DataFrame(y, columns=multilabel.classes_)


# In[ ]:


y_train


# In[ ]:


s = pd.Series([enc_label for enc_label in y])


# In[ ]:


s


# In[ ]:


sample_df['enc_labels'] = s


# In[ ]:


sample_df.enc_labels[0]


# In[ ]:


import numpy as np
y_pred = np.array([[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])


# In[ ]:


y_pred.shape


# In[ ]:


multilabel.inverse_transform(y_pred)


# In[ ]:


y


# In[ ]:


sample_dataset = Dataset.from_pandas(sample_df[['text', 'enc_labels']])


# In[ ]:


sample_dataset = sample_dataset.train_test_split(test_size=0.1)


# In[ ]:


sample_dataset


# In[ ]:


sample_dataset['train'][1]


# ## Load DistilBERT model

# In[ ]:


from transformers import AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased"
num_labels = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = (AutoModelForSequenceClassification.from_pretrained(model_name, problem_type="multi_label_classification", num_labels=num_labels).to(device))


# In[ ]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name, problem_type="multi_label_classification")


# ## Tokenizing

# In[ ]:


def tokenize(batch):
  return tokenizer(batch["text"], padding=True, truncation=True)


# In[ ]:


# remove all columns except for enc_labels
cols = sample_dataset["train"].column_names
cols.remove("enc_labels")

sample_dataset_encoded = sample_dataset.map(tokenize, batched=True, batch_size=None, remove_columns=cols)


# In[ ]:


sample_dataset_encoded['train']


# In[ ]:


# cast label IDs to floats
sample_dataset_encoded.set_format("torch")


# In[ ]:


sample_dataset_encoded['train']


# In[ ]:


sample_dataset_encoded = (sample_dataset_encoded
          .map(lambda x : {"float_labels": x["enc_labels"].to(torch.float)})
          .rename_column("float_labels", "labels"))


# In[ ]:


sample_dataset_encoded['train'][0]


# ## Training

# In[ ]:


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(".", num_train_epochs=1)

trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer,
    train_dataset=sample_dataset_encoded["train"],
    eval_dataset=sample_dataset_encoded["test"])


# In[ ]:


trainer.train()


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


pt_save_directory = '/content/drive/MyDrive/ML/'
tokenizer.save_pretrained(pt_save_directory)
model.save_pretrained(pt_save_directory)


# ## Inference

# In[ ]:


# DistilBert uses a max_length of 512 so we cut the article to 512 tokens.

abstract = """
Data scientists and statisticians are often at odds when determining the best approach, machine learning or statistical modeling, 
to solve an analytics challenge. However, machine learning and statistical modeling are more cousins than adversaries on different 
sides of an analysis battleground. Choosing between the two approaches or in some cases using both is based on the problem to be solved 
and outcomes required as well as the data available for use and circumstances of the analysis. Machine learning and statistical modeling 
are complementary, based on similar mathematical principles, but simply using different tools in an overall analytics knowledge base. 
Determining the predominant approach should be based on the problem to be solved as well as empirical evidence, such as size and completeness 
of the data, number of variables, assumptions or lack thereof, and expected outcomes such as predictions or causality. Good analysts and 
data scientists should be well versed in both techniques and their proper application, thereby using the right tool for the right project 
to achieve the desired results. """

actual_category = "cs, stat"  

inputs = tokenizer(abstract, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")


# In[ ]:


outputs = model(**inputs)


# In[ ]:


outputs


# In[ ]:


# Because we can have multiple lables for an abstract and they are NOT mutually exclusive, we pass logits to sigmoid function
probs = torch.sigmoid(outputs.logits).tolist()[0]


# In[ ]:


preds = [1 for prob in probs if prob>0.5 else 0]


# In[ ]:


categories = multilabel.inverse_transform(preds)

print(f'Actual Categories: {actual_category}\n')
print(f'Predicted Categories: {categories}\n')


# In[ ]:


# predicted cs and stat, which are above 0.50 threshold! :)


# ## Error Analysis
