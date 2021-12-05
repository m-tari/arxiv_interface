#!/usr/bin/env python
# coding: utf-8

# # Title Generation for arXiv dataset

# In[ ]:


# Installing libraries in Google Colab:


# In[54]:


pip install transformers datasets rouge_score


# In[ ]:


# STEPS

# Load dataset
# Preprocessing
# fine-tuning

# source: https://github.com/huggingface/notebooks/blob/master/examples/summarization-tf.ipynb


# ### loading the dataset

# In[9]:


from datasets import load_metric, load_dataset


# In[10]:


model_checkpoint = "t5-small"


# In[30]:


dataset = load_dataset('csv', data_files='./input/sample_df_2021.csv', split='train[:10%]')


# In[31]:


dataset[0]


# In[32]:


dataset = dataset.train_test_split()


# In[33]:


dataset


# ### preprocessing

# In[34]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[35]:


tokenizer("Hello, this one sentence!")


# In[36]:


if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""


# In[37]:


max_input_length = 1024
max_target_length = 128


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["abstract"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["title"], max_length=max_target_length, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[38]:


preprocess_function(dataset['train'][:2])


# In[39]:


tokenized_datasets = dataset.map(preprocess_function, batched=True)


# ## Fine-tuning the model

# In[40]:


from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


# In[41]:


batch_size = 8
learning_rate = 2e-5
weight_decay = 0.01
num_train_epochs = 1


# In[42]:


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")


# In[43]:


tokenized_datasets


# In[44]:


train_dataset = tokenized_datasets["train"].to_tf_dataset(
    batch_size=batch_size,
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=True,
    collate_fn=data_collator,
)
validation_dataset = tokenized_datasets["test"].to_tf_dataset(
    batch_size=8,
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=False,
    collate_fn=data_collator,
)


# In[45]:


from transformers import AdamWeightDecay
import tensorflow as tf

optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)
model.compile(optimizer=optimizer)


# In[47]:


model.fit(train_dataset, validation_data=validation_dataset, epochs=4)


# ## Evaluate the performance of the model

# In[48]:


import numpy as np

decoded_predictions = []
decoded_labels = []
for batch in validation_dataset:
    labels = batch["labels"]
    predictions = model.predict_on_batch(batch)["logits"]
    predicted_tokens = np.argmax(predictions, axis=-1)
    decoded_predictions.extend(
        tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
    )
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))


# In[50]:


nltk.download('punkt')


# In[55]:


import nltk
import numpy as np

metric = load_metric("rouge")

# Rouge expects a newline after each sentence
decoded_predictions = [
    "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_predictions
]
decoded_labels = [
    "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
]

result = metric.compute(
    predictions=decoded_predictions, references=decoded_labels, use_stemmer=True
)
# Extract a few results
result = {key: value.mid.fmeasure for key, value in result.items()}

# Add mean generated length
prediction_lens = [
    np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
]
result["gen_len"] = np.mean(prediction_lens)

print({k: round(v, 4) for k, v in result.items()})


# ## Inference

# In[65]:


dataset['test']['abstract'][0]


# In[66]:


dataset['test']['title'][0]


# In[72]:


random_num = 0

actual_title = dataset['test']['title'][random_num]
actual_abstract = dataset['test']['abstract'][random_num]


# T5 uses a max_length of 512 so we cut the article to 512 tokens.
inputs = tokenizer("summarize: " + actual_abstract, return_tensors="tf", max_length=512)
outputs = model.generate(
    inputs["input_ids"], max_length=20, min_length=5, length_penalty=2.0, num_beams=4, early_stopping=True
)

print(f'Actual Title: {actual_title}\n')
print(f'Predicted Title: {tokenizer.decode(outputs[0])}\n')
print(f'Actual Abstract: {actual_abstract}')

# print(tokenizer.decode(outputs[0]))


# In[ ]:




