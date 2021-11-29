#!/usr/bin/env python
# coding: utf-8

# ## Automatic tagging

# In[95]:


sample_df = pd.read_csv('../input/sample_df_2021.csv', converters={'general_category': pd.eval})


# In[97]:


sample_df['general_category'][0]


# In[98]:


dir_path = os.path.dirname(os.getcwd())
SRC_PATH = os.path.join(dir_path, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


# In[99]:


import train
from train import train_model


# In[100]:


import importlib
importlib.reload(train)


# In[101]:


train_model(3, 'n_bayes') 

