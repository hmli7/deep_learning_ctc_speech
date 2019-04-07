#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


# In[2]:


import paths
import data
import ctc_model
import phoneme_list


# In[18]:


import importlib
reload_packages = [paths, data, ctc_model]
for package in reload_packages:
    importlib.reload(package)
# importlib.reload(data)


# In[4]:


train_loader = data.get_loader("train")
val_loader = data.get_loader("val")


# In[5]:


# for x, y in val_loader:
#     print(model(x))
#     break


# In[6]:


# torch.cat(test)


# In[5]:


model = ctc_model.SpeechModel(phoneme_list.N_PHONEMES,256,1)


# In[6]:


print(model)


# In[7]:


# ctc_model.run(model, train_loader, val_loader)


# In[ ]:





# In[ ]:

model.cuda()
test = ctc_model.run_eval(model, val_loader)


# In[15]:


test[0].size()


# In[ ]:


print(1)


# In[ ]:




