#!/usr/bin/env python
# coding: utf-8

# ### import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### now read the data

# In[2]:


df = pd.read_csv(r'F:\Study\Machine Learning\datasets\house-prices-advanced-regression-techniques\train.csv')
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.describe


# In[8]:


df.describe()


# In[9]:


pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)


# In[10]:


df.head()


# In[11]:


df.tail(6)


# In[12]:


df.isnull().sum()


# In[13]:


df.info()


# In[14]:


plt.figure(figsize=(25,25))
sns.heatmap(df.isnull())


# In[15]:


null_var = df.isnull().sum()/df.shape[0]* 100
null_var


# In[16]:


drop_column = null_var[null_var>20].keys()
drop_column


# In[19]:


df2_drop_clm = df.drop(columns=drop_column)


# In[20]:


df2_drop_clm.shape


# In[21]:


df.shape


# In[24]:


sns.heatmap(df2_drop_clm.isnull())


# In[25]:


plt.figure(figsize=(25,25))
sns.heatmap(df2_drop_clm.isnull())


# In[26]:


df3_drop_row = df2_drop_clm.dropna()


# In[27]:


df3_drop_row.shape


# In[29]:


df2_drop_clm.shape

df.shape


# In[30]:


df3_drop_row.select_dtypes(include=['int64', 'float64']).columns


# In[32]:


sns.displot(df['MSSubClass'])


# In[33]:


sns.distplot(df['MSSubClass'])


# In[34]:


sns.distplot(df3_drop_row['MSSubClass'])

