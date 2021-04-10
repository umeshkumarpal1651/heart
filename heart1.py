#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import os


# In[2]:


os.chdir(r'C:\Users\Mummy\Documents\data source')


# In[3]:


df=pd.read_csv('heart.csv')


# In[5]:


X=df.drop(columns=['DEATH_EVENT'])   #train test split
y=df['DEATH_EVENT']


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[8]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'heart.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[ ]:




