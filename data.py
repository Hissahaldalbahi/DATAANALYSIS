#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pyxlsb
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[3]:


dataframe = pd.read_excel("C:/Users/onlyh/Desktop/stc TV Data Set_T1.xlsb",sheet_name="Final_Dataset")


# In[4]:


dataframe.shape


# In[5]:


dataframe.head()


# In[6]:


dataframe = dataframe.drop(columns=['Column1'])


# In[7]:


dataframe['program_name'] = dataframe['program_name'].str.strip()


# In[8]:


dataframe.head()


# In[9]:


dataframe['date_'] = pd.to_datetime(dataframe['date_'], unit='d', origin='30/12/1899')


# In[10]:


dataframe[['duration_seconds', 'season','episode','series_title','hd']] = dataframe[['duration_seconds', 'season','episode','series_title','hd']].apply(pd.to_numeric)


# In[11]:


dataframe[['user_id_maped', 'program_name','program_class','program_desc','program_genre','original_name']] = dataframe[['user_id_maped', 'program_name','program_class','program_desc','program_genre','original_name']].astype(str)


# In[12]:


dataframe.describe()


# In[13]:


dataframe.isnull().any()


# In[14]:


df=dataframe.copy()


# In[15]:


grouped1=df.copy()
grouped1.loc[grouped1['program_class'] == 'SERIES/EPISODES', 'program_name'] = grouped1['program_name']+'_SE'+grouped1['season'].astype(str)+'_EP'+grouped1['episode'].astype(str)
grouped1 = grouped1.groupby(['program_name','program_class']).agg({'user_id_maped': [('co1', 'nunique'),('co2', 'count')],      'duration_seconds': [('co3', 'sum')] }).reset_index()
grouped1.columns = ['program_name','program_class','No of Users who Watched', 'No of watches', 'Total watch time in seconds']
grouped1['Total watch time in hours']=grouped1['Total watch time in seconds']/3600
grouped1 = grouped1.drop(columns=['Total watch time in seconds'])
grouped1 = grouped1.sort_values(by=['Total watch time in hours', 'No of watches','No of Users who Watched'], ascending=False).reset_index(drop=True)


# In[16]:


grouped1.head(30)


# In[17]:


fig = px.pie(grouped1.head(10), values='Total watch time in hours', names='program_name',             hover_data=['program_class'],title='top 10 programs in total watch time in hours')
fig.show()


# In[18]:


grouped=df.copy()
grouped = grouped.groupby('program_class').agg({'user_id_maped': [('co1', 'nunique'),('co2', 'count')],      'duration_seconds': [('co3', 'sum')] }).reset_index()
grouped.columns = ['program_class','No of Users who Watched', 'No of watches', 'Total watch time in seconds']
grouped['Total watch time in houres']=grouped['Total watch time in seconds']/3600
grouped = grouped.drop(columns=['Total watch time in seconds'])
grouped = grouped.sort_values(by=['Total watch time in houres', 'No of watches','No of Users who Watched'], ascending=False).reset_index(drop=True)


# In[19]:


grouped.head()


# In[20]:


fig = px.pie(grouped, values='Total watch time in houres', names='program_class',             hover_data=['program_class'],title='Total duration spent by program_class')
fig2 = px.pie(grouped, values='No of Users who Watched', names='program_class',             hover_data=['program_class'],title='Total Users watching by program_class')

fig.update_traces(sort=False)
fig2.update_traces(sort=False)
fig.show()
fig2.show()


# In[21]:


grouped2=df.copy()


# In[22]:


grouped2 = grouped2.groupby('hd').agg({'user_id_maped': [('co1', 'nunique'),('co2', 'count')],      'duration_seconds': [('co3', 'sum')] }).reset_index()
grouped2.columns = ['HD or not','No of Users who Watched', 'No of watches', 'Total watch time in seconds']
grouped2['Total watch time in hours']=grouped2['Total watch time in seconds']/3600
grouped2 = grouped2.drop(columns=['Total watch time in seconds'])
grouped2 = grouped2.sort_values(by=['Total watch time in hours', 'No of watches','No of Users who Watched'], ascending=False).reset_index(drop=True)


# In[23]:


grouped2.head()


# In[24]:


grouped2['HD or not'] = grouped2['HD or not'].map({1: 'HD', 0: 'SD'})


# In[25]:


fig = px.pie(grouped2, values='Total watch time in hours', names='HD or not',             hover_data=['HD or not'],title='Total duration watching HD compared to SD')
fig2 = px.pie(grouped2, values='No of Users who Watched', names='HD or not',             hover_data=['HD or not'],title='Total number of users watched in HD compared to SD')

fig.update_traces(sort=False)
fig2.update_traces(sort=False)
fig.show()
fig2.show()


# In[ ]:




