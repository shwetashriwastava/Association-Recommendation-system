#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import warnings


# In[4]:


book = pd.read_csv('D:\\data science\\assignments\\ass-10 Recommendation system\\book.csv')
book


# In[5]:


book.head()


# In[6]:


book.corr() #EDA


# In[7]:


book.describe()


# In[8]:


book.isnull().sum()


# In[9]:


book1=book.drop('Unnamed: 0',axis=1)
book1


# In[11]:


book2=book1.dropna()     #Dropping NA value
book2


# In[12]:


book3 = book2.rename({'User.ID':'userid','Book.Title':'booktitle','Book.Rating':'bookrating'},axis = 1)
book3


# In[13]:


book3[0:5]


# In[14]:


len(book3.userid.unique())
#array_user = book1['userid'].unique()


# In[15]:


len(book3.booktitle.unique())


# In[16]:


book_data1 = book3.pivot_table(index = 'userid',
                        columns = 'booktitle',
                        values = 'bookrating').reset_index(drop = True)


# In[17]:


book_data1.head()


# In[19]:


book_data1.index = book3.userid.unique()


# In[20]:


book_data1.fillna(0, inplace = True)


# In[21]:


book_data1.head()


# In[22]:


warnings.filterwarnings("ignore")

user = 1 - pairwise_distances(book_data1.values, metric = 'cosine')


# In[24]:


user_data = pd.DataFrame(user)


# In[25]:


user_data.iloc[0:5,0:5]                       


# In[27]:


user_data.index = book3.userid.unique()
user_data.columns = book3.userid.unique()


# In[28]:


user_data.iloc[0:5,0:5]


# In[29]:


np.fill_diagonal(user,0)


# In[31]:


user_data.idxmax(axis = 1)


# In[32]:


book3[(book3['userid'] == 162107) | (book3['userid'] == 276726)]


# In[33]:


user_1 = book3[book3['userid'] == 276729]
user_1


# In[34]:


user_2 = book3[book3['userid'] == 276726]


# In[35]:


pd.merge(user_1,user_2, on = 'booktitle', how = 'outer')


# In[ ]:





# In[ ]:




