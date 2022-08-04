#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[3]:


book=pd.read_csv("D:\\data science\\assignments\\ass-9 association\\book.csv")


# In[4]:


book


# In[5]:


book.info()


# # Apriori Algorithm 

# In[7]:


# at min_support=0.2 , support values will be more then 0.2
frequent_itemsets_2=apriori(book, min_support=0.2, use_colnames=True)
frequent_itemsets_2


# In[8]:


rules2 = association_rules(frequent_itemsets_2, metric="lift", min_threshold=0.7)
rules2


# In[9]:


rules2.sort_values('lift',ascending = False)[0:20]


# In[10]:


rules2[rules2.lift>1]


# In[11]:


df1=pd.DataFrame(data=frequent_itemsets_2)
df1
df1.duplicated()


# In[12]:


# visualization of obtained rule
rules2.plot(kind='bar',x='support',y='confidence',color='darkblue')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')


# In[13]:


plt.scatter(rules2.support,rules2.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[15]:


#at min_support=0.17 , support values will be greater then 0.17
frequent_itemsets_17=apriori(book, min_support=0.17, use_colnames=True)
frequent_itemsets_17


# In[16]:


rules17 = association_rules(frequent_itemsets_17, metric="lift", min_threshold=0.7)
rules17


# In[17]:


rules17.sort_values('lift',ascending = False)[0:20]


# In[18]:


rules17[rules17.lift>1]


# In[19]:


df17=pd.DataFrame(data=frequent_itemsets_17)
df17
df17.duplicated()


# In[20]:


rules17.plot(kind='bar',x='support',y='confidence',color='yellow')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')


# In[21]:


plt.scatter(rules17.support,rules17.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[23]:


#at min_support=0.15 ,support values will be greater then 0.17
frequent_itemsets_15=apriori(book, min_support=0.15, use_colnames=True)
frequent_itemsets_15


# In[24]:


rules15 = association_rules(frequent_itemsets_15, metric="lift", min_threshold=0.7)
rules15


# In[25]:


rules15.sort_values('lift',ascending = False)[0:20]


# In[26]:


df15=pd.DataFrame(data=frequent_itemsets_15)
df15
df15.duplicated()


# In[27]:


rules15.plot(kind='bar',x='support',y='confidence',color='purple')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')
plt.figure(figsize=(20,10))


# In[28]:


plt.scatter(rules15.support,rules15.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[29]:


x=[0.15,0.17,0.2]
y=[21,9,2]
plt.scatter(x,y)
plt.xlabel('Minimum Support')
plt.ylabel('Frequent Itemsets')
plt.title('Relation Between Min Support Value and Frequent Itemsets')


# In[30]:


x=[0.1,0.2,0.3]
y=[52,12,6]
plt.scatter(x,y)
plt.xlabel('Minimum Support')
plt.ylabel('Frequent Itemsets')
plt.title('Relation Between Min Support Value and Frequent Itemsets')


# In[ ]:





# In[32]:



import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[34]:


movie=pd.read_csv("D:\\data science\\assignments\\ass-9 association\\my_movies.csv")


# In[35]:


movie.info()


# In[38]:


movie1=movie.drop(['V1','V2','V3','V4','V5'], axis=1)
movie1


# # Apriori Algorithm

# In[40]:


#with min_support of 0.1
frequent_itemsets1=apriori(movie1,min_support=0.1,use_colnames=True)
frequent_itemsets1


# In[42]:


rules1= association_rules(frequent_itemsets1, metric="lift", min_threshold=0.7)
rules1


# In[43]:


rules1.sort_values('lift',ascending = False)[0:20]


# In[44]:


rules1[rules1.lift>1]


# In[45]:


df1=pd.DataFrame(data=frequent_itemsets1)
df1
df1.duplicated()


# In[46]:



# visualization of obtained rule
rules1.plot(kind='bar',x='support',y='confidence',color='green')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')
#plt.figure(figsize=(30,10))


# In[48]:


# visualization of obtained rule
plt.scatter(rules1.support,rules1.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[50]:


#with min_support of 0.2
frequent_itemsets2=apriori(movie1,min_support=0.2,use_colnames=True)
frequent_itemsets2


# In[51]:


rules2= association_rules(frequent_itemsets2, metric="lift", min_threshold=0.7)
rules2


# In[52]:


rules2.sort_values('lift',ascending = False)[0:20]


# In[53]:


rules2[rules2.lift>1]


# In[54]:


df2=pd.DataFrame(data=frequent_itemsets2)
df2
df2.duplicated()


# In[55]:


# visualization of obtained rule
rules2.plot(kind='bar',x='support',y='confidence',color='black')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')
#plt.figure(figsize=(30,10))


# In[57]:


# visualization of obtained rule
plt.scatter(rules2.support,rules2.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[59]:


#with min_support of 0.3
frequent_itemsets3=apriori(movie1,min_support=0.3,use_colnames=True)
frequent_itemsets3


# In[60]:


rules3= association_rules(frequent_itemsets3, metric="lift", min_threshold=0.7)
rules3


# In[61]:


rules3.sort_values('lift',ascending = False)[0:20]


# In[62]:


rules3[rules3.lift>1]


# In[63]:


df3=pd.DataFrame(data=frequent_itemsets3)
df3
df3.duplicated()


# In[65]:


# visualization of obtained rule
rules3.plot(kind='bar',x='support',y='confidence',color='green')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')
#plt.figure(figsize=(30,10))


# In[66]:


# visualization of obtained rule
plt.scatter(rules3.support,rules3.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[67]:


x=[0.1,0.2,0.3]
y=[52,12,6]
plt.scatter(x,y)
plt.xlabel('Minimum Support')
plt.ylabel('Frequent Itemsets')
plt.title('Relation Between Min Support Value and Frequent Itemsets')


# In[ ]:





# In[ ]:




