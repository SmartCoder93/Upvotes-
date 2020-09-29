#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


UpvotesTrain=pd.read_csv("/Users/shankarraja/Downloads/Data Science/DSProjects/Upvotes/UPVTrain.csv.xls")


# In[5]:


UpvotesTest=pd.read_csv("/Users/shankarraja/Downloads/Data Science/DSProjects/Upvotes/UPVTrain.csv.xls")


# In[6]:


UpvotesTrain


# In[7]:


UpvotesTrain.shape


# In[9]:


UpvotesTrain.apply(lambda x: sum(x.isnull()))


# In[10]:


UpvotesTrain.dtypes


# In[11]:


UpvotesTrain.Tag.value_counts()


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


labelencoderX=LabelEncoder()


# In[15]:


UpvotesTrain["Tag"]=labelencoderX.fit_transform(UpvotesTrain.iloc[:,1].values)


# In[16]:


UpvotesTrain.head()


# In[17]:


UpvotesTrain.dtypes


# In[18]:


UpvotesTrain.Tag.value_counts()


# In[19]:


X_train=UpvotesTrain.iloc[:,0:6]
Y_train=UpvotesTrain.iloc[:,-1]


# In[20]:


X_train


# In[23]:


pd.DataFrame(Y_train)


# In[24]:


from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(X_train, Y_train)


# In[25]:


UpvotesTest


# In[26]:


UpvotesTest.apply(lambda x: sum(x.isnull()))


# In[27]:


UpvotesTest.dtypes


# In[28]:


UpvotesTest["Tag"]=labelencoderX.fit_transform(UpvotesTest.iloc[:,1].values)
UpvotesTest.head()


# In[29]:


UpvotesTest.dtypes


# In[36]:


UpvotesTest


# In[38]:


UpvotesTest_1=UpvotesTest.iloc[:,0:6]
UpvotesTest_1


# In[39]:


X_test=UpvotesTest_1


# In[40]:


Y_pred=Regressor.predict(X_test)


# In[41]:


SampSub=pd.read_csv("/Users/shankarraja/Downloads/Data Science/DSProjects/Upvotes/UPVTrain.csv.xls")
SampSub


# In[3]:


SampSub["Upvotes"]=Y_pred
SampSub


# In[2]:


SampSub.to_csv("UpvotesSampleSubCompleted.csv",index=False)

