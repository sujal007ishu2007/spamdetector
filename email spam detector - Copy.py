#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --user numpy==1.23.5 scipy==1.9.3 scikit-learn==1.1.3')
import numpy
import scipy
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('mail_data.csv')


# In[3]:


print(df)


# In[4]:


data = df.where((pd.notnull(df)),'')


# In[5]:


data.head(10)


# In[6]:


data.info()


# In[7]:


data.shape


# In[8]:


X = data['Message']
Y =  data['Category']


# In[9]:


print(X)


# In[10]:


print(Y)


# In[11]:


X_train , X_test , Y_train , Y_test  = train_test_split(X,Y,test_size=0.2,random_state = 3)


# In[12]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[13]:


feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
# Encode labels in Y_train and Y_test
Y_train = Y_train.map({'ham': 1, 'spam': 0})
Y_test = Y_test.map({'ham': 1, 'spam': 0})

# Proceed with conversion (this step may be unnecessary after map)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[14]:


print(X_train)


# In[15]:


print(X_train_features)


# In[16]:


Model = LogisticRegression()


# In[17]:


Model.fit(X_train_features,Y_train)


# In[18]:


prediction_on_training_data = Model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)


# In[19]:


print('acc on training data :',accuracy_on_training_data )


# In[20]:


prediction_on_test_data = Model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[21]:


print('acc on test data :',accuracy_on_test_data )


# In[22]:


input_your_mail = ['''Free entry in 2 a wkly  21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
''']
input_data_features = feature_extraction.transform(input_your_mail)
prediction = Model.predict(input_data_features)
print(prediction)
if(prediction[0]==1):
    print("It is not a spam mail")
else:
    print("It is spam mail")


# In[ ]:




