
# coding: utf-8

# #FAKE NEWS CLASSIFIER USING NLP 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df= pd.read_csv("train.csv")


# In[3]:


df.head()


# In[4]:


X = df.drop('label',axis=1)


# In[5]:


y= df['label']


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df = df.dropna()


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


# In[11]:


messages = df.copy()


# In[12]:


messages.reset_index(inplace=True)


# In[13]:


import nltk


# In[18]:


from nltk.corpus import stopwords


# In[21]:


from nltk.stem import PorterStemmer
import re


# In[22]:


ps = PorterStemmer()


# In[23]:


corpus=[]
for i in range(0 , len(messages)):
    review =  re.sub('[^a-zA-Z]',' ',messages['title'][i])
    review = review.lower()
    review= review.split()
    review =[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    


# In[24]:


from sklearn.feature_extraction.text import CountVectorizer


# In[25]:


cv = CountVectorizer(max_features= 5000, ngram_range=(1,3))


# In[34]:


X=cv.fit_transform(corpus).toarray()


# In[35]:


X.shape


# In[36]:


X


# In[37]:


y= messages['label']


# # Splitting the Data

# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X_train, X_test , y_train, y_test= train_test_split(X,y,test_size=0.33, random_state=0)


# In[40]:


X_train.shape


# In[41]:


cv.get_feature_names()[:20]


# In[42]:


cv.get_params()


# In[43]:


count_df= pd.DataFrame(X_train, columns= cv.get_feature_names())


# In[44]:


count_df.head()


# # Building Prediction Model using MultinomialNB

# In[46]:


from sklearn.naive_bayes import MultinomialNB


# In[48]:


classifier= MultinomialNB()


# In[49]:


from sklearn import metrics


# In[50]:


import itertools


# In[51]:


classifier.fit(X_train,y_train)
pred= classifier.predict(X_test)
score= metrics.accuracy_score(y_test,pred)


# In[52]:


print(score)


# In[53]:


from sklearn.metrics import confusion_matrix


# In[55]:


cm = confusion_matrix(y_test,pred)


# In[56]:


cm


# # from sklearn.linear_model import PassiveAggressiveClassifier

# In[59]:


linear_clf= PassiveAggressiveClassifier()


# In[60]:


linear_clf.fit(X_train,y_train)
pred = linear_clf.predict(X_test)
score= metrics.accuracy_score(y_test,pred)


# In[65]:


print(score)


# In[66]:


y_pred= linear_clf.predict(X_test)


# In[67]:


y_pred

