#!/usr/bin/env python
# coding: utf-8

# ### Support Vector Machine (SVM)
# A support vector machine (SVM) is a supervised machine learning model that uses classification algorithms for two-group classification problems. After giving an SVM model sets of labeled training data for each category, they're able to categorize new text. 
# A detailed explanation of the theory along with the derivation is aviable in my notes, which is available in the attached pdf
# I recommend you to have a quick glance at the theory , However feel free to skip the derivation of optimization using lagrange multiplier if you are not familiar with it. 
# 
# I made this note while listening to Prof Patrick Winston's Course on Artificial Intelligence Lecture 16:SVM
# If you have any queries on my notes please go though the lecture 
# https://www.youtube.com/watch?v=_PwhiWxHK8o 
# 
# 
# 

# 
# ### Data Set Information:
# 
# This is one of three domains provided by the Oncology Institute that has repeatedly appeared in the machine learning literature. (See also lymphography and primary-tumor.)
# 
# This data set includes 201 instances of one class and 85 instances of another class. The instances are described by 9 attributes, some of which are linear and some are nominal.
# 
# 
# This breast cancer domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia. Thanks go to M. Zwitter and M. Soklic for providing the data. Please include this citation if you plan to use this database.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


df=pd.read_csv("bcdata.csv")


# In[13]:


df.head()


# In[14]:


df.dtypes


# The data we have are not all numbers, hence Lets convert these categorical data into numbers for mathematical convenience.

# In[15]:


df['Class']=df.Class.astype("category").cat.codes
df['age']=df.age.astype("category").cat.codes
df['menopause']=df.menopause.astype("category").cat.codes
df['tumorsize']=df.tumorsize.astype("category").cat.codes
df['invnodes']=df.invnodes.astype("category").cat.codes
df['nodecaps']=df.nodecaps.astype("category").cat.codes
df['breast']=df.breast.astype("category").cat.codes
df['breastquad']=df.breastquad.astype("category").cat.codes
df['irradiat']=df.irradiat.astype("category").cat.codes


# In[18]:


df.head()


# In[59]:


X=df.iloc[:,9].values.reshape(-1, 1)
Y=df.iloc[:,0].values.reshape(-1, 1)


# In[60]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(X,Y,test_size = 0.25, random_state =0)


# Feature Scaling # first we will try without training and then come back and run this part of the code to see the the difference in accuracy #since this is categorical, scaling is not needed

# In[61]:


from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
x_train =sc.fit_transform(x_train)
x_test =sc.fit_transform(x_test)


# Traning the svm model on the traning data

# In[62]:


from sklearn.svm import SVC
classifier =SVC(kernel = 'linear',random_state =0)
classifier.fit(x_train,y_train)


# Predict the test data

# In[63]:


y_pred = classifier.predict(x_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# Creating confusion matrix

# In[64]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

