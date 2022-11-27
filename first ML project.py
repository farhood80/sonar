#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# <b>Data colletion and Data Processin<b>
# 

# In[23]:


#loading the data set in pandas format

sonar =pd.read_csv("/home/farhood/Desktop/sonar_data.csv", header=None)
sonar.head()


# In[24]:


# number of columns and rows , just if you are curios
sonar.shape


# In[25]:


sonar.describe()


# In[27]:


sonar[60].value_counts()


# M == Mines
# 
# R == Rocks

# In[28]:


sonar.groupby(60).mean()


# In[29]:


#separating data and labels

h1 = sonar.drop(columns = 60, axis=1)
h2 = sonar[60]

print(h1)
print(h2)


# <b>Training and Testing Data<b>

# In[30]:


h1_train, h1_test, h2_train , h2_test = train_test_split(h1,h2, test_size = 0.1, stratify = h2, random_state = 1)


# In[32]:


print(h1.shape, h1_train.shape, h1_test.shape)


# <b>Model Training (with help of the Logistic Regression<b>

# In[33]:


model = LogisticRegression()


# In[34]:


#training with LogisticRegression with training data
model.fit(h1_train, h2_train)


# <b> Model Evaluation<b>

# In[37]:


# accuracy of the training data
h1_train_prediction = model.predict(h1_train)
training_data_accuracy = accuracy_score(h1_train_prediction, h2_train)
print('Accuracy for the training model is :',training_data_accuracy )


# In[39]:


# accuracy of the test data
h1_test_prediction = model.predict(h1_test)
test_data_accuracy = accuracy_score(h1_test_prediction, h2_test)
print('Accuracy for the test model is :',test_data_accuracy )


# <b> Making a Predict System<b>

# In[46]:


#the input data should be 60 columns!
input_data = ()

#changing input data to a numpy array
input_data_numpy_array = np.asarray(input_data)

#reshape the np array as the qulified form
input_data_reshaped = input_data_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped )
prediction

if(prediction[0]=='R'):
    print("the object is a rock")
else:
    print("its a mine")


# In[ ]:




