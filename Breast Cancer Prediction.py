#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Loading the dataset

# In[2]:


df = pd.read_csv('cancer_classification.csv')
df.head()


# In[3]:


df.info()


# ### Knowing somes statistical details

# In[4]:


df.describe().T


# ### EXPLORATORY DATA ANALYSIS

# In[5]:


sns.countplot(df['benign_0__mal_1'])


# In[6]:


df.corr()['benign_0__mal_1'].sort_values().plot(kind = 'bar')


# ### Finding the most correlated features

# In[7]:


sns.heatmap(df.corr())


# In[8]:


X = df.drop('benign_0__mal_1', axis = 1).values
y = df['benign_0__mal_1'].values


# In[9]:


from sklearn.model_selection import train_test_split


# ### Dividing data into training and test set

# In[10]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 101)


# In[11]:


from sklearn.preprocessing import MinMaxScaler


# In[12]:


scaler = MinMaxScaler()


# ### Normalizing the independant features

# In[13]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[14]:


from tensorflow.keras.models import Sequential


# In[15]:


from tensorflow.keras.layers import Dense,Dropout


# In[16]:


X_train.shape


# In[17]:


model = Sequential()


# In[18]:



model.add(Dense(30,activation = 'relu'))

model.add(Dense(15,activation = 'relu'))


model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam')


# In[19]:


model.fit(X_train, y_train, epochs = 600, validation_data = (X_test,y_test))


# In[21]:


losses = pd.DataFrame(model.history.history)


# In[22]:


losses.plot()


# In[23]:



model.add(Dense(30,activation = 'relu'))

model.add(Dense(15,activation = 'relu'))


model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam')


# ### Early stopping

# In[24]:


from tensorflow.keras.callbacks import EarlyStopping


# In[25]:


help(EarlyStopping)


# In[26]:


early_stop = EarlyStopping(monitor='val_loss', mode = 'min', verbose = 1,
                          patience = 25)


# In[27]:


model.fit(X_train,y_train, epochs = 600, validation_data = (X_test,y_test)
         ,callbacks = [early_stop])


# In[28]:


model_loss = pd.DataFrame(model.history.history)


# In[29]:


model_loss.plot()


# In[30]:


from tensorflow.keras.layers import Dropout


# In[31]:



model.add(Dense(30,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(15,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam')


# In[32]:


model.fit(X_train,y_train, epochs = 600, validation_data = (X_test,y_test)
         ,callbacks = [early_stop])


# In[33]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[34]:


predictions = model.predict_classes(X_test)


# In[35]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:




