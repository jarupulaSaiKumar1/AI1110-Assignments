#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing modules
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier


# In[2]:


# Loading the train dataset
train_data = pd.read_csv('train_input.csv')
train_data.head()


# In[3]:


# Loading the test dataset
test_data = pd.read_csv('test_input.csv')
test_data.head()


# In[4]:


# Replacing NaN with mean values
train_data = train_data.fillna(train_data.mean())
test_data = test_data.fillna(test_data.mean())
train_data.head()


# In[5]:


# Correlation Matrix
corr = train_data.corr()
corr


# In[6]:


# Deleting Higly correlated features
# feature 10 & feature 14
# feature 3 & feature 1
# feature 11 & feature 10
# feature 23 & feature 1
train_data = train_data.drop(columns = ['Feature 10','Feature 3 (Discrete)','Feature 11','Feature 23 (Discrete)'])
test_data = test_data.drop(columns = ['Feature 10','Feature 3 (Discrete)','Feature 11','Feature 23 (Discrete)'])
train_data.head()


# In[7]:


# Splitting into X_train and Y_train
X_train, y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
X_test = test_data.iloc[:]


# In[8]:


# seeding random for consistent output
np.random.seed(100)


# In[9]:


# Using Stacking
# stacking XGBoost and Random Forest Estimators
estimators = [
    ('xg', xgb.XGBClassifier()),
    ('rf', RandomForestClassifier(n_estimators = 600, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth = 60, bootstrap=False))
]

# creating a stackig classifier
clf = StackingClassifier(
     estimators=estimators
)

# Fitting the train data set
clf.fit(X_train,y_train)

# Making prediction
y_pred = clf.predict(X_test)


# In[10]:


# Outputting data
prediction = pd.DataFrame(y_pred, columns=['Category'])
prediction.index+=1
prediction.index.name = 'Id'
prediction.to_csv('test_output.csv')
