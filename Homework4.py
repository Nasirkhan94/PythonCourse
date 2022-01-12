#!/usr/bin/env python
# coding: utf-8

# # Homework 4
# ## Nasir Khan (0075244)

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('immSurvey.csv')
print(df)


# In[105]:


data= df.iloc[:,0]
sentiment=df.iloc[:,1]
sentiment=df.iloc[:,1]
sentiment=sentiment.to_numpy()
print(data)


# In[104]:


vectorizer = CountVectorizer()

matrix = vectorizer.fit_transform(data)
matrix=matrix.toarray()

print(vectorizer.get_feature_names())


# In[ ]:





# In[66]:


counts = pd.DataFrame(matrix.toarray(),
                      columns=vectorizer.get_feature_names()) ## Words with frequency in ascending order
counts


# In[67]:


counts.T.sort_values(by=0, ascending=False).head(20)


# In[68]:


tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')

tfidf_matrix=tfidf.fit_transform(data)

df_tfidf = pd.DataFrame(tfidf_matrix.todense())

df_tfidf


# ## Splitting the data Using word frequency as the feature 

# In[69]:


matrix_words = pd.DataFrame(matrix.todense())
x_train, x_valid, y_train, y_valid = train_test_split(matrix_words,sentiment,test_size=0.5,random_state=2)


# ## Using Support Vector Regressor SVR

# In[121]:


from sklearn.svm import SVR
from sklearn.metrics import accuracy_score


model = SVR(kernel='rbf')
model.fit(x_train, y_train)
# make predictions
yhat = model.predict(x_valid)
# evaluate predictions



# In[100]:


import numpy as np
import matplotlib.pyplot as plt


plt.plot(x_valid, yhat, color = 'blue')

plt.ylabel("Sentiment")
plt.xlabel("word_count")
plt.show()


# ## Using Random forest Regressor RFR
# 

# In[120]:


# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor
  
 # create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
  
# fit the regressor with x and y data
regressor.fit(x_train, y_train) 

#Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))  # test the output by changing values

Y_pred=regressor.predict(x_valid)
# evaluate predictions


# In[117]:



X_grid = np.arange(min(x_valid), max(x_valid), 0.01) 
  
# reshape for reshaping the data into a len(X_grid)*1 array, 
# i.e. to make a column out of the X_grid value                  
X_grid = X_grid.reshape((len(X_grid), 1))

plt.plot(x_valid,Y_pred , 
         color = 'green') 
plt.title('Random Forest Regression')
plt.ylabel("Sentiment")
plt.xlabel("word_count")
plt.show()


# In[ ]:





# ## Splitting the data Using TF-IDF as the feature 

# In[31]:


x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(df_tfidf,sentiment,test_size=0.5,random_state=17)


# ## Using Support Vector Regressor SVR

# In[102]:



model = SVR(kernel='rbf')
model.fit(x_train_tfidf, y_train_tfidf)
# make predictions
yhat = model.predict(x_valid_tfidf)
# evaluate predictions


# In[103]:




plt.plot(x_valid_tfidf, yhat, color = 'red')

plt.ylabel("Sentiment")
plt.xlabel("TF-IDF values")
plt.show()


# ## Using Random forest Regressor RFR

# In[ ]:


# create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
 
# fit the regressor with x and y data
regressor.fit(x_train_tfidf,y_train_tfidf) 

#Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))  # test the output by changing values

Y_pred=regressor.predict(x_valid_tfidf)
# evaluate predictions
print(Y_pred)


# In[118]:


plt.plot(x_valid,Y_pred , 
         color = 'blue') 
plt.title('Random Forest Regression')
plt.ylabel("Sentiment")
plt.xlabel("TF-IDF values")
plt.show()


# ## Conclusion:
#     

# Since the data for sentiment analysis is a continuous data and not discrete so, we need to perform one hot encoding 
# for the label(sentiment) as accuracy is a poor performance metric for continuous trpe data. However, we see that both support vector regressor and random forest regresson perfomr well and fit the data well.
# 

# In[ ]:




