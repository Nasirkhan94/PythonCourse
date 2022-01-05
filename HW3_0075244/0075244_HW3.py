

# # CSSM 550 - Review and Practice Fundamental Machine Learning Concepts
# 


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


## Read the Data set

data = pd.read_csv('cses4_cut.csv')
data
X = data.iloc[:,:-1] #select the columns from position 0 till one before the last column
y = data.iloc[:,-1]  #select the last column

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30,random_state=97) # split data 70/30 ratio


# ## Classifiers

# In[3]:


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

#### Linear Discriminant Analysis
LDA = LinearDiscriminantAnalysis()
LDA_accuracy=cross_val_score(LDA, X, y, cv=cv).mean()

#### Quadratic Discriminant Analysis
QDA = QuadraticDiscriminantAnalysis()
QDA_accuracy=cross_val_score(QDA, X, y, cv=cv).mean()

#### Random Forest Classifier
random_forest = RandomForestClassifier()
RF_accuracy=cross_val_score(random_forest, X, y, cv=cv).mean()

#### K-Nearest Neighbors
KNN = KNeighborsClassifier()
KNN_accuracy=cross_val_score(KNN, X, y, cv=cv).mean()

#### Logistic Regression
LR = LogisticRegression()
LR_accuracy=cross_val_score(LR, X, y, cv=cv).mean()

#### Decision Tree
decision_tree = DecisionTreeClassifier()
DT_accuracy=cross_val_score(decision_tree, X, y, cv=cv).mean()

#### Support Vector Machine
SVM = SVC(probability = True)
SVM_accuracy=cross_val_score(SVM, X, y, cv=cv).mean()



#### Naive Bayes
bayes = GaussianNB()
BAYES_accuracy=cross_val_score(bayes, X, y, cv=cv).mean()

pd.options.display.float_format = '{:,.2f}%'.format
accuracies1 = pd.DataFrame({
    'Model'       : ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],
    'Accuracy'    : [100*LR_accuracy, 100*DT_accuracy, 100*SVM_accuracy, 100*LDA_accuracy, 100*QDA_accuracy, 100*RF_accuracy, 100*KNN_accuracy, 100*BAYES_accuracy],
    }, columns = ['Model', 'Accuracy'])

accuracies1.sort_values(by='Accuracy', ascending=False)


# ## Feature selection and Dimensionality-reduction

# In[4]:


#Select features according to the k highest scores

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

test = SelectKBest(score_func=chi2, k='all')
fit = test.fit(X, y)
kscores = fit.scores_
X_new = test.fit_transform(X, y)

# Features in descending order by score
dicts = {}
dicts=dict(zip(data.columns, kscores))
sort_dicts = sorted(dicts.items(), key=lambda x: x[1], reverse=True)

# 20 features with the highest score
sort_dicts[:20]


# In[5]:


# Subset of 10 feature selection starting from 2016
X_new=data[['D2016','D2021','D2022','D2023','D2026','D2027','D2028','D2029','D2030','age']] #Data subset
X_new


# In[6]:


# data distribution of new table

import matplotlib.pyplot as plt
plt.figure(figsize = (20, 15))
plotnumber = 1

for column in X_new:
    if plotnumber <= 10:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(X_new[column],fit_kws={"color":"blue"})
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()




# Preprocess, the subset for plotting 

from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer

quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_new_trans = quantile_transformer.fit_transform(X_new)
import matplotlib.pyplot as plt
plt.figure(figsize = (20, 15))
plotnumber = 1

for column in range(X_new_trans.shape[1]):
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(X_new_trans[column])
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# ## Classifiers with dimensionality-reduction and pre-processing

# In[8]:


#### Support Vector Machine
SVM = SVC(probability = True)
SVM_accuracy=cross_val_score(SVM, X_new_trans, y, cv=cv).mean()

#### Linear Discriminant Analysis
LDA = LinearDiscriminantAnalysis()
LDA_accuracy=cross_val_score(LDA, X_new_trans, y, cv=cv).mean()

#### Logistic Regression
LR = LogisticRegression()
LR_accuracy=cross_val_score(LR, X_new_trans, y, cv=cv).mean()

#### Decision Tree
decision_tree = DecisionTreeClassifier()
DT_accuracy=cross_val_score(decision_tree, X_new_trans, y, cv=cv).mean()


#### Quadratic Discriminant Analysis
QDA = QuadraticDiscriminantAnalysis()
QDA_accuracy=cross_val_score(QDA, X_new_trans, y, cv=cv).mean()

#### Random Forest Classifier
random_forest = RandomForestClassifier()
RF_accuracy=cross_val_score(random_forest, X_new_trans, y, cv=cv).mean()

#### K-Nearest Neighbors
KNN = KNeighborsClassifier()
KNN_accuracy=cross_val_score(KNN, X_new_trans, y, cv=cv).mean()

#### Naive Bayes
bayes = GaussianNB()
BAYES_accuracy=cross_val_score(bayes, X_new_trans, y, cv=cv).mean()

pd.options.display.float_format = '{:,.2f}%'.format
accuracies2 = pd.DataFrame({
    'Model'       : ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],
    'Accuracy'    : [100*LR_accuracy, 100*DT_accuracy, 100*SVM_accuracy, 100*LDA_accuracy, 100*QDA_accuracy, 100*RF_accuracy, 100*KNN_accuracy, 100*BAYES_accuracy],
    }, columns = ['Model', 'Accuracy'])

accuracies2.sort_values(by='Accuracy', ascending=False)


# ## Hyperparameter Tuning

# In[9]:


#### Linear Discriminant Analysis
        
best_score=0        
solver=['svd', 'lsqr', 'eigen']
for i in solver:    
    LDA = LinearDiscriminantAnalysis(solver=i)
    LDA_accuracy=cross_val_score(LDA, X_new_trans, y, cv=cv).mean()
    if LDA_accuracy>best_score:
        best_score=LDA_accuracy
        best_solver=i
LDA_accuracy=best_score
print("Best score is:",best_score,"with solver:",best_solver)

#### Random Forest Classifier

best_score=0
n_estimators= [100,200,500,1000]
criterions=['gini', 'entropy']
for i in n_estimators:
    for k in criterions:
        random_forest = RandomForestClassifier(n_estimators=i,criterion=k)
        RF_accuracy=cross_val_score(random_forest, X_new_trans, y, cv=cv).mean()
        if RF_accuracy > best_score:
            best_score=RF_accuracy
            best_est=i
            best_cri=k
RF_accuracy=best_score
print("Best score is:",best_score,"with estimator:",best_est,"criterion:",best_cri)

        
#### Logistic Regression

best_score=0     
penalty=['l1', 'l2', 'elasticnet', 'none']
for i in penalty:
    LR = LogisticRegression(penalty=i)
    LR_accuracy=cross_val_score(LR, X_new_trans, y, cv=cv).mean()
    if LR_accuracy > best_score:
        best_score=LR_accuracy
        best_p=i
LR_accuracy=best_score
print("Best score is:",best_score,"with penalty",best_p)


#### K-Nearest Neighbors
        
best_score=0
for i in range(2,10):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN_accuracy=cross_val_score(KNN, X_new_trans, y, cv=cv).mean()
    if KNN_accuracy > best_score:
        best_score=KNN_accuracy
        best_n=i
KNN_accuracy=best_score
print("Best score is:",best_score,"with number of neighbors:",best_n)
    

pd.options.display.float_format = '{:,.2f}%'.format
accuracies3 = pd.DataFrame({
    'Model'       : ['Logistic Regression', 'Linear Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors',],
    'Accuracy'    : [100*LR_accuracy, 100*LDA_accuracy, 100*RF_accuracy, 100*KNN_accuracy],
    }, columns = ['Model', 'Accuracy'])

accuracies3.sort_values(by='Accuracy', ascending=False)


# # Conclusion and Final Outcomes

# In[11]:


print("Classifiers with no preprocessing:")
print(accuracies1.sort_values(by='Accuracy', ascending=False))
print("Classifiers accuracy with pre-processing and optimizing hperparameters:")
print(accuracies2.sort_values(by='Accuracy', ascending=False))





