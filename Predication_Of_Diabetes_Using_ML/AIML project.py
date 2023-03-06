#!/usr/bin/env python
# coding: utf-8

# # Importing essential libraries

# In[14]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier


# # Loading the dataset

# In[15]:


df = pd.read_csv('diabetes.csv')


# In[16]:


print('Shape of the Data:::',df.shape)


# In[17]:


df.head()


# In[18]:


df.info()  


# In[19]:


df.describe()


# In[20]:


df.isna().sum()


# In[21]:


# Lets check the Duplicates columns 
df.duplicated().sum()


# # EDA

# In[28]:


#plt.figure(figsize=(10,10))
df.boxplot()


# # 1)Pregnancies

# In[23]:


df['Pregnancies'].unique()


# In[24]:


df['Pregnancies'].nunique()


# In[25]:


df['Pregnancies'].value_counts()


# In[ ]:


df['Pregnancies'].


# In[26]:


sns.boxplot(df['Pregnancies'])


# # 2)Glucose

# In[13]:


df['Glucose'].unique()


# In[14]:


df['Glucose'].nunique()


# In[15]:


df['Glucose'].value_counts()


# In[16]:


#need to repace 0 to Nan and then handling messing vaues with [[ mean ]] because Glucose laval can not be zero


# # 3)BloodPressure

# In[17]:


df['BloodPressure'].unique()


# In[18]:


df['BloodPressure'].nunique()


# In[19]:


df['BloodPressure'].value_counts().head(10)


# In[20]:


#need to repace 0 to Nan and then handling messing vaues with [[ mean ]] because BloodPressure laval can not be zero


# # 4)SkinThickness

# In[21]:


df['SkinThickness'].unique()


# In[22]:


df['SkinThickness'].value_counts().head(3)


# In[23]:


#need to repace 0 to Nan and then handling messing vaues with [[ median ]] because BloodPressure laval can not be zero
# value by mean, median depending upon distribution


# # 5)Insulin

# In[24]:


df['Insulin'].nunique()


# In[25]:


df['Insulin'].value_counts()


# In[26]:


#need to repace 0 to Nan and then handling messing vaues with [[ median ]] because BloodPressure laval can not be zero
# value by mean, median depending upon distribution
# no need to encoading all are in numerical formate....


# # 6)BMI

# In[27]:


df['BMI'].nunique()


# In[28]:


df['BMI'].value_counts()


# In[29]:


#need to repace 0 to Nan and then handling messing vaues with [[ median ]] because BloodPressure laval can not be zero
# value by mean, median depending upon distribution
# no need to encoading all are in numerical formate....


# # 7)DiabetesPedigreeFunction

# In[30]:


df['DiabetesPedigreeFunction'].nunique()


# In[31]:


df['DiabetesPedigreeFunction'].value_counts()


# # 8)Age

# In[32]:


df['Age'].unique()


# In[33]:


df['Age'].nunique()


# In[34]:


df['Age'].value_counts().head(5)


# # Feture engineering

# In[35]:


# Replacing NaN value by mean, median depending upon distribution
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)


# In[36]:


# handling all zeros with mean, median depending upon distribution


# # 9 ) Outcome

# In[37]:


df['Outcome'].unique() ## target contain 0 or 1 this is the problem of classification


# In[38]:


df['Outcome'].value_counts()


# In[39]:


df['Outcome'].value_counts(normalize=True)  # cheak for data balance or imbalcnace 


# # Model Building

# In[40]:


x = df.drop(columns='Outcome')
y = df['Outcome']


# In[41]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


# In[42]:


print('x_train::',x_train.shape)
print('x_test::',x_test.shape)
print('y_train::',y_train.shape)
print('y_test::',y_test.shape)


# ## LOG_Reg base model

# In[43]:


log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)


# In[44]:


#testing Evaluation
y_predict_test=log_reg.predict(x_test)
cnf_matrix=confusion_matrix(y_test,y_predict_test)
clss_report=classification_report(y_test,y_predict_test)
print('CONFUSION_MATRIX:\n',cnf_matrix)
print('classification_report:\n',clss_report)


# In[45]:


#Training Evaluation
y_predict_train=log_reg.predict(x_train)
cnf_matrix=confusion_matrix(y_train,y_predict_train)
clss_report=classification_report(y_train,y_predict_train)
print('CONFUSION_MATRIX:\n',cnf_matrix)
print('classification_report:\n',clss_report)


# In[46]:


print('Accuracy For Testing:',np.around(accuracy_score(y_test,y_predict_test),2))
print('Accuracy For Training:',np.around(accuracy_score(y_train,y_predict_train),2))


# # KNN

# In[47]:


knn_clf=KNeighborsClassifier()
knn_clf.fit(x_train,y_train)
#testing Evaluation
y_predict_test=knn_clf.predict(x_test)
cnf_matrix=confusion_matrix(y_test,y_predict_test)
clss_report=classification_report(y_test,y_predict_test)
print('CONFUSION_MATRIX:\n',cnf_matrix)
print('classification_report:\n',clss_report)


# In[48]:


#Training Evaluation
y_predict_train=knn_clf.predict(x_train)
cnf_matrix=confusion_matrix(y_train,y_predict_train)
clss_report=classification_report(y_train,y_predict_train)
print('CONFUSION_MATRIX:\n',cnf_matrix)
print('classification_report:\n',clss_report)


# In[49]:


print('Accuracy For Testing:',np.around(accuracy_score(y_test,y_predict_test),2))
print('Accuracy For Training:',np.around(accuracy_score(y_train,y_predict_train),2))


# ## Hyperpyarameter Tuning

# In[50]:


knn_clf=KNeighborsClassifier()
hyperpyarameter={'n_neighbors':np.arange(1,30),
                'p':[1,2]}
gscv_knn=GridSearchCV(knn_clf,hyperpyarameter,cv=5)
gscv_knn.fit(x_train,y_train)
gscv_knn.best_params_


# In[51]:


gscv_knn.best_estimator_


# In[52]:


#testing Evaluation
y_predict_test=gscv_knn.predict(x_test)
cnf_matrix=confusion_matrix(y_test,y_predict_test)
clss_report=classification_report(y_test,y_predict_test)
print('CONFUSION_MATRIX:\n',cnf_matrix)
print('classification_report:\n',clss_report)


# In[53]:


#Training Evaluation
y_predict_train=gscv_knn.predict(x_train)
cnf_matrix=confusion_matrix(y_train,y_predict_train)
clss_report=classification_report(y_train,y_predict_train)
print('CONFUSION_MATRIX:\n',cnf_matrix)
print('classification_report:\n',clss_report)


# In[54]:


print('Accuracy For Testing:',np.around(accuracy_score(y_test,y_predict_test),2))
print('Accuracy For Training:',np.around(accuracy_score(y_train,y_predict_train),2))


# ## Creating Random Forest Model

# In[55]:


rf_model=RandomForestClassifier(random_state=12)
rf_model.fit(x_train,y_train)


# In[56]:


#testing Evaluation
y_predict_test=rf_model.predict(x_test)
cnf_matrix=confusion_matrix(y_test,y_predict_test)
clss_report=classification_report(y_test,y_predict_test)
print('CONFUSION_MATRIX:\n',cnf_matrix)
print('classification_report:\n',clss_report)


# In[57]:


#Training Evaluation
y_predict_train=rf_model.predict(x_train)
cnf_matrix=confusion_matrix(y_train,y_predict_train)
clss_report=classification_report(y_train,y_predict_train)
print('CONFUSION_MATRIX:\n',cnf_matrix)
print('classification_report:\n',clss_report)


# In[58]:


print('Accuracy For Testing:',np.around(accuracy_score(y_test,y_predict_test),2))
print('Accuracy For Training:',np.around(accuracy_score(y_train,y_predict_train),2))


# In[59]:


rf_model=RandomForestClassifier(random_state=12)
hyp={"n_estimators":np.arange(10,150),
    'criterion':['gini','entropy'],
    'max_depth':np.arange(5,15),
    'min_samples_split':np.arange(5,20),
    'min_samples_leaf':np.arange(4,15),
    'random_state':np.arange(1,10)}
rscv=RandomizedSearchCV(rf_model,hyp,cv=5)
rscv.fit(x_train,y_train)
rscv.best_params_


# In[60]:


#testing Evaluation
y_predict_test=rscv.predict(x_test)
cnf_matrix=confusion_matrix(y_test,y_predict_test)
clss_report=classification_report(y_test,y_predict_test)
print('CONFUSION_MATRIX:\n',cnf_matrix)
print('classification_report:\n',clss_report)


# In[61]:


#Training Evaluation
y_predict_train=rscv.predict(x_train)
cnf_matrix=confusion_matrix(y_train,y_predict_train)
clss_report=classification_report(y_train,y_predict_train)
print('CONFUSION_MATRIX:\n',cnf_matrix)
print('classification_report:\n',clss_report)


# In[62]:


print('Accuracy For Testing:',np.around(accuracy_score(y_test,y_predict_test),2))
print('Accuracy For Training:',np.around(accuracy_score(y_train,y_predict_train),2))


# In[64]:


# # Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(rscv,open(filename,'wb'))


# # THE END  #####

# In[ ]:




