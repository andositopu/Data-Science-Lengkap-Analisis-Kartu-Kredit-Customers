#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


data=pd.read_csv('datacreditcard.csv')
data


# In[9]:


data.drop('CLIENTNUM', axis=1, inplace=True)
data.drop('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', axis=1, inplace=True)
data.drop('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2', axis=1, inplace=True)
data.head()


# In[10]:


display(data.columns)


# In[11]:


display(data.info)


# In[12]:


display(data.describe)


# In[13]:


data.isna().sum()


# In[14]:


databaru= data.sum(axis=1)
databaru


# In[15]:


databaru.rank(ascending= False, method='dense').sort_values().head()


# In[16]:


#ini salah,
databaru=data.sum(axis=1).sort_values(ascending=False)
databaru.plot(kind='bar', style='b', alpha=0.4, title='Grafik Data')


# In[17]:


import matplotlib.pyplot as plt
df=pd.value_counts(data['Attrition_Flag']).tolist()
plt.pie(x=df, labels=["Attrited Customer","Existing Customers"], autopct='%.2f%%', colors=['r','c'],shadow=False)
plt.figure(figsize=(5,5))


# In[18]:


import seaborn as sns
sns.countplot(data['Customer_Age'], hue=data['Attrition_Flag'])
plt.figure(figsize=(30,10))
plt.title('Distribusi berdasarkan umur')
plt.show()


# In[19]:


display(data.columns)


# In[20]:


data3=['Months_Inactive_12_mon','Contacts_Count_12_mon', 'Credit_Limit', 'Total_Amt_Chng_Q4_Q1','Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1']
data1=data[data['Attrition_Flag'] == 'Existing Customer']
data2= data[data['Attrition_Flag'] == 'Attrited Customer']

for xie in data3:
    sns.distplot(data1[xie], label="Existing")
    sns.distplot(data2[xie], label="Attrited")
    plt.legend()
    plt.figure()


# In[21]:


data.skew()


# In[22]:


dataX=data.select_dtypes(include=[object])
dataY=data.select_dtypes(exclude=[object])


# In[23]:


#one-hot-encoding
dataZ=pd.get_dummies(dataX.drop(columns=['Attrition_Flag']),drop_first=True)
dataZ


# In[24]:


#menggabungkan dataY dan objek, 
#Fungsi CONCAT menggabungkan teks dari beberapa rentang dan/atau string, namun tidak memberikan argumen pemisah atau IgnoreEmpty.
df=pd.concat([dataZ,dataY], axis=1)
df


# In[25]:


#split data independent dan dependent
#Fungsi get_dummies digunakan untuk mengubah variabel kategorikal menjadi variabel numerikal dengan melakukan proses One-Hot-Encode terhadap variabel kategorikal
Y=pd.get_dummies(dataX['Attrition_Flag'], drop_first=True)
X=df
print(Y)


# In[26]:


#pembagian data untuk sebagai data test dan data train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.5, random_state=30)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[27]:


#SMOTETomek (over sampling dan under sampling) adalah untuk meratakan distribusi data dengan resampling data agar seimbang
#ravel() adalah untuk mengembalikan tampilan asli aray ex: x [1,2][3,4] menjadi x [1,2,3,4]

from imblearn.combine import SMOTETomek
from collections import Counter

smot=SMOTETomek(random_state=40)
smotX, smotY= smot.fit_sample(X_train, y_train.values.ravel())

print(smotX.shape, smotY.shape)
print(Counter(smotY))

sns.countplot(smotY, edgecolor='black')
plt.title('SMOTETomek')


# In[28]:


#klasifikasi data 
#kita akan mencari yang terbaik dari berbagai algoritma berikut sebagai data test

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


# In[29]:


preprocessor =  make_pipeline(SelectKBest(f_classif, k=10) ,PolynomialFeatures(2))


AdaBoost = make_pipeline(preprocessor,AdaBoostClassifier(random_state=0))
SVM= make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))
GBoost = make_pipeline(preprocessor, StandardScaler(), GradientBoostingClassifier())
RandomForest = make_pipeline( preprocessor, RandomForestClassifier())
XGB = make_pipeline( preprocessor, XGBClassifier())
Extree = make_pipeline( preprocessor, ExtraTreesClassifier())

dict_of_models = {'AdaBoost':AdaBoost,'SVM':SVM,'GBoost':GBoost,'RandomForest':RandomForest,'XGB':XGB,'Extree':Extree}


# In[30]:


from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

def evaluation(model_1):
    model_1.fit(smotX, smotY)
    ypred = model_1.predict(X_test)
    cm = confusion_matrix(y_test, ypred)
    N, train_score, val_score = learning_curve(model_1, smotX, smotY,cv=4, scoring='f1',train_sizes=np.linspace(0.1, 1, 10))
   
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    plt.show()
    
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))


# In[31]:


for name, model in dict_of_models.items():
    print(name)
    evaluation(model)


# In[35]:


import numpy as np

param_grid=[{'max_depth': np.arange(5,8,1), 
             'min_child_weight': [1,2,3], 
             'gamma': [0], 
             'learning_rate':[0.01,0.05,0.1,0.5,1],
             'subsample': [0.6,0.7,0.8]}]
model = XGBClassifier()
grid = GridSearchCV(model, param_grid, scoring='recall', n_jobs=1, cv=3)
grid_result = grid.fit(smotX, smotY)

print(grid_result.best_score_, grid_result.best_params_)
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]

for mean, stdev, param in zip (means, stds, params):
    print("%f (%f) with : %r" % (mean, stdev, param))


# In[49]:


#dari nilai diatas didapatkan nilai terbaik pada 
#0.9824500799263053 {'gamma': 0, 'learning_rate': 0.5, 'max_depth': 5, 'min_child_weight': 1, 'subsample': 0.8}
from sklearn.metrics import*
from sklearn.model_selection import* 
from sklearn.preprocessing import*

model = XGBClassifier(gamma=0, learning_rate=0.5, mak_depth=5, min_child_weight=1, subsample=0.8)
model.fit(smotX,smotY)
zpred = model.predict(X_test)

acc = accuracy_score(y_test, zpred)
error = mean_squared_error(y_test.values, zpred)

print('\033[91m' + '\033[1m' + "akurasi: ", acc)
print('\033[93m' + "Dengan Error : "+ str(error))


# In[51]:


pd.DataFrame(model.feature_importances_, index=X_train.columns).plot(kind='barh', figsize=(18, 10))


# In[60]:


pd.DataFrame(model.feature_importances_, index=X_train.columns).plot(kind='hist', color='red',style='step', figsize=(18, 10))

