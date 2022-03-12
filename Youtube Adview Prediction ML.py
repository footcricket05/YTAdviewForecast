#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv(r"C:\Users\Shaurya\Desktop\ML project\train.csv")


# In[2]:


data_train.head()


# In[3]:


# Assigning each category a number for Category feature
category={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}
data_train["category"]=data_train["category"].map(category)
data_train.head()


# In[4]:


#removing character "F" present in data using pandas slicing
data_train=data_train[data_train.views!='F']
data_train=data_train[data_train.likes!='F']
data_train=data_train[data_train.dislikes!='F']
data_train=data_train[data_train.comment!='F']

#convert values to integers for views , likes, comments , dislikes and adviews
data_train["views"] = pd.to_numeric(data_train["views"]) 
data_train["comment"] = pd.to_numeric(data_train["comment"])
data_train["likes"] = pd.to_numeric(data_train["likes"])
data_train["dislikes"] = pd.to_numeric(data_train["dislikes"])
data_train["adview"] = pd.to_numeric(data_train["adview"])
column_vidid=data_train['vidid']

data_train.head()


# In[5]:


#convert time in sec for duration
import datetime
import time

def checki(x):
  y=x[2:]
  h=''
  m=''
  s=''
  mm=''
  P=['H','M','S']
  for i in y:
    if i not in P:
      mm+=i
    else :
      if(i=="H"):
        h=mm
        mm=''
      elif(i=="M"):
        m=mm
        m=''
      else:
        s=mm
        s=''
    if(h==''):
      h='00'
    if(m==''):
      m='00'
    if(s==''):
      s='00'
    bp=h+':'+m+':'+s
    return bp
  train=pd.read_csv("train.csv")
  mp = pd.read_csv(path + "train.csv")["duration"]
  time = mp.apply(checki)

  def func_sec(time_string):
    h,m,s = timestring.split(':')
    return int(h) *3600 + int(m) * 60 + int(s)

  time1=time.apply(func_sec)

  data_train["duration"]=time1


# In[6]:


data_train.head()


# In[7]:


#encoding features like category, duration, vidid
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data_train["duration"]=le.fit_transform(data_train["duration"])
data_train["published"]=le.fit_transform(data_train["published"])
data_train["vidid"]=le.fit_transform(data_train["vidid"])

data_train.head()


# In[8]:


#Visualisation
#Individual Plots
plt.hist(data_train["category"])
plt.show()
plt.plot(data_train["adview"])
plt.show()


# In[9]:


#remove videos with adview greater than 2000000 as outlier
data_train=data_train[data_train["adview"]<2000000]    


# In[10]:


#heatmap
import seaborn as sns
f,ax= plt.subplots(figsize=(10,8))
corr = data_train.corr()
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool_),cmap=sns.diverging_palette(220,10,as_cmap=True),
            square=True,ax=ax,annot=True)
plt.show()


# In[11]:


#split data
Y_train = pd.DataFrame(data= data_train.iloc[:,1].values,columns = ['target'])
data_train=data_train.drop(["adview"],axis=1)
data_train=data_train.drop(["vidid"],axis=1)
data_train.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(data_train,Y_train, test_size = 0.33, random_state=42)

X_train.shape


# In[12]:


#Normalise data (convert into Numpy array from Panda array)
from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[13]:


# to evaluate performance of the particuar model we define a func to calc error
from sklearn import metrics
def print_error(X_test, y_test, model_name):
       prediction = model_name.predict(X_test)
       print('Mean Absolute Error :',metrics.mean_absolute_error(y_test,prediction))
       print('Mean Squared Error :',metrics.mean_squared_error(y_test,prediction))
       print('Root Mean Squared error : ',np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[14]:


#linear Regression
from sklearn import linear_model
linear_regression=linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)
print_error(X_test,y_test,linear_regression)


# In[15]:


#Decision Tree
from sklearn.tree import DecisionTreeRegressor
decision_tree=DecisionTreeRegressor()
decision_tree.fit(X_train,y_train)
print_error(X_test,y_test,decision_tree)


# In[16]:


#Randomn forest
from sklearn.ensemble import RandomForestRegressor
n_estimators=200
max_depth =25
min_samples_split=15
min_samples_leaf=2
random_forest = RandomForestRegressor(n_estimators = n_estimators,max_depth= max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
random_forest.fit(X_train,y_train)
print_error(X_test,y_test,random_forest)


# In[17]:


#Support Vector Regressor
from sklearn.svm import SVR
supportvector_regressor=SVR()
supportvector_regressor.fit(X_train,y_train)
print_error(X_test,y_test,linear_regression)


# In[19]:


# Artificial Neural Network
import keras
import tensorflow as tf
from keras.layers import Dense
ann = keras.models.Sequential([
                                Dense(6, activation="relu",
                                input_shape=X_train.shape[1:]),
                                Dense(6,activation="relu"),
                                Dense(1)
                                ])

optimizer=tf.keras.optimizers.Adam()
loss=keras.losses.mean_squared_error
ann.compile(optimizer=optimizer,loss=loss,metrics=["mean_squared_error"])

history=ann.fit(X_train,y_train,epochs=100)

ann.summary()

print_error(X_test,y_test,ann)


# In[20]:


#Saving Scikitlearn models
import joblib
joblib.dump(decision_tree, "decisiontree_youtubeadview.pkl")

# Saving Keras Artificial Neural Network model
ann.save("ann_youtubeadview.h5")


# In[21]:


#Testing
data_test = pd.read_csv(r"C:\Users\Shaurya\Desktop\ML project\test.csv")


# In[22]:


data_test.head()


# In[23]:


from keras.models import load_model
model = load_model(r"C:\Users\Shaurya\ann_youtubeadview.h5")


# In[24]:


# Removing character "F" present in data
data_test=data_test[data_test.views!='F']
data_test=data_test[data_test.likes!='F']
data_test=data_test[data_test.dislikes!='F']
data_test=data_test[data_test.comment!='F']


# In[25]:


data_test.head()


# In[26]:


# Assigning each category a number for Category feature
category={'A': 1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}
data_test["category"]=data_test["category"].map(category)
data_test.head()


# In[27]:


# Convert values to integers for views, likes, comments, dislikes and adview
data_test["views"] = pd.to_numeric(data_test["views"])
data_test["comment"] = pd.to_numeric(data_test["comment"])
data_test["likes"] = pd.to_numeric(data_test["likes"])
data_test["dislikes"] = pd.to_numeric(data_test["dislikes"])
column_vidid=data_test['vidid']

# Endoding features like Category, Duration, Vidid
from sklearn.preprocessing import LabelEncoder
data_test['duration']=LabelEncoder().fit_transform(data_test['duration'])
data_test['vidid']=LabelEncoder().fit_transform(data_test['vidid'])
data_test['published']=LabelEncoder().fit_transform(data_test['published'])
data_test.head()


# In[28]:


# Convert Time_in_sec for duration
import datetime
import time
def checki(x):
  y = x[2:]
  h = ''
  m = ''
  s = ''
  mm = ''
  P = ['H','M','S']
  for i in y:
    if i not in P:
      mm+=i
    else:
      if(i=="H"):
        h = mm
        mm = ''
      elif(i == "M"):
        m = mm
        mm = ''
      else:
        s = mm
        mm = ''
  if(h==''):
    h = '00'
  if(m == ''):
    m = '00'
  if(s==''):
    s='00'
  bp = h+':'+m+':'+s
  return bp

train=pd.read_csv(r"C:\Users\Shaurya\Desktop\ML project\test.csv")
mp = pd.read_csv(r"C:\Users\Shaurya\Desktop\ML project\test.csv")["duration"]
time = mp.apply(checki)

def func_sec(time_string):
  h, m, s = time_string.split(':')
  return int(h) * 3600 + int(m) * 60 + int(s)

time1=time.apply(func_sec)

data_test["duration"]=time1
data_test.head()


# In[29]:


data_test=data_test.drop(["vidid"],axis=1)
data_test.head()


# In[30]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_test = data_test
X_test=scaler.fit_transform(X_test)


# In[31]:


prediction = model.predict(X_test)


# In[32]:


prediction=pd.DataFrame(prediction)
prediction.info()


# In[33]:


prediction = prediction.rename(columns={0: "Adview"})


# In[34]:


prediction.head()


# In[35]:


prediction.to_csv('predictions.csv')


# In[ ]:




