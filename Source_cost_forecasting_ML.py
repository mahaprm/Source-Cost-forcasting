#!/usr/bin/env python
# coding: utf-8

# # Product forcasting

# In[26]:


#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import math
pd.set_option("display.max_columns", 101)
from sklearn import preprocessing, metrics, linear_model
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import calendar


# ## Data Wrangling & Visualization

# In[2]:


# The dataset is already loaded below
data = pd.read_csv("train.csv")


# In[3]:


data.head()


# In[4]:


#Explore columns
data.columns


# In[5]:


#Description
data.describe()


# In[6]:


#Showing shape of the dataframe
data.shape


# In[7]:


#Showing data types of dataframe
data.dtypes


# In[8]:


# check columns with data type object
col_list = [c for c in data.columns if data[c].dtype == 'object' and c != 'timestamp']
print(col_list)


# In[9]:


#Checking null are available in dataframe columns
data.isnull().sum()


# In[10]:


#pre-processing the data.

print(data['ProductType'].unique())
print(data['Manufacturer'].unique())
print(data['Area Code'].unique())
print(data['Sourcing Channel'].unique())
print(data['Product Size'].unique())
print(data['Product Type'].unique())
print(data['Month of Sourcing'].unique())


# In[11]:


data['source_year'] = data['Month of Sourcing'].str.split('-').str[1]
data['source_month'] = data['Month of Sourcing'].str.split('-').str[0]
data = data.drop(['Month of Sourcing'], axis=1)


# In[12]:


abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}


# In[13]:


le = preprocessing.LabelEncoder()
data['ProductType']  = le.fit_transform(data['ProductType'])
data['Manufacturer']  = le.fit_transform(data['Manufacturer'])
data['Area Code']  = le.fit_transform(data['Area Code'])
data['Sourcing Channel']  = le.fit_transform(data['Sourcing Channel'])
data['Product Size']  = le.fit_transform(data['Product Size'])
data['Product Type']  = le.fit_transform(data['Product Type'])
data['source_month'] = data['source_month'].apply(lambda x : abbr_to_num[x])
data['source_year'] = data['source_year'].astype('int32')


# ## Visualization, Modeling, Machine Learning

# In[14]:


#Loading Test data
test_data=pd.read_csv('test.csv')


# In[15]:


test_data.isnull().sum()
test_data.shape


# In[16]:


test_data.head()


# In[17]:


test_data['source_year'] = test_data['Month of Sourcing'].str.split('-').str[1]
test_data['source_month'] = test_data['Month of Sourcing'].str.split('-').str[0]
test_data = test_data.drop(['Month of Sourcing'], axis=1)


# In[18]:


# converting to categorical data as per train dataset
test_data['ProductType']  = le.fit_transform(test_data['ProductType'])
test_data['Manufacturer']  = le.fit_transform(test_data['Manufacturer'])
test_data['Area Code']  = le.fit_transform(test_data['Area Code'])
test_data['Sourcing Channel']  = le.fit_transform(test_data['Sourcing Channel'])
test_data['Product Size']  = le.fit_transform(test_data['Product Size'])
test_data['Product Type']  = le.fit_transform(test_data['Product Type'])
test_data['source_month'] = test_data['source_month'].apply(lambda x : abbr_to_num[x])
test_data['source_year'] = test_data['source_year'].astype('int32')


# In[19]:


test_data


# **Identifying the most important features of the model for management.**
# 

# In[20]:


corre_metrics = data[['ProductType', 'Manufacturer', 'Area Code', 'Sourcing Channel', 'Product Size', 'Product Type', 'Sourcing Cost']].corr()
corre_metrics


# In[21]:


sb.heatmap(corre_metrics, annot=True)
plt.show()


# In[22]:


# bar plots for categorical features
cols = [col for col in data.columns if 'Sourcing Cost' not in col]
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
for i, c in enumerate(cols):
    ax = axes.ravel()[i]
    sb.barplot(x=c, y="Sourcing Cost", ax=ax, data=data)


# Trying Linear Regression model

# In[29]:


#Split input variable and output variable.
X = data[[col for col in data.columns if 'Sourcing Cost' not in col]]
Y = data['Sourcing Cost']


# In[30]:


X


# In[31]:


Y


# In[32]:


#Spliting train test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#Reseting the index for trian and test
x_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

x_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


# In[33]:


# important features for random forest regression
# train random forest regression model
randomf = RandomForestRegressor(n_estimators=1000, max_depth=8)
randomf.fit(x_train, y_train)
for name, importance in zip(X.columns, randomf.feature_importances_):
    print('feature:', name, "=", importance)
importances = randomf.feature_importances_
indices = np.argsort(importances)
features = X.columns
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[34]:


x_train


# In[35]:


y_train


# In[36]:


# trying with linear regression

lr_model = linear_model.LinearRegression()
lr_model


# In[37]:


lr_model.fit(x_train, y_train)


# In[38]:


#print accuracy for model and coefficants.
lr = lr_model.score(x_train, y_train)
print('model accuracy', lr)
print('model coefficiants', lr_model.coef_)
print('model coefficiants', lr_model.intercept_)


# In[39]:


predict = cross_val_predict(lr_model, x_train, y_train, cv=3)
predict


# In[40]:


# Calculating r-sequard and mean sequared metrics to finalize the model.

r2_score = cross_val_score(lr_model, x_train, y_train, cv=3)
print('R2 score', np.average(r2_score))


# In[41]:


#model performance on test data.

lr_pred = lr_model.predict(x_test)
lr_pred


# In[42]:


rmse = math.sqrt(metrics.mean_squared_error(y_test, lr_pred))
mae = metrics.mean_absolute_error(y_test, lr_pred)
r2_score = metrics.r2_score(y_test, lr_pred)

print(r2_score)
print('rmse', rmse)
print('mae', mae)


# Trying DecissionTree Regresser

# In[43]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_leaf_nodes=10)
dtr


# In[44]:


dtr.fit(x_train, y_train)


# In[45]:


#checking accuracy of the model.
dtr_score = dtr.score(x_train, y_train)
dtr_score


# In[46]:


predict = cross_val_predict(dtr, x_train, y_train, cv=3)
predict


# In[47]:


# Calculating r-sequard and mean sequared metrics to finalize the model.

r2_score = cross_val_score(dtr, x_train, y_train, cv=3)
print('R2 score', np.average(r2_score))


# In[48]:


#model performance on test data.

dtr_pred = dtr.predict(x_test)

metrics.r2_score(y_test, dtr_pred)


# In[49]:


rmse = math.sqrt(metrics.mean_squared_error(y_test, dtr_pred))
mae = metrics.mean_absolute_error(y_test, dtr_pred)
r2_score = metrics.r2_score(y_test, dtr_pred)

print(r2_score)
print('rmse', rmse)
print('mae', mae)


# Trying randomForest Regresser

# In[50]:


from sklearn.ensemble import RandomForestRegressor
rfg = RandomForestRegressor(n_estimators=200)
rfg


# In[51]:


rfg.fit(x_train, y_train)
rfg_score = rfg.score(x_train, y_train)
rfg_score


# In[52]:


predict = cross_val_predict(rfg, x_train, y_train, cv=3)
predict
r2_score = cross_val_score(dtr, x_train, y_train, cv=3)
print('R2 score', np.average(r2_score))


# In[53]:


#model performance on test data.

rfg_pred = dtr.predict(x_test)
rfg_pred


# In[54]:


rmse = math.sqrt(metrics.mean_squared_error(y_test, rfg_pred))
mae = metrics.mean_absolute_error(y_test, rfg_pred)

print('rmse', rmse)
print('mae', mae)


# In[55]:


print(x_train.shape[1])
print(y_train)


# In[56]:


# Train XGBoost Regression
xgbr = XGBRegressor(objective='reg:squarederror', n_estimators=500, max_depth=4)
xgbr.fit(X,Y)


# In[57]:


xgbr_pred = xgbr.predict(test_data[cols])
xgbr_pred


# In[58]:


rmse = math.sqrt(metrics.mean_squared_error(test_data['Sourcing Cost'], xgbr_pred))
mae = metrics.mean_absolute_error(test_data['Sourcing Cost'], xgbr_pred)
r2_score = metrics.r2_score(test_data['Sourcing Cost'], xgbr_pred)

print(r2_score)
print('rmse', rmse)
print('mae', mae)


# In[59]:


from sklearn.neighbors import KNeighborsRegressor


# In[60]:


knnr = KNeighborsRegressor(n_neighbors = 8)
knnr.fit(X, Y)


# In[61]:


knn_pred = knnr.predict(test_data[cols])


# In[62]:


nrmse = math.sqrt(metrics.mean_squared_error(test_data['Sourcing Cost'], knn_pred))
mae = metrics.mean_absolute_error(test_data['Sourcing Cost'], knn_pred)
r2_score = metrics.r2_score(test_data['Sourcing Cost'], knn_pred)

print(r2_score)
print('rmse', rmse)
print('mae', mae)


# ### As compare to Machine learning algorithms XGBRegressor given better r2 squard value and less mae and rmse losses.
