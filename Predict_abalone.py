
# coding: utf-8

# #Predicting the number of rings on an Abalone

# In[ ]:

get_ipython().magic('matplotlib inline')


# ###import necessary imports

# In[ ]:

import os
import warnings
from decimal import Decimal
import requests
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# ###import models

# In[ ]:

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse


# ###upload data

# In[ ]:

Abalone_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"


# In[ ]:

def download_data(url, path='data'):
    if not os.path.exists(path):
        os.mkdir(path)

    response = requests.get(url)
    name = os.path.basename(url)
    with open(os.path.join(path, name), 'wb') as f:
        f.write(response.content)


# In[ ]:

download_data(Abalone_data)
Abalone_data


# In[ ]:

Abalone_data   = pd.read_csv('data/abalone.data', header=None)


# In[ ]:

Abalone_data.head()


# ###Identify columns

# In[ ]:

Abalone_data.columns = ['Sex','Length','Diameter','Height','Whole weight', 'Shucked weight', 
                        'Viscera weight','Shell weight','Rings']
print(Abalone_data.head())


# ###run descriptive statistics

# In[ ]:

Abalone_data.describe()


# ###visualization

# In[ ]:

scatter_matrix(Abalone_data, alpha=0.2, figsize=(18,18), diagonal='kde')
plt.show()


# ###transform binary column [first col]

# In[ ]:

encoder = LabelEncoder() 
Transform_col = encoder.fit_transform(Abalone_data['Sex']) 
Abalone_data['Sex'] = Transform_col


# ###identify features and labels

# In[ ]:

Abalone_data_features = Abalone_data.iloc[:, :-1]
Abalone_data_labels = Abalone_data.iloc[:,-1:]
print(Abalone_data_features.head())
print(Abalone_data_labels.head())


# ###predictive models

# In[ ]:

model = RandomizedLasso(alpha=0.01)
model.fit(Abalone_data_features, Abalone_data_labels["Rings"])
names = list(Abalone_data_features)

print("Features by their score:")
print(sorted(zip(map(lambda x: round(x, 4), model.scores_), names), reverse=True))


# In[ ]:

sring_labels = Abalone_data.iloc[:,-1:]


# In[ ]:

splits = tts(Abalone_data_features, sring_labels, test_size=0.2)
X_train, X_test, y_train, y_test = splits


# In[ ]:

model = Ridge(alpha=0.1)
model.fit(X_train, y_train)

expected = y_test
predicted = model.predict(X_test)

print("Ridge Regression model")
print("Mean Squared Error: %0.3f" % mse(expected, predicted))
print("Coefficient of Determination: %0.3f" % r2_score(expected, predicted))


# In[ ]:

model = LinearRegression()
model.fit(X_train, y_train)

expected = y_test
predicted = model.predict(X_test)

print("Linear Regression model")
print("Mean Squared Error: %0.3f" % mse(expected, predicted))
print("Coefficient of Determination: %0.3f" % r2_score(expected, predicted))


# ###use ravel() to change the shape of y to (n_samples,)

# In[ ]:

y = np.array(y_train)
y_train = np.ravel(y)


# In[ ]:

model = RandomForestRegressor()
model.fit(X_train, y_train)

expected = y_test
predicted = model.predict(X_test)

print("Random Forest model")
print("Mean squared error = %0.3f" % mse(expected, predicted))
print("R2 score = %0.3f" % r2_score(expected, predicted))

