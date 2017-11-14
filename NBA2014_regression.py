
# coding: utf-8

# In[ ]:

get_ipython().magic('matplotlib inline')


# #Imports

# In[ ]:

import os
import csv
import pandas as pd
import numpy as np

from scipy import stats, integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# ###import visualization tools

# In[ ]:

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


# ###upload CSV into a dataframe named nba

# In[ ]:

nba_data = pd.read_csv("https://raw.githubusercontent.com/georgetown-analytics/nba/master/fixtures/nba_players.csv")


# ###display number of rows and columns

# In[ ]:

nba_data = pd.DataFrame(nba_data)
nba_data.head()


# ###display the first row of data

# In[ ]:

nba_data.head(1)


# ###display rows and columns using shape

# In[ ]:

nba_data.shape


# ###run descriptive statistics

# In[ ]:

nba_data.describe()


# ###visualization

# In[ ]:

scatter_matrix(nba_data, alpha=0.2, figsize=(18,18), diagonal='kde')
plt.show()


# In[ ]:

sns.swarmplot(x="HT", y="REBR", data=nba_data)


# ###initialize linear regression model

# In[ ]:

regression_model = linear_model.LinearRegression()
regression_model.fit(X = pd.DataFrame(nba_data["VA"]), 
                     y = nba_data["SALARY"])

print(regression_model.intercept_)
print(regression_model.coef_)


# ###variance in the response variable is explained by the model using the model.score() function

# In[ ]:

regression_model.score(X = pd.DataFrame(nba_data["VA"]), 
                       y = nba_data["SALARY"])


# ###plot line of best fit

# In[ ]:

nba_data.plot(kind="scatter",
           x="VA",
           y="SALARY",
           figsize=(9,9),
           color="black",
           xlim = (-100, 910))

plt.plot(nba_data["VA"], train_prediction,  color="blue")    

