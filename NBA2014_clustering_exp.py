
# coding: utf-8

# In[ ]:

get_ipython().magic('matplotlib inline')


# #Import model family

# In[ ]:

import os
import requests
import warnings

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from sklearn.linear_model import Ridge
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

from sklearn import manifold
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from time import time


# ###import visualization tools

# In[ ]:

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


# ###load CSV into a dataframe

# In[ ]:

nba_data = pd.read_csv("https://raw.githubusercontent.com/georgetown-analytics/nba/master/fixtures/nba_players.csv")


# ###display the number of rows and columns

# In[ ]:

nba_data = pd.DataFrame(nba_data)
nba_data.head()


# ###display rows and columns using shape

# In[ ]:

nba_data.shape


# ###run descriptives

# In[ ]:

nba_data.describe()


# ###define features and labels

# In[ ]:

nba_data_features = nba_data.iloc[:, 3:17]
nba_data_labels = nba_data.iloc[:,-1:]
print(nba_data_features.head())
print(nba_data_labels.head())


# ###predictive models

# In[ ]:

pSalary_labels = nba_data.iloc[:,-1:]


# In[ ]:

splits = cv.train_test_split(nba_data_features, pSalary_labels, test_size=0.2)
X_train, X_test, y_train, y_test = splits


# ###ridge regression

# In[ ]:

model = Ridge(alpha=0.1)
model.fit(X_train, y_train)

expected = y_test
predicted = model.predict(X_test)

print("Ridge Regression model")
print("Mean Squared Error: %0.3f" % mse(expected, predicted))
print("Coefficient of Determination: %0.3f" % r2_score(expected, predicted))


# ###linear regression

# In[ ]:

model = LinearRegression()
model.fit(X_train, y_train)

expected = y_test
predicted = model.predict(X_test)

print("Linear Regression model")
print("Mean Squared Error: %0.3f" % mse(expected, predicted))
print("Coefficient of Determination: %0.3f" % r2_score(expected, predicted))


# ###kneighbors regression

# In[ ]:

model = KNeighborsRegressor()
model.fit(X_train, y_train)

expected  = y_test
predicted = model.predict(X_test)

# Evaluate fit of the model
print("Mean Squared Error: %0.3f" % mse(expected, predicted))
print("Coefficient of Determination: %0.3f" % r2_score(expected, predicted))


# ###define X and y [Kmeans clustering]

# In[ ]:

X = nba_data_features
y = pSalary_labels


# ###using the "elbow method to determine the right number of clusters
# ####code adapted from https://www.packtpub.com/books/content/clustering-k-means

# In[ ]:

K = range(1,10)
meandistortions = []

for k in K:

    elbow = KMeans(n_clusters=k, n_jobs=-1, random_state=1)
    elbow.fit(X)
    meandistortions.append(sum(np.min(euclidean_distances(X, elbow.cluster_centers_), axis=1)) / X.shape[0])

    
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()


# ###Higher Silhouette Coefficient score relates to a model with better defined clusters

# In[ ]:

kmeans = KMeans(n_clusters=4, n_jobs=-1, random_state=1)
kmeans.fit(X)
labels = kmeans.labels_
silhouette_score(X, labels, metric='euclidean')


# In[ ]:

kmeans = KMeans(n_clusters=2, n_jobs=-1, random_state=1)
kmeans.fit(X)
labels = kmeans.labels_
silhouette_score(X, labels, metric='euclidean')


# ###visualize silhouette plot

# In[ ]:

def silhouette_plot(X, range_n_clusters = range(2, 12, 2)):

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1
        ax1.set_xlim([-.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values =                 sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()


# In[ ]:

silhouette_plot(X)

