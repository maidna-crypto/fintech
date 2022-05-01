import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
import pandas as pd
from joblib import dump, load

data = 'Dataset.csv'

df = pd.read_csv(data)
X = df
X = np.array(X)
X.shape

kmeans = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
M=kmeans.fit(X)

label = kmeans.fit_predict(X)

X0=X[label==0, 0]
X1=X[label==1, 0]
X2=X[label==2, 0]

centroids=kmeans.cluster_centers_

#filter rows of original data
y0 = X[label == 0]
y1 = X[label == 1]
y2 = X[label == 2]


y1.shape
#plotting the results:

plt.scatter(X[label==0, 0], y0,  c='red', label ='Cluster 0')
plt.scatter(X[label==1, 0], y1,  c='blue', label ='Cluster 1')
plt.scatter(X[label==2, 0], y2,  c='green', label ='Cluster 2')    

plt.legend()
plt.show()

F=dump(M, 'Clusters.joblib') 