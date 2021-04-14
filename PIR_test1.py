# synthetic classification dataset
"""from numpy import where
from sklearn.datasets import make_classification
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# create scatter plot for samples from each class
for class_value in range(2):
	# get row indexes for samples with this class
	row_ix = where(y == class_value)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()"""

#KMEANS CLUSTERING
import numpy as np
import pickle
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
#define dataset
#X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=10)

with open("data_train_100.dat", "rb") as f:
    train = pickle.load(f)
    f.close()
    
#print(train[0][0][0])

numpy_array = train[0][0]
numpy_array.shape

first_tuple_element = [a_tuple[0] for a_tuple in train]

table = np.array(first_tuple_element)

A = table[:,0] 
B = table[:,1]
C = table[:,2]
D = table[:,3]

"E = np.column_stack((A,B))  - mettre avec le temps"

S1 = A[0 :301];


"scenario 1:"
Scenario1 = np.zeros([301]);

#for j in range(9):
for i in range(100):
      Scenario1 = np.column_stack((Scenario1, A[9*301*i : 9*301*i + 301])) 
                                         