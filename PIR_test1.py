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
Scenario_1 = np.zeros([301]);
Scenario_2 = np.zeros([301]);
Scenario_3 = np.zeros([301]);
Scenario_4 = np.zeros([301]);
Scenario_5 = np.zeros([301]);
Scenario_6 = np.zeros([301]);
Scenario_7 = np.zeros([301]);
Scenario_8 = np.zeros([301]);
Scenario_9 = np.zeros([301]);

Scenarios = {Scenario_1, Scenario_2, Scenario_3, Scenario_4, Scenario_5, Scenario_6, Scenario_7, Scenario_8, Scenario_9};

k = 0;
for j in range(9):
    k += 1;
    for i in range(10):
          Scenarios['Scenario_ + str(j)'] = np.column_stack((Scenarios['Scenario_ + str(j)'], A[9*k*301*i : 9*k*301*i + 301]));
          
          
          
          
          
          
          
          
                                         