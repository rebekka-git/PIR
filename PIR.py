#KMEANS CLUSTERING
import numpy as np
import pandas as pd
import pickle
from numpy import zeros,array
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot
from matplotlib import pyplot


with open("data_train_100_15_04.dat", "rb") as f:
    train = pickle.load(f)
    f.close()
    
#print(train[0][0][0])

numpy_array = train[0][0]
numpy_array.shape

first_tuple_element = [a_tuple[0] for a_tuple in train]

table = np.array(first_tuple_element)

Qp = table[:,0]
y1 = table[:,1]
y2 = table[:,2]
Up = table[:,3]
        
# temps2 = np.zeros(301);
# for i in range(0,301):
#     temps2[i] = i

# Scenario1_Qp = np.zeros([301]);
    
# for j in range(0,100):
#     Scenario1_Qp = np.column_stack((Scenario1_Qp, Qp[9*301*j : 9*301*j+301])) 
# #       Scenario1_y1 = np.column_stack((Scenario1_y1, y1[9*301*j : 9*301*j + 301])) 
# #       Scenario1_y2 = np.column_stack((Scenario1_y2, y2[9*301*j : 9*301*j + 301])) 
# #       Scenario1_Up = np.column_stack((Scenario1_Up, Up[9*301*j : 9*301*j + 301])) 
      
# Scenario1_Qp[:,0] = temps2

"scenario 1:"

temps = np.zeros(30100)
for j in range (0,100):
    for i in range(0,301):
        temps[i+j*301] = i
        
Scenario1_Qp = np.zeros([30100]);
Scenario1_y1 = np.zeros([30100]);
Scenario1_y2 = np.zeros([30100]);
Scenario1_Up = np.zeros([30100]);

for j in range (0,100):
    for i in range (0,301):
        Scenario1_Qp[301*j+i] = Qp[9*301*j+i];
        Scenario1_y1[301*j+i] = y1[9*301*j+i];
        Scenario1_y2[301*j+i] = y2[9*301*j+i];
        Scenario1_Up[301*j+i] = Up[9*301*j+i];

E = np.column_stack((temps,Scenario1_Qp))
     
        
model = DBSCAN(eps=0.0000001, min_samples=9)
# fit model and predict clusters
yhat = model.fit_predict(E)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(E[row_ix, 0], E[row_ix, 1])
# show the plot
pyplot.show()
