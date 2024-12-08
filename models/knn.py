##############################
#    K Nearest Neighbours    #
##############################
import pandas as pd
import numpy as np
import sklearn
import sys
import os



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import preprocessing as pp


class KNearestNeighbours:
    def __init__(self, k, x_train, x_validation, y_train, y_validation):
        # get data from preprocessing
        self.k = k
        self.x_train = np.array(x_train)
        self.x_validation = np.array(x_validation)
        self.y_train = np.array(y_train)
        self.y_train = self.y_train.reshape(-1,1)
        self.y_validation = np.array(y_validation)
        self.y_validation = self.y_validation.reshape(-1,1)
        print (self.x_train.shape)
        print (self.y_train.shape)

        #concatenate together x and y, last label is classification
        self.data = np.hstack((self.x_train,self.y_train))
        print(self.data.shape)

        self.validation_data = np.hstack((self.x_validation,self.y_validation))

    def euclidean_distance (self, x1, x2): # calculates the euclidean distance
        return np.sqrt(np.sum(np.square(x1-x2)))
    
    def find_distances (self,x):
        print (type(x))
        distances = []
        for sample in self.data:
            distance = self.euclidean_distance(sample[:21],x[:21])
            distances.append((distance,sample[21])) # add distance and class to list
        return distances

    def classify (self,x):
        # get the distances of all data points and their class 
        distances = self.find_distances(x)
        # sort based on distance
        distances.sort(key= lambda x: x[0])

        # get majority of first k classes 
        classes = []
        for i in range (self.k):
            classes.append(distances[i][1])

        # get count of 1s and 0s
        ones = classes.count(1.0)
        zeros = classes.count(0.0)

        if ones > zeros:
            return 1.0
        else:
            return 0.0
    

def run_knn():
    x_train, x_validation, y_train, y_validation = pp.preprocessing()
    model = KNearestNeighbours(10,x_train, x_validation, y_train, y_validation)
    print (model.classify(model.validation_data[0, :]))

run_knn()

'''
julia notes:
- try out KD trees or ball trees
- feature selection
- evaluation needed 
'''