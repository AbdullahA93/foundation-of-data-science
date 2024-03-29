import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X = X
        self.y = y


        return

    def dist(self,x):
        distances=[]
        # Calculate distances of training data to a single input data point (np.array)
        if self.metric == "minkowski":
         for i in range(len(self.X)):
             arr =0.0
             for j in range(len(x)):
                 arr+=((abs(self.X.iloc[i][j]-x[j]))**self.p)**(1/self.p)
             distances.append(arr)
         return distances

        elif self.metric == "euclidean":
         for i in range(len(self.X)):
             arr = 0.0
             for j in range(len(x)):
                arr+=np.square(self.X.iloc[i][j]-x[j])
             distances.append(np.sqrt(arr))
         return distances

        elif self.metric == "manhattan":
         for i in range(len(self.X)):
             arr = 0.0
             for j in range(len(x)):
                 arr+=(abs(self.X.iloc[i][j] - x[j]))
             distances.append(arr)
         return distances

        elif self.metric == "cosine":
            for i in range(len(self.X)):
                v1,v2,den = 0,0,0
                for j in range(len(x)):
                    v1 +=x[j] ** 2
                    v2 += self.X.iloc[i][j] ** 2
                    den += x[j] *self.X.iloc[i][j]
                distances.append(1-(den/ (np.sqrt(v1)*np.sqrt(v2))))
            return distances

        else:
            raise Exception("Unknown criterion.")
        return distances

    def k_neighbors(self,x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors)
        distances = self.dist(x)
        list=[]
        output = []
        for i in range(len(self.X)):
            list.append((i,self.y[i], distances[i]))
        list.sort(key=lambda tuple: tuple[2])
        for j in range(self.n_neighbors):
            output.append(list[j][1])
        output = Counter(output)
        return output

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")

        for x in X_feature.to_numpy():
            neighbors = self.k_neighbors(x)
            probs.append({key: neighbors[key] / float(self.n_neighbors) for key in self.classes_})
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs



