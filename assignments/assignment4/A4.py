from my_KNN import my_KNN
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":
    #  Load training data
    data_train = pd.read_csv("../data/Iris_train.csv")
    # Separate independent variables and dependent variables
    independent = ["SepalLengthCm",	"SepalWidthCm",	"PetalLengthCm","PetalWidthCm"]
    X = data_train[independent]
    y = data_train["Species"]
    # Train model
    #clf = KNeighborsClassifier(algorithm="brute", metric="minkowski")
    clf = my_KNN()
    clf.fit(X,y)
    # Load testing data
    data_test = pd.read_csv("../data/Iris_test.csv")
    X_test = data_test[independent]
    # Predict
    predictions = clf.predict(X_test)
    # Predict probabilities
    probs = clf.predict_proba(X_test)
    # Print results
    for i,pred in enumerate(predictions):
        print("%s\t%f" % (pred, probs[pred][i]))
        #print("%s\t%f" % (pred, np.max(probs[i])))
