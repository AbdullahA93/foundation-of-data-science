import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import time
sys.path.insert(0, '..')
from assignment8.my_evaluation import my_evaluation


class my_model():
    def fit(self, X, y):
        data_set = X.iloc[:, :4]
        str_regex = data_set.replace({r'[^a-zA-Z]': ' '}, regex=True, inplace=False)
        clean = str_regex.apply(lambda d: d.str.encode('ascii', 'ignore').str.decode('ascii').str.lower())
        # do not exceed 29 mins
        X2 =clean[['description','requirements','title']].apply(lambda x: ''.join(x), axis=1)
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False , ngram_range=(1,3))
        XX = self.preprocessor.fit_transform(X2)
        self.clf = LinearSVC(class_weight="balanced")
        self.clf.fit(XX, y)

        return

    def predict(self, X):
        data_set = X.iloc[:, :4]
        str_regex = data_set.replace({r'[^a-zA-Z]': ' '}, regex=True, inplace=False)
        clean = str_regex.apply(lambda d: d.str.encode('ascii', 'ignore').str.decode('ascii').str.lower())
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X2 =clean[['description','requirements','title']].apply(lambda x: ''.join(x), axis=1)
        XX = self.preprocessor.transform(X2)
        predictions = self.clf.predict(XX)

        return predictions


def split(data):
    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=False, train_size=0.75)
    clf = my_model()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    f1 = f1_score(predictions,y_test)
    return f1


if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("../data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    f1 = split(data)
    print("F1 score: %f" % f1)
    runtime = (time.time() - start) / 60.0
    print('runtime :' ,runtime)
