import os,sys
from sklearn import datasets
import copy
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from six.moves import urllib
from sklearn.datasets import fetch_mldata
import numpy as np
def main():
    print("Could not download MNIST data from mldata.org, trying alternative...")
    
    # Alternative method to load MNIST, if mldata.org is down
    from scipy.io import loadmat
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"
    if not os.path.exists(mnist_path):
        response = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_path, "wb") as f:
            content = response.read()
            f.write(content)
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    print("Success!")

    X_all, y_all = mnist['data']/255., mnist['target']
    index_array = range(len(y_all))
    np.random.shuffle(index_array)
    X_all_new = copy.deepcopy(X_all)
    y_all_new = copy.deepcopy(y_all)
    for i in range(len(index_array)):
      X_all_new[i] = X_all[index_array[i]]
      y_all_new[i] = y_all[index_array[i]]
    X_all = X_all_new
    y_all = y_all_new
    print("scaling")
    param1 = 10000
    param2 = 20000
    X = X_all[:param1, :]
    y = y_all[:param1]

    X_test = X_all[param1:param2, :]
    y_test = y_all[param1:param2]

    svm = SVC(cache_size=param1, kernel='rbf', tol=0.01)

    parameters = {'C':10. ** np.arange(5,10), 'gamma':2. ** np.arange(-5, -1)}
    print("grid search")
    grid = GridSearchCV(svm, parameters, cv=StratifiedKFold(y, 5), verbose=3, n_jobs=-1)
    grid.fit(X, y)
    print("predicting")
    print "score: ", grid.score(X_test, y_test)
    print grid.best_estimator_

if __name__ == "__main__":
    main()
