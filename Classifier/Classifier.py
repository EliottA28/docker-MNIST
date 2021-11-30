# Author: Issam H. Laradji
#         Arnaud Joly <arnaud.v.joly@gmail.com>
# License: BSD 3 clause

import os
from time import time
import argparse
import numpy as np
from joblib import Memory

import joblib

from sklearn.datasets import fetch_openml
from sklearn.datasets import get_data_home
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import zero_one_loss
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import cv2

# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory = Memory(os.path.join(get_data_home(), "mnist_benchmark_data"), mmap_mode="r")


# @memory.cache
def load_data(dtype=np.float32, order="F"):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    # Load dataset
    print("Loading dataset...")
    data = fetch_openml("mnist_784")
    X = check_array(data["data"], dtype=dtype, order=order)
    y = data["target"]

    # Normalize features
    X = X / 255

    # Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test


ESTIMATOR = {
    "MultilayerPerceptron": MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=400,
        alpha=1e-4,
        solver="sgd",
        learning_rate_init=0.2,
        momentum=0.9,
        verbose=1,
        tol=1e-4,
        random_state=1,
    )
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-jobs",
        nargs="?",
        default=1,
        type=int,
        help=(
            "Number of concurrently running workers for "
            "models that support parallelism."
        ),
    )
    parser.add_argument(
        "--order",
        nargs="?",
        default="C",
        type=str,
        choices=["F", "C"],
        help="Allow to choose between fortran and C ordered data",
    )
    parser.add_argument(
        "--random-seed",
        nargs="?",
        default=0,
        type=int,
        help="Common seed used by random number generator.",
    )
    args = vars(parser.parse_args())

    X_train, X_test, y_train, y_test = load_data(order=args["order"])
    
    # for i in range(10):
    #     cv2.imwrite("{}.JPG".format(i), 255*X_test[i].reshape((28,28)))

    print("")
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    print(
        "%s %d (size=%dMB)"
        % (
            "number of train samples:".ljust(25),
            X_train.shape[0],
            int(X_train.nbytes / 1e6),
        )
    )
    print(
        "%s %d (size=%dMB)"
        % (
            "number of test samples:".ljust(25),
            X_test.shape[0],
            int(X_test.nbytes / 1e6),
        )
    )

    print()
    print("Training Classifiers")
    print("====================")
    
    print("Training ... \n")
    estimator = ESTIMATOR["MultilayerPerceptron"]
    estimator_params = estimator.get_params()

    estimator.set_params(
        **{
            p: args["random_seed"]
            for p in estimator_params
            if p.endswith("random_state")
        }
    )

    if "n_jobs" in estimator_params:
        estimator.set_params(n_jobs=args["n_jobs"])

    time_start = time()
    estimator.fit(X_train, y_train)
    train_time = time() - time_start

    time_start = time()
    y_pred = estimator.predict(X_test)
    test_time = time() - time_start

    error = zero_one_loss(y_test, y_pred)

    print("done\n")

    joblib.dump(estimator, 'MNIST_classifier_MLP.joblib')