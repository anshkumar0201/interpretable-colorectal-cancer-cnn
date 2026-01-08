"""
Entry point for training the colorectal cancer CNN.
"""

import numpy as np
from model.train import train_model

# Load data (example format)
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

train_model(X_train, y_train, X_test, y_test)
