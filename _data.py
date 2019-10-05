"""データの読み書きなど"""
import pathlib

import numpy as np

import pytoolkit as tk

data_dir = pathlib.Path(f"data")


def load_data():
    return load_train_data(), load_test_data()


def load_train_data():
    X_train = np.load(data_dir / "ukiyoe-train-imgs.npz")["arr_0"]
    y_train = np.load(data_dir / "ukiyoe-train-labels.npz")["arr_0"]
    return tk.data.Dataset(X_train, y_train)


def load_test_data():
    X_test = np.load(data_dir / "ukiyoe-test-imgs.npz")["arr_0"]
    return tk.data.Dataset(X_test)
