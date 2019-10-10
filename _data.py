"""データの読み書きなど"""
import pathlib

import numpy as np
import pandas as pd

import pytoolkit as tk

data_dir = pathlib.Path(f"data")
logger = tk.log.get(__name__)


def load_data():
    """訓練データ・テストデータの読み込み"""
    return load_train_data(), load_test_data()


def load_train_data():
    """訓練データの読み込み"""
    X_train = np.load(data_dir / "ukiyoe-train-imgs.npz")["arr_0"]
    y_train = np.load(data_dir / "ukiyoe-train-labels.npz")["arr_0"]
    return tk.data.Dataset(X_train, y_train)


def load_test_data():
    """テストデータの読み込み"""
    X_test = np.load(data_dir / "ukiyoe-test-imgs.npz")["arr_0"]
    return tk.data.Dataset(X_test)


def save_oofp(models_dir, train_set, pred):
    """訓練データのout-of-fold predictionsの保存と評価"""
    if tk.hvd.is_master():
        tk.utils.dump(pred, models_dir / "pred_train.pkl")

        tk.evaluations.print_classification_metrics(train_set.labels, pred)


def save_prediction(models_dir, test_set, pred):
    """テストデータの予測結果の保存"""
    if tk.hvd.is_master():
        tk.utils.dump(pred, models_dir / "pred_test.pkl")

        df = pd.DataFrame()
        df["id"] = np.arange(1, len(test_set) + 1)
        df["y"] = pred.argmax(axis=-1)
        df.to_csv(models_dir / "submission.csv", index=False)
