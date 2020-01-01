#!/usr/bin/env python3
import pathlib

import numpy as np
import sklearn.linear_model

import _data
import pytoolkit as tk

num_classes = 10
model_names = [
    # "model_baseline",
    # "model_gap",
    "model_clean",
    "model_clean2",
    "model_cutmix",
    # "model_large",
]
nfold = 5
split_seed = 99
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


@app.command(then="validate")
def train():
    train_set = _data.load_train_data()
    train_set.data = np.concatenate(
        [tk.utils.load(f"models/{n}/pred_train.pkl") for n in model_names], axis=-1
    )
    folds = tk.validation.split(train_set, nfold, stratify=True, split_seed=split_seed)
    model = create_model()
    model.cv(train_set, folds)
    # tk.notifications.post_evals(evals)


@app.command(then="predict")
def validate():
    train_set = _data.load_train_data()
    train_set.data = np.concatenate(
        [tk.utils.load(f"models/{n}/pred_train.pkl") for n in model_names], axis=-1
    )
    folds = tk.validation.split(train_set, nfold, stratify=True, split_seed=split_seed)
    model = create_model().load(models_dir)
    pred = model.predict_oof(train_set, folds)
    _data.save_oofp(models_dir, train_set, pred)


@app.command()
def predict():
    test_set = _data.load_test_data()
    test_set.data = np.concatenate(
        [tk.utils.load(f"models/{n}/pred_test.pkl") for n in model_names], axis=-1
    )
    model = create_model().load(models_dir)
    pred_list = model.predict_all(test_set)
    pred = np.mean(pred_list, axis=0)
    _data.save_prediction(models_dir, test_set, pred)


def create_model():
    if True:
        return tk.pipeline.SKLearnModel(
            sklearn.linear_model.LogisticRegression(n_jobs=-1),
            nfold=nfold,
            models_dir=models_dir,
            predict_method="predict_proba",
        )
    else:
        return tk.pipeline.LGBModel(
            params={
                "num_class": num_classes,
                "objective": "multiclass",
                "metric": ["multi_logloss", "multi_error"],
                "nthread": -1,
                "verbosity": -1,
                # "feature_fraction": 0.8,
                # "bagging_freq": 0,
            },
            nfold=nfold,
            models_dir=models_dir,
            seeds=[1],
        )


if __name__ == "__main__":
    app.run(default="train")
