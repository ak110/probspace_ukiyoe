#!/usr/bin/env python3
import pathlib

import numpy as np

import _data
import pytoolkit as tk

source_models = {
    # "model_baseline": 1,
    # "model_gap": 1,
    "model_clean": 1,
    "model_cutmix": 0.5,
    "model_large2": 1,
    # "model_large": 1,
    # "model_xlarge": 1,
}
model_names, model_weights = zip(*source_models.items())
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


@app.command()
def predict():
    logger.info(f"source_models = {source_models}")
    test_set = _data.load_test_data()
    pred = np.average(
        [_load_pred(n) for n in model_names], weights=model_weights, axis=0,
    )
    _data.save_prediction(models_dir, test_set, pred)


def _load_pred(n):
    pred = tk.utils.load(pathlib.Path("models") / n / "pred_test.pkl")
    checks = np.abs(pred.sum(axis=-1) - 1)
    assert (checks < 1e-5).all(), f"value error: model={n} pred={pred} checks={checks}"
    return pred


if __name__ == "__main__":
    app.run(default="predict")
