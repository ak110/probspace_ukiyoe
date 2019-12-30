#!/usr/bin/env python3
import pathlib

import numpy as np

import _data
import pytoolkit as tk

source_models = {"model_baseline": 1.0, "model_gap": 1.0, "model_large": 1.0, "model_xlarge": 1.0}
model_names, model_weights = zip(*source_models.items())
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


@app.command()
def predict():
    test_set = _data.load_test_data()
    pred = np.average(
        [
            tk.utils.load(pathlib.Path("models") / n / "pred_test.pkl")
            for n in model_names
        ],
        weights=model_weights,
        axis=0,
    )
    _data.save_prediction(models_dir, test_set, pred)


if __name__ == "__main__":
    app.run(default="predict")
