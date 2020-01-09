#!/bin/bash
set -eux
python3 model_baseline.py predict
python3 model_clean.py predict
python3 averaging.py
