# 浮世絵作者予測

<https://prob.space/competitions/ukiyoe-author>

## data

```bash
mkdir data
pushd data
wget https://probspace-stg.s3.amazonaws.com/problems/39/data/37/ukiyoe-train-labels.npz
wget https://probspace-stg.s3.amazonaws.com/problems/39/data/38/ukiyoe-test-imgs.npz
wget https://probspace-stg.s3.amazonaws.com/problems/39/data/39/ukiyoe-train-imgs.npz
popd
```

## 学習手順

```bash
./model_baseline.py
./model_mixup.py
```

## 推論手順

```bash
./model_baseline.py predict
./model_mixup.py predict
./averaging.py
```

`models/averaging/submission.csv` が最終結果。2GPUで20分くらい。

