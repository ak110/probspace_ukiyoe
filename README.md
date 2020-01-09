# 浮世絵作者予測

<https://prob.space/competitions/ukiyoe-author>

## データの配置

```bash
mkdir data
pushd data
wget https://probspace-stg.s3.amazonaws.com/problems/39/data/37/ukiyoe-train-labels.npz
wget https://probspace-stg.s3.amazonaws.com/problems/39/data/38/ukiyoe-test-imgs.npz
wget https://probspace-stg.s3.amazonaws.com/problems/39/data/39/ukiyoe-train-imgs.npz
popd
```

## 学習

```bash
./model_baseline.py
./model_mixup.py
```

## 推論

```bash
docker run --gpus=all --rm --interactive --tty --volume=$PWD:/usr/src/app keras-docker:0.1.0 ./predict.sh
```

`models/averaging/submission.csv` が最終結果。
