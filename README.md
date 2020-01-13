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
./predict.sh
```

Dockerを使うなら、

```bash
docker run --gpus=all --rm --interactive --tty --volume=$PWD:/usr/src/app keras-docker:0.1.0 ./predict.sh
```

`models/averaging/submission.csv` が最終結果。

### 動作確認済み環境

- GTX 1080 ×2
- Ubuntu 18.04.3 LTS
- Python 3.7.4
- albumentations==0.4.3
- better-exceptions==0.2.2
- numba==0.47.0
- numpy==1.18.1
- pandas==0.25.3
- scikit-learn==0.22.1
- scipy==1.4.1
- tensorflow==2.1.0
- tqdm==4.41.1

### 学習済みモデル

<https://github.com/ak110/probspace_ukiyoe/releases/download/v1.0/20200112_probspace_ukiyoe.tar.bz2>
