import os
import argparse

import config

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):
    df = pd.read_csv(config.TRAINING_FILE)
    # 引数の fold と一致しないデータを学習に利用
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # fold と一致するデータを検証に利用
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # 訓練／検証データの説明変数・目的変数を numpy 配列に変換
    x_train, y_train = _xy_from_df(df_train)
    x_valid, y_valid = _xy_from_df(df_valid)

    # 決定木
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    # 検証データを予測
    preds = clf.predict(x_valid)
    acc = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Acc={acc}")
    # モデル保存
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))

def _xy_from_df(df: pd.DataFrame):
    x = df.drop("label", axis=1).values
    y = df.label.values
    return x, y

if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument("--fold", type=int)
    args = psr.parse_args()
    run(fold=args.fold)