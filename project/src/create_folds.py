from argparse import ArgumentParser
import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    psr = ArgumentParser()
    psr.add_argument("--input", type=str)
    psr.add_argument("--output", type=str)
    psr.add_argument("--target_name", type=str)
    psr.add_argument("--n_splits", type=int)
    args_ = psr.parse_args()

    df = pd.read_csv(args_.input)
    df["kfold"] = -1
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    # 目的変数
    y = df[args_.target_name].values
    # 層化抽出
    kf = model_selection.StratifiedKFold(n_splits=args_.n_splits)
    # kfold 列を埋める
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f
    # 保存
    df.to_csv(args_.output, index=False)
