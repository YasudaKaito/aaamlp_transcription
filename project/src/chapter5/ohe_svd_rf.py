import pandas as pd

from scipy import sparse
from sklearn import decomposition
from sklearn import metrics
from sklearn import preprocessing

from chapter5 import config
from chapter5 import model_dispatcher

from common import utils


def run(fold):
    df = pd.read_csv(config.TRAINING_FILE)
    ftrs = utils.exclude_cols_from_df(df, ("id", "target", "kfold"))
    # すべて質的変数のデータなので、すべてのカラムの欠損値を同様に補完
    for col in ftrs:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # 引数と一致しない番号を学習に、さもなくば検証に利用
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()
    # 学習・検証両データを結合して one hot エンコードを学習
    full = pd.concat([df_train[ftrs], df_valid[ftrs]], axis=0)
    ohe.fit(full[ftrs])
    # one hot エンコード
    x_train = ohe.transform(df_train[ftrs])
    x_valid = ohe.transform(df_valid[ftrs])

    # 特異値分解. 120次元に圧縮
    svd = decomposition.TruncatedSVD(n_components=120)
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)
    x_train = svd.transform(x_train)
    x_valid = svd.transform(x_valid)

    # 学習
    mdl = model_dispatcher.models["rf"](n_jobs=-1)
    mdl.fit(x_train, df_train.target.values)

    # AUCを計算(target は 0 or 1)
    # predict_proba で [[クラス「0」の確率、クラス「1」の確率]] の配列を取得できる
    valid_preds = mdl.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    print(f"Fold={fold}, AUC={auc}")


if __name__ == "__main__":
    for i in range(5):
        run(i)
