import pandas as pd

from sklearn import metrics
from sklearn import preprocessing

from chapter5 import config
from chapter5 import model_dispatcher

from common import utils


def run(fold):
    df = pd.read_csv(config.CENSUS_FILE_FOLDS)

    # 目的変数を変換
    target_mapping = {"<=50K": 0, ">50K": 1}
    df.loc[:, "income"] = df["income"].map(target_mapping)

    ftrs = utils.exclude_cols_from_df(df, ("kfold", "income"))
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

    # 学習
    mdl = model_dispatcher.models["logres"]
    mdl.fit(x_train, df_train.income.values)

    # AUCを計算
    # predict_proba で下記のような、[[クラス「0」の確率、クラス「1」の確率]] の配列を取得できる
    # [[0.39864811 0.60135189]
    #  [0.96318742 0.03681258]
    #  [0.9632642  0.0367358 ]
    #  ...
    #  [0.93810959 0.06189041]
    #  [0.95024203 0.04975797]
    #  [0.97925932 0.02074068]]
    # print(mdl.predict_proba(x_valid))
    valid_preds = mdl.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    print(f"Fold={fold}, AUC={auc}")


if __name__ == "__main__":
    for i in range(5):
        run(i)
