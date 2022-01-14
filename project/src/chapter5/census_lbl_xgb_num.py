import pandas as pd

from sklearn import metrics
from sklearn import preprocessing

from chapter5 import config
from chapter5 import model_dispatcher

from common import utils


def run(fold):
    df = pd.read_csv(config.CENSUS_FILE_FULL_FOLDS)

    # 数値を含む列
    num_cols = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]

    # 目的変数を変換
    target_mapping = {"<=50K": 0, ">50K": 1}
    df.loc[:, "income"] = df["income"].map(target_mapping)

    # 数値以外の列のみ欠損値補完
    ftrs = utils.exclude_cols_from_df(df, ("kfold", "income"))
    for col in ftrs:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # ラベルエンコード
    # one hot エンコードに対し決定木系は時間がかかるため
    for col in ftrs:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])

    # 引数と一致しない番号を学習に、さもなくば検証に利用
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    x_train = df_train[ftrs].values
    x_valid = df_valid[ftrs].values

    # 学習
    mdl = model_dispatcher.models["xgb"](n_jobs=-1)
    mdl.fit(x_train, df_train.income.values)

    # AUCを計算
    # predict_proba で [[クラス「0」の確率、クラス「1」の確率]] の配列を取得できる
    valid_preds = mdl.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    print(f"Fold={fold}, AUC={auc}")


if __name__ == "__main__":
    for i in range(5):
        run(i)
