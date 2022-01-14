import copy

import pandas as pd
from sklearn import metrics, preprocessing

from chapter5 import config, model_dispatcher


def mean_target_encoding(data: pd.DataFrame) -> pd.DataFrame:
    """ターゲットエンコーディングを行う

    :param data: 学習・検証両方入ったデータセット
    :return: 各 fold で学習されたカテゴリごとの平均値を追加の列として持つデータセット
    """
    # ディープコピー
    df = copy.deepcopy(data)

    # 数値を含む列
    num_cols = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]
    # 目的変数を変換
    target_mapping = {"<=50K": 0, ">50K": 1}
    df.loc[:, "income"] = df["income"].map(target_mapping)

    ftrs = [c for c in df.columns if c not in num_cols and c not in ("kfold", "income")]
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

    encoded_dfs = []
    for fold in range(5):
        # 引数と一致しない番号を学習に、さもなくば検証に利用
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        for col in ftrs:
            # {カテゴリ: 目的変数の平均} の辞書を作成
            mapping_dict = dict(df_train.groupby(col)["income"].mean())
            # もとの列名と suffix で新しい列を作成
            df_valid.loc[:, col + "_enc"] = df_valid[col].map(mapping_dict)
        # リストに格納
        encoded_dfs.append(df_valid)
    # すべての検証データを結合してフルデータにして返却
    return pd.concat(encoded_dfs, axis=0)


def run(df: pd.DataFrame, fold):
    # 引数と一致しない番号を学習に、さもなくば検証に利用
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    ftrs = [f for f in df.columns if f not in ("kfold", "income")]
    x_train = df_train[ftrs].values
    x_valid = df_valid[ftrs].values

    # 学習
    mdl = model_dispatcher.models["xgb"](n_jobs=-1, max_depth=7)
    mdl.fit(x_train, df_train.income.values)

    # AUCを計算
    # predict_proba で [[クラス「0」の確率、クラス「1」の確率]] の配列を取得できる
    valid_preds = mdl.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    print(f"Fold={fold}, AUC={auc}")


if __name__ == "__main__":
    df = pd.read_csv(config.CENSUS_FILE_FULL_FOLDS)
    df = mean_target_encoding(df)
    for i in range(5):
        run(df, i)
