import io
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model, metrics, model_selection


def load_vectors(fname) -> Dict[str, List[float]]:
    """単語とその値のベクトルの辞書を返却"""
    # https://fasttext.cc/docs/en/english-vectors.html
    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def sentence_to_vec(
    s: str,
    embedding_dict: Dict[str, List[float]],
    tokenizer,
    stop_words: Optional[Iterable] = None,
):
    """文全体（単語ベクトルの総和を正規化したもの）の埋め込み表現を返す

    :param s: 文章
    :param embedding_dict: 単語の埋め込み表現の辞書
    :param tokenizer:
    :param stop_words: ストップワードのリスト, defaults to None
    :return: 大きさが1の、(1, 300)の形状の文全体の埋め込み表現ベクトル
    """
    # 小文字に
    words = str(s).lower()
    words: Iterable[str] = tokenizer(words)
    # ストップワードがあれば除去
    if stop_words:
        words = [w for w in words if not w in stop_words]
    # すべての文字が英字のトークンのみ残す
    words = [w for w in words if w.isalpha()]

    # 埋め込み表現を格納するリスト
    M = []
    for w in words:
        # すべての単語について埋め込み表現を獲得
        if w in embedding_dict:
            M.append(embedding_dict[w])
    # すべての単語が語彙にない場合
    if not M:
        return np.zeros(300)

    # numpy に変換
    M = np.asarray(M)
    # (1, 300) の行列
    v = M.sum(axis=0)
    # 正規化（大きさが1のベクトルにする）
    # ベクトルa を単位ベクトル化するには ベクトルa/ベクトルaの大きさ（各成分の平方和のルートを取る）
    return v / np.sqrt((v ** 2).sum())


if __name__ == "__main__":
    df = pd.read_csv("../../input/imdb.csv")
    # 肯定的、否定的を1、0に置換
    df.sentiment = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    # サンプルをシャッフル
    df = df.sample(frac=1).reset_index(drop=True)

    # 埋め込み表現の読み込み
    print("Loading embeddings")
    embeddings = load_vectors("../../input/wiki-news-300d-1M.vec")

    # 文全体の埋め込み表現作成
    print("Creating sentence vectors")
    vectors = []
    for review in df.review.values:
        vectors.append(
            sentence_to_vec(
                s=review, embedding_dict=embeddings, tokenizer=word_tokenize
            )
        )
    # (データ数, 300)の行列
    vectors = np.asarray(vectors)

    # 目的変数
    y = df.sentiment.values
    # 層化抽出
    kf = model_selection.StratifiedKFold(n_splits=5)
    # 訓練インデックス、検証インデックスが入ってくる
    for fold_, (t_, v_) in enumerate(kf.split(X=vectors, y=y)):
        print(f"Training fold: {fold_}")
        # 学習用と評価用に
        xtrain = vectors[t_, :]
        ytrain = y[t_]
        xtest = vectors[v_, :]
        ytest = y[v_]

        model = linear_model.LogisticRegression()
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        # 正答率を使う
        acc = metrics.accuracy_score(ytest, preds)
        print(f"Acc = {acc}")
        print("")
