import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import metrics, model_selection, naive_bayes
from sklearn.feature_extraction.text import CountVectorizer

# IMDB をナイーブベイズで分類
if __name__ == "__main__":
    df = pd.read_csv("../../input/imdb.csv")
    # 肯定的、否定的を1、0に置換
    df.sentiment = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.sentiment.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    # kfold列をうめる
    for f, (_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    # 各分割
    for fold_ in range(5):
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)

        # tokenizer は nltk の word_tokenize
        count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
        count_vec.fit(train_df.review)
        # 疎行列化
        xtrain = count_vec.transform(train_df.review)
        xtest = count_vec.transform(test_df.review)

        model = naive_bayes.MultinomialNB()
        model.fit(xtrain, train_df.sentiment)

        # 評価用データに対する予測
        # 閾値は0.5
        preds = model.predict(xtest)
        acc = metrics.accuracy_score(test_df.sentiment, preds)
        print(f"Fold: {fold_}")
        print(f"Acc = {acc}")
        print("")
