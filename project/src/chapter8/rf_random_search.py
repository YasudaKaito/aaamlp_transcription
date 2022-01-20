import numpy as np
import pandas as pd
from sklearn import ensemble, model_selection

if __name__ == "__main__":
    df = pd.read_csv("../../input/mobile_train.csv")
    # 特徴量に price_range 以外のすべての列を使用
    X = df.drop("price_range", axis=1).values
    y = df["price_range"].values

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    # パラメタの探索範囲
    param_grid = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 31),
        "criterion": ["gini", "entropy"],
    }

    # ランダムサーチ
    # param_distributions がパラメータのリストの辞書の場合、
    # 非復元無作為抽出を実施
    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=20,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5,
    )

    model.fit(X, y)
    print(f"Best score: {model.best_score_}")
    print(f"Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")