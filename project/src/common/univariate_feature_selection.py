from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)


class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type: str, scoring: str) -> None:
        """sklearn の複数の手法に対応した単変量特徴量選択のためのラッパー

        :param n_features: float 型（1~0を想定）の場合 SelectPercentile、さもなくば SelectKBest を利用
        :param problem_type: 分類か回帰
        :param scoring: 手法名
        """
        # 分類か回帰かに応じて対応手法を設定
        if problem_type == "classification":
            valid_scoring = {
                "f_classif": f_classif,
                "chi2": chi2,
                "mutual_info_classif": mutual_info_classif,
            }
        else:
            valid_scoring = {
                "f_regression": f_regression,
                "mutual_info_regression": mutual_info_regression,
            }
        # 手法が対応してない場合
        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")

        # 選択したい特徴量の量の型に応じて適切なクラスを使用
        if isinstance(n_features, int):
            self.selection = SelectKBest(valid_scoring[scoring], k=n_features)
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring], percentile=int(n_features * 100)
            )
        else:
            raise Exception("Invalid type of n_features")

        def fit(self, X, y):
            return self.selection.fit(X, y)

        def transform(self, X):
            return self.selection.transform(X)

        def fit_transform(self, X, y):
            return self.selection.fit_transform(X, y)
