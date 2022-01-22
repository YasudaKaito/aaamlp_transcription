from functools import partial

import numpy as np
# 参考: https://kenyu-life.com/2019/06/01/dhsm/
from scipy.optimize import fmin
from sklearn import metrics


class OptimizeAUC:
    """AUCを最適化するために複数のモデルの最適な重みを探索するクラス"""

    def __init__(self) -> None:
        self.coef_ = 0

    def _auc(self, coef, X, y):
        """AUCを計算する

        :param coef: 係数のリスト、要素数はモデル数と等しい
        :param X: 予測確率(0~1)、この場合は2次元配列（行数はデータ数, 列数はモデル数）
        :param y: 目的変数、この場合は二値の1次元配列
        """
        # 係数をそれぞれの列の予測値に掛け合わせる
        x_coef = X * coef
        # 行ごとに合計して、予測を計算
        predictions = np.sum(x_coef, axis=1)
        # AUCを計算
        auc_score = metrics.roc_auc_score(y, predictions)
        # AUCにマイナスをかける（最適化対象とするため負の値とする）
        return -1.0 * auc_score

    def fit(self, X, y):
        loss_partial = partial(self._auc, X=X, y=y)
        # ディリクレ分布で初期化
        # 合計は1が望ましい
        # 参考: https://stackoverflow.com/questions/59674211/what-does-numpy-random-dirichlet-do
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)

        # 損失関数を最小化
        self.coef_ = fmin(loss_partial, initial_coef, disp=True)

    def predict(self, X):
        # 係数をそれぞれの列の予測値に掛け合わせる
        x_coef = X * self.coef_
        # 行ごとに合計して、予測を計算
        predictions = np.sum(x_coef, axis=1)
        return predictions
