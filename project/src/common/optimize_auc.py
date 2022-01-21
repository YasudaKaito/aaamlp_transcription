from functools import partial

import numpy as np

# 参考: https://kenyu-life.com/2019/06/01/dhsm/
from scipy.optimize import fmin
from sklearn import metrics


class OptimizeAUC:
    def __init__(self) -> None:
        self.coef_ = 0

    def _auc(self, coef, X, y):
        """AUCを計算する

        :param coef: 係数のリスト、要素数はモデル数と等しい
        :param X: 予測確率、この場合は2次元配列（行数はデータ数, 列数はモデル数）
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
