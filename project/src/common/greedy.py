from pprint import pprint
from typing import Tuple

from sklearn import linear_model, metrics
from sklearn.datasets import make_classification


class GreedyFeatureSelection:
    """貪欲法による特徴量選択のクラス. 対象のデータセットに適用するためには微修正が必要"""

    def evaluate_score(self, X, y):
        """モデルを学習しAUCを計算

        学習とAUCの計算に同じデータセットを使っているのに注意
        過学習しているが、貪欲法の一つの実装方法でもある
        交差検証すると分割数倍の時間がかかるため、簡便のためこのようになっている

        :param X: 学習用データセット
        :param y: 目的変数
        :return: AUC
        """
        # ロジスティック回帰モデルを学習し、同じデータセットに対するAUCを計算
        # データセットに適したモデルに変更可能
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        preds = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, preds)
        return auc

    def _feature_selection(self, X, y) -> Tuple[list, list]:
        """貪欲法による特徴量選択

        :param X: numpy配列の特徴量
        :param y: numpy配列の目的変数
        :return (最も良いスコア, 選ばれた特徴量)
        """
        # 有用な特徴量群
        good_features = []
        best_scores = []
        num_features = X.shape[1]

        while True:
            # 試行ごとに初期化
            this_feature = None
            best_score = 0

            # 特徴量を一つずつ見て、「最も有用な」特徴量を this_feature に
            for feature in range(num_features):
                if feature in good_features:
                    continue
                # 判明している有用な特徴量群 + 今回追加した特徴量でスコア計算
                selected_features = good_features + [feature]
                xtrain = X[:, selected_features]
                score = self.evaluate_score(xtrain, y)
                # 今回の試行の最高スコアとその特徴量を保存
                if score > best_score:
                    this_feature = feature
                    best_score = score

            # 今回のループで「最も有用な」特徴量を有用リスト、最高スコアリストに追加
            if this_feature != None:
                good_features.append(this_feature)
                best_scores.append(best_score)

            # 直前の反復で改善しなかった（これ以上追加で特徴量を加えても改善しなかった）場合には while ループを終了
            if len(best_scores) > 2:
                if best_scores[-1] < best_scores[-2]:
                    break

        # 有用リスト、最高スコアリストの最後の要素は「これ以上改善しなかった特徴量」のため含めない
        return best_scores[:-1], good_features[:-1]

    def __call__(self, X, y):
        """インスタンス呼び出し時の処理"""
        # 特徴量選択
        scores, features = self._feature_selection(X, y)
        return X[:, features], scores


if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=100)
    X_transformed, scores = GreedyFeatureSelection()(X, y)
    print("X_transformed features:")
    pprint(X_transformed[0])
    pprint(X_transformed[0].size)
    print("scores:")
    pprint(scores)
    pprint(len(scores))
