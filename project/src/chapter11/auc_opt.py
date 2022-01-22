# 単純な平均と最適化されたモデルの重みを比較
import numpy as np
import xgboost as xgb
from common.optimize_auc import OptimizeAUC
from sklearn import ensemble, linear_model, metrics, model_selection
from sklearn.datasets import make_classification


def fit_and_auc(xfold1, yfold1, xfold2, yfold2, predict_target_fold):
    # ロジスティック回帰、ランダムフォレスト、XGBoost の3モデル
    logres = linear_model.LogisticRegression()
    rf = ensemble.RandomForestClassifier()
    xgbc = xgb.XGBClassifier()

    logres.fit(xfold1, yfold1)
    rf.fit(xfold1, yfold1)
    xgbc.fit(xfold1, yfold1)

    # fold2 に対して予測
    # クラス1である予測確率
    pred_logres = logres.predict_proba(xfold2)[:, 1]
    pred_rf = rf.predict_proba(xfold2)[:, 1]
    pred_xgbc = xgbc.predict_proba(xfold2)[:, 1]

    # 単純なアンサンブルとして平均
    avg_pred = (pred_logres + pred_rf + pred_xgbc) / 3
    fold2_preds = np.column_stack((pred_logres, pred_rf, pred_xgbc, avg_pred))
    # それぞれのAUCを算出
    aucs_fold2 = []
    for i in range(fold2_preds.shape[1]):
        auc = metrics.roc_auc_score(yfold2, fold2_preds[:, i])
        aucs_fold2.append(auc)

    print(f"Fold-{predict_target_fold}: LR AUC = {aucs_fold2[0]}")
    print(f"Fold-{predict_target_fold}: RF AUC = {aucs_fold2[1]}")
    print(f"Fold-{predict_target_fold}: XGB AUC = {aucs_fold2[2]}")
    print(f"Fold-{predict_target_fold}: Average Pred AUC = {aucs_fold2[3]}")
    return fold2_preds


# 10000行25列の二値分類データセット
X, y = make_classification(n_samples=10000, n_features=25)

# 2分割
xfold1, xfold2, yfold1, yfold2 = model_selection.train_test_split(
    X, y, test_size=0.5, stratify=y
)

# fold1 でモデル学習し、 fold2 に対し予測
fold2_preds = fit_and_auc(xfold1, yfold1, xfold2, yfold2, 2)
# 逆
fold1_preds = fit_and_auc(xfold2, yfold2, xfold1, yfold1, 1)

# 最適な重みを探索
opt = OptimizeAUC()
# 平均値列を削除
opt.fit(fold1_preds[:, :-1], yfold1)
opt_preds_fold2 = opt.predict(fold2_preds[:, :-1])
auc = metrics.roc_auc_score(yfold2, opt_preds_fold2)
print(f"Optimized AUC, Fold 2 = {auc}")
print(f"Coefficients = {opt.coef_}")

opt = OptimizeAUC()
opt.fit(fold2_preds[:, :-1], yfold2)
opt_preds_fold1 = opt.predict(fold1_preds[:, :-1])
auc = metrics.roc_auc_score(yfold1, opt_preds_fold1)
print(f"Optimized AUC, Fold 1 = {auc}")
print(f"Coefficients = {opt.coef_}")
