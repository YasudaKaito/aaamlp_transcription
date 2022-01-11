from sklearn import linear_model, ensemble
import xgboost as xgb

models = {
    "logres": linear_model.LogisticRegression(),
    "rf": ensemble.RandomForestClassifier,
    "xgb": xgb.XGBClassifier,
}
