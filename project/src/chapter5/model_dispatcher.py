from sklearn import linear_model, ensemble

models = {
    "logres": linear_model.LogisticRegression(),
    "rf": ensemble.RandomForestClassifier,
}
