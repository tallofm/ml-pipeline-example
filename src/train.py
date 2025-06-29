# src/train.py

from sklearn.ensemble import RandomForestClassifier


def run(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
