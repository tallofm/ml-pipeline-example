# src/evaluate.py

from sklearn.metrics import accuracy_score

def run(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.4f}")
