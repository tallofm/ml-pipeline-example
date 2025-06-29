# src/preprocess.py

from sklearn.model_selection import train_test_split
import pandas as pd

def run():
    # dummy dataset
    df = pd.DataFrame({
        'x1': range(100),
        'x2': range(100, 200),
        'y': [1 if i % 2 == 0 else 0 for i in range(100)]
    })

    X = df[['x1', 'x2']]
    y = df['y']
    return train_test_split(X, y, test_size=0.2, random_state=42)
