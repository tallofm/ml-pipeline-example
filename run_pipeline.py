# run_pipeline.py

from src import config, preprocess, train, evaluate

def main():
    print("🚀 Starting ML pipeline...")

    # step 1: load and clean data
    X_train, X_test, y_train, y_test = preprocess.run()

    # step 2: train model
    model = train.run(X_train, y_train)

    # step 3: evaluate
    evaluate.run(model, X_test, y_test)

if __name__ == "__main__":
    main()
