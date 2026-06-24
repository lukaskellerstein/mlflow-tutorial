import argparse, mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=5)
    a = p.parse_args()
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=a.n_estimators, max_depth=a.max_depth)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    mlflow.log_params({"n_estimators": a.n_estimators, "max_depth": a.max_depth})
    mlflow.log_metric("accuracy", acc)
    print(f"Accuracy: {acc:.4f}")
if __name__ == "__main__":
    main()
