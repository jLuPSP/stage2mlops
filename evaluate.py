import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
import sys

clf = joblib.load("models/model.pkl")
X, y = load_iris(return_X_y=True)
y_pred = clf.predict(X)

f1 = f1_score(y, y_pred, average='macro')
print(f"F1 Score: {f1}")

if f1 < 0.90:
    print("Model did not meet the F1 threshold.")
    sys.exit(1)

print("Model passed evaluation.")