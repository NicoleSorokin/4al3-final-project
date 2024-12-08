import numpy as np
import pickle
from sklearn.metrics import accuracy_score

def evaluate_model(y_true, y_pred):
    y_true_binary = np.where(y_true == -1, 0, 1)
    y_pred_binary = np.where(y_pred == -1, 0, 1)
    return accuracy_score(y_true_binary, y_pred_binary)

def main():
    # Load model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load test data
    X_test = np.load("test_features.npy")
    y_test = np.load("test_labels.npy")

    y_test_converted = np.where(y_test <= 0, -1, 1)
    y_pred = model.predict(X_test)
    acc = evaluate_model(y_test_converted, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
