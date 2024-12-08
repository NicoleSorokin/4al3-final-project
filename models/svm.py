### training.py Implementation ###
## By Aniruddh Arora ##

import numpy as np
import pandas as pd
import pickle
import preprocessing as pp
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def initialize_weights(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0.0

    def compute_loss(self, X, y):
        margins = 1 - y * (X.dot(self.w) + self.b)
        return np.mean(np.maximum(0, margins))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        y_ = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        self.initialize_weights(n_features)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (x_i.dot(self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - x_i * y_[idx])
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.sign(X.dot(self.w) + self.b)

def evaluate_model(y_true, y_pred):
    y_true_binary = np.where(y_true == -1, 0, 1)
    y_pred_binary = np.where(y_pred == -1, 0, 1)
    acc = accuracy_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    return acc, recall, f1

def main():
    X_train, X_test, y_train, y_test = pp.preprocessing(percentage=0.5, kfold=False)
    X_train = X_train.astype(float).values
    y_train = y_train.astype(int).values
    X_test = X_test.astype(float).values
    y_test = y_test.astype(int).values

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies, fold_recalls, fold_f1s = [], [], []

    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
        svm.fit(X_tr, y_tr)
        y_val_pred = svm.predict(X_val)
        acc, recall, f1 = evaluate_model(y_val, y_val_pred)
        fold_accuracies.append(acc)
        fold_recalls.append(recall)
        fold_f1s.append(f1)

    avg_acc = np.mean(fold_accuracies)
    avg_recall = np.mean(fold_recalls)
    avg_f1 = np.mean(fold_f1s)
    print(f"Average validation accuracy across {k} folds: {avg_acc*100:.2f}%")
    print(f"Average validation recall across {k} folds: {avg_recall*100:.2f}%")
    print(f"Average validation F1-score across {k} folds: {avg_f1*100:.2f}%")

    final_svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    final_svm.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(final_svm, f)

    y_test_pred = final_svm.predict(X_test)
    test_acc, test_recall, test_f1 = evaluate_model(y_test, y_test_pred)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Recall: {test_recall*100:.2f}%")
    print(f"Test F1-score: {test_f1*100:.2f}%")

    np.save("test_features.npy", X_test)
    np.save("test_labels.npy", y_test)

if __name__ == "__main__":
    main()
