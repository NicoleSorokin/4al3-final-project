import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score # for evaluating
from torch.utils.data import DataLoader, TensorDataset # for batch training
import numpy as np
from sklearn.model_selection import KFold
import pickle
from sklearn.metrics import confusion_matrix

# percentage is how much of data is used for testing. appropriate locations have been updated with the correct preprocessing signature.
def preprocessing(percentage, kfold, corr_threshold=0.1, model='svm'):
    file_path='diabetes_binary_health_indicators_BRFSS2015.csv'
    # can be downloaded from here: 
    # Load the dataset
    data = pd.read_csv(file_path)
    data = shuffle(data, random_state=42) 

    data = data[:int(len(data)*percentage)] # select the number of samples you want to use

    # drop features that have nan
    data = data.dropna()

    # Separate target and features
    target = 'Diabetes_binary'
    X = data.drop(columns=[target])
    y = data[target]

    sex_column = X['Sex']

    # Compute correlations with the target
    if model == 'svm':
        correlations = X.corrwith(y)
        selected_features = correlations[correlations.abs() >= corr_threshold].index.tolist()
        print(f"Selected Features based on correlation threshold ({corr_threshold}): {selected_features}")
        
    numerical_columns = ['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Income' , 'Education', 'GenHlth' ] 
    existing_num_cols = []
    
    for feature in numerical_columns:
        if feature in numerical_columns :
            existing_num_cols.append(feature)
    # Normalize numerical features using Min-Max Scaling
    if model == 'svm':
        scaler = StandardScaler()
        X[existing_num_cols] = scaler.fit_transform(X[existing_num_cols])
        X = X[selected_features]
        X['Sex'] = sex_column
    else:
        scaler = MinMaxScaler()
        X[existing_num_cols] = scaler.fit_transform(X[existing_num_cols])

    def balance_classes(X, y):
        # count samples
        class_counts = y.value_counts()
        max_count = max(class_counts.values)
        
        # X_balanced and y_balanced store the balanced classes
        X_balanced = []
        y_balanced = []

        for cls in class_counts.keys():
            # X_class = X[y == cls]
            # y_class = y[y == cls]
            
            X_class = X.loc[y.index[y == cls]]
            y_class = y.loc[y.index[y == cls]]

            if class_counts[cls] < max_count:
                multiplier = max_count // class_counts[cls]
                remainder = max_count % class_counts[cls]
                X_upsampled = pd.concat([X_class] * multiplier + [X_class.sample(remainder, replace=True)])
                y_upsampled = pd.concat([y_class] * multiplier + [y_class.sample(remainder, replace=True)])
            else:
                X_upsampled = X_class
                y_upsampled = y_class
            
            X_balanced.append(X_upsampled)
            y_balanced.append(y_upsampled)
        
        X_balanced = pd.concat(X_balanced)
        y_balanced = pd.concat(y_balanced)
        return X_balanced, y_balanced

    X_balanced, y_balanced = balance_classes(X, y)

    if not kfold:
        # Split the data into training and testing sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    # if for k fold, we want to return all the data
    return X_balanced, y_balanced

# loading training data + pre-processing
# for neural network do 0.1 for first parameter (10% data) and 0.05 for second parameter (5% data)
X_train, X_test, y_train, y_test = preprocessing(0.1, False, corr_threshold=0.1, model='nn')

# training model
##############################
#    Neural Network Model    #
##############################
class diabetes_neural_network(nn.Module) :
  def __init__(self, features) :
    super(diabetes_neural_network, self).__init__()
    # define the fully connected layers
    # play around with these
    self.fc1 = nn.Linear(features, 32)
    self.fc2 = nn.Linear(32, 16)
    self.fc3 = nn.Linear(16, 1)  
    self.dropout = nn.Dropout(0.2) # trying regularization
    
  # define the forward pass of the model
  def forward(self, x) :
    # fc
    x = self.fc1(x)
    # relu
    x = F.relu(x)
    # fc
    x = self.fc2(x)
    # relu
    x = F.relu(x)
    # fc
    x = self.fc3(x)
    x = self.dropout(x) # regularization
    x = torch.sigmoid(x) # apply sigmoid to get probabilities
    return x
  
average_loss_per_epoch = []

# define the training function
def training_nn(model, train_loader, optimizer, loss_func, X_val_tensor, y_val_tensor, fold_idx, k) :
    global average_loss_per_epoch
    training_losses = [] # to store the training losses
    # Run the algorithm for this many iterations
    EPOCH = 150
    for e in range(EPOCH):
        model.train() # sets the model to training mode
        cumulative_loss = 0 # intantiate the cumulative loss

        for x, y in train_loader: # for each batch in the training set
           optimizer.zero_grad() # reset the graident to 0, do this to avoid accumualtion
           outputs = model(x) # predict the probabilities (between 0 and 1)
           loss = loss_func(outputs, y) # calculate the loss
           loss.backward() # backpropagation
           optimizer.step() # perform gradient descent
           cumulative_loss += loss.item() # add the loss to the cumulative loss

        epoch_loss = cumulative_loss / len(train_loader)  # average loss for the epoch
        training_losses.append(epoch_loss) # append the average loss for this epoch to the list

        # add the epoch loss to average_loss_per_epoch (initialize if first fold)
        if len(average_loss_per_epoch) <= e:
            average_loss_per_epoch.append(epoch_loss)  # first one
        else:
            average_loss_per_epoch[e] += epoch_loss  # all the following fold

    # normalize fold contributions to average at the end of all folds
    if fold_idx == k - 1:  # If this is the last fold
        average_loss_per_epoch = [loss / k for loss in average_loss_per_epoch]

    # evaluate on validation set
    model.eval() # sets the model to evaluation mode
    with torch.no_grad(): # no gradient computation during evaluation
        val_outputs = model(X_val_tensor) # predict the probabilities (between 0 and 1)
        val_predictions = (val_outputs >= 0.5).float() # threshold at 0.5 to get binary predictions
        y_val_numpy = y_val_tensor.numpy() # convert tensor to numpy array

        # evaluation metrics
        val_accuracy = accuracy_score(y_val_numpy, val_predictions)
        val_recall = recall_score(y_val_numpy, val_predictions)
        val_f1 = f1_score(y_val_numpy, val_predictions)

    # print the evaluation metrics 
    print("Current epoch:", e)
    print("\tAvg training loss:", cumulative_loss / len(train_loader))
    print("\tValidation accuracy:", val_accuracy)
    print("\tValidation recall:", val_recall)
    print("\tValidation f1 score:", val_f1)

    # plot the validation losses over epochs
    plt.title("Training Loss Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(EPOCH), training_losses, label="Training Loss")
    plt.legend()
    plt.show()

def neural_network_model(X_train, y_train, k) :
    global average_loss_per_epoch
    print("Training model...")
    kfold = KFold(n_splits=k, shuffle=True, random_state=42) # initializing kfold
    
    num_features = X_train.shape[1] # getting num of features

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print("Fold:", fold+1)
        # Split train and validation data for this fold
        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_fold.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold.values, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val_fold.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold.values, dtype=torch.float32).unsqueeze(1)

        # Create dataloaders for batch training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Initialize the model for this fold
        model = diabetes_neural_network(num_features)

        # Define loss function
        loss_func = nn.BCELoss()

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # Train the model on the current fold
        training_nn(model, train_loader, optimizer, loss_func, X_val_tensor, y_val_tensor, fold, k)


    model_path = "nn.pkl"
    torch.save(model, model_path)

    # Plotting average loss after training
    plt.figure(figsize=(10, 6))
    plt.title("Average Training Loss Over Epochs Across Folds")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1, len(average_loss_per_epoch) + 1), average_loss_per_epoch, label="Average Loss")
    plt.legend()
    plt.show()


# COMMENT THIS OUT WHEN YOU RUN TEST.PY!!!!!
# neural_network_model(X_train, y_train, k=5)
# svm
##############################
#   Support Vector Machine   #
##############################

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=150):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.training_losses = []

    def initialize_weights(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0.0

    def compute_loss(self, X, y):
        margins = 1 - y * (X.dot(self.w) + self.b)  # Hinge loss margins
        hinge_loss = np.maximum(0, margins).mean()  # Mean hinge loss
        reg_loss = self.lambda_param * np.sum(self.w**2)  # Regularization term
        return hinge_loss + reg_loss

        # margins = 1 - y_i * (x_i.dot(self.w) + self.b)  # Compute margins
        # losses = np.maximum(0, margins)  # Element-wise max with 0
        # if isinstance(losses, np.ndarray):  # Handle case for multiple samples
        #     return np.mean(losses)
        # return losses  # For a single sample


        # return max(0, 1 - y_i * (x_i.dot(self.w) + self.b))

    def fit(self, X, y, X_val=None, y_val=None, print_epochs=None, fold_idx=0, total_folds=1):
        self.training_losses = []  # Initialize empty list for losses
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        y_ = np.where(y <= 0, -1, 1)  # Convert labels to {-1, 1}
        
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)

        if print_epochs is None:
            print_epochs = []

        for epoch in range(1, self.n_iters + 1):
            epoch_loss = 0  # set to 0 to track cumulative loss for the epoch

            for idx in range(n_samples):
                x_i = X[idx]
                y_i = y_[idx]

                # Check condition for hinge loss
                condition = y_i * (x_i.dot(self.w) + self.b) < 1

                # Update weights and bias
                if condition:
                    grad_w = 2 * self.lambda_param * self.w - y_i * x_i
                    grad_b = -y_i
                else:
                    grad_w = 2 * self.lambda_param * self.w
                    grad_b = 0

                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

                # Accumulate loss for the current instance
                instance_loss = max(0, 1 - y_i * (x_i.dot(self.w) + self.b))
                epoch_loss += instance_loss

            # Calculate and store average loss for the epoch
            avg_loss = epoch_loss / n_samples
            self.training_losses.append(avg_loss)

            # Optionally print metrics for specified epochs
            if epoch in print_epochs and fold_idx == total_folds - 1 and X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                acc, recall, precision, f1 = evaluate_model(y_val, y_val_pred)
                print(f"Epoch: {epoch}")
                print(f"Training Loss: {avg_loss:.4f}")
                print(f"Validation Accuracy: {acc*100:.2f}%")
                print(f"Validation Recall: {recall*100:.2f}%")
                print(f"Validation Precision: {precision*100:.2f}%")
                print(f"Validation F1 Score: {f1*100:.2f}%")


        acc, recall, precision, f1 = 0, 0, 0, 0
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            acc, recall, precision, f1 = evaluate_model(y_val, y_val_pred)

        return acc, recall, precision, f1
    
           
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.sign(X.dot(self.w) + self.b)

def compute_bias_svm(model, X_test, y_test, sex_data):

    # 1. Get model predictions for test set
    y_pred = model.predict(X_test)  # y_pred will be {-1,1}, we need to convert to {0,1}
    
    # Convert predictions from {-1,1} to {0,1} to match y_test format
    y_pred_binary = np.where(y_pred == 1, 1, 0)

    # 2. Create masks for the subgroups: male and female
    male_mask = (sex_data == 1)
    female_mask = (sex_data == 0)

    # 3. Compute confusion matrices for each subgroup
    # Subgroup: Male
    y_true_male = y_test[male_mask]
    y_pred_male = y_pred_binary[male_mask]
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_true_male, y_pred_male).ravel()

    # Subgroup: Female
    y_true_female = y_test[female_mask]
    y_pred_female = y_pred_binary[female_mask]
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_true_female, y_pred_female).ravel()

    # 4. Compute TPR/FPR for each subgroup
    # TPR = TP/(TP+FN), FPR = FP/(FP+TN)
    male_tpr = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
    male_fpr = fp_m / (fp_m + tn_m) if (fp_m + tn_m) > 0 else 0

    female_tpr = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
    female_fpr = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0

    # 5. Print the results
    print("Evaluating bias for SVM model...")
    print(f"Male TPR: {male_tpr:.4f}, Male FPR: {male_fpr:.4f}")
    print(f"Female TPR: {female_tpr:.4f}, Female FPR: {female_fpr:.4f}")

def evaluate_model(y_true, y_pred):
    # y_true_binary = np.where(y_true == -1, 0, 1)
    y_pred_binary = np.where(y_pred == 1, 1, 0)
    y_true_binary = y_true
    test_acc = accuracy_score(y_true_binary, y_pred_binary)
    test_recall = recall_score(y_true_binary, y_pred_binary)
    test_precision = precision_score(y_true_binary, y_pred_binary)
    test_f1 = f1_score(y_true_binary, y_pred_binary)
    # print("Final results:")
    # print(f"Accuracy: {test_acc*100:.1f}%")
    # print(f"Recall: {test_recall*100:.1f}%")
    # print(f"Precision: {test_precision*100:.1f}%")
    # print(f"F1 score: {test_f1*100:.1f}%")
    return test_acc, test_recall, test_precision, test_f1

def train_svm_model():
    X, y = preprocessing(percentage=0.05, kfold=True, corr_threshold=0.1, model='svm')
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    hyperparams = [
        # {'learning_rate': 0.0001, 'lambda_param': 0.001, 'n_iters': 150},
        {'learning_rate': 0.0001, 'lambda_param': 0.005, 'n_iters': 201},
        # {'learning_rate': 0.01, 'lambda_param': 0.01, 'n_iters': 150},
        # {'learning_rate': 0.1, 'lambda_param': 0.1, 'n_iters': 150},
    ]
 
    best_f1 = 0
    best_params = {}
    best_k = None
    best_model = None

    # k_values = range(2,15)
    # for k in k_values:
    for params in hyperparams:
        svm = SVM(learning_rate=params['learning_rate'], lambda_param=params['lambda_param'], n_iters=params['n_iters'])
        k = 8
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            acc, recall, precision, f1 = svm.fit(X_tr, y_tr, X_val, y_val, print_epochs=[10, 90, 200], fold_idx=fold-1, total_folds=k)
            fold_metrics.append((acc, recall, precision, f1))

        avg_acc = np.mean([m[0] for m in fold_metrics])
        avg_recall = np.mean([m[1] for m in fold_metrics])
        avg_precision = np.mean([m[2] for m in fold_metrics])
        avg_f1 = np.mean([m[3] for m in fold_metrics])
        if avg_f1 > best_f1:        # checking for best hyperparameter
            best_f1 = avg_f1
            best_params = params
            best_k = k
            best_model = svm

    print("Best Hyperparameters:")
    print(f"K = {best_k}, Params = {best_params}")
    print(best_params)
    print(f"Best Average Validation F1 Score: {best_f1*100:.2f}%\n")

    X_train_full, X_test, y_train_full, y_test = preprocessing(percentage=0.001, kfold=False, corr_threshold=0.1, model='svm')
    sex_data = X_test['Sex'].values # can be changed to check the bias from a particular column
    X_train_full = np.asarray(X_train_full, dtype=float)
    y_train_full = np.asarray(y_train_full, dtype=int)
    X_test1 = X_test
    Y_test1 = y_test 
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=int)
 
    # best_model.fit(X_train_full, y_train_full, print_epochs=[2])
    final_acc, final_recall, final_precision, final_f1 = best_model.fit(X_train_full, y_train_full)
    final_training_loss = best_model.training_losses[-1] if best_model.training_losses else None

    print("Final Results on Test Set after all epochs:")
    y_test_pred = best_model.predict(X_test)
    test_acc, test_recall, test_precision, test_f1 = evaluate_model(y_test, y_test_pred)
    print(f"Final Training Loss: {final_training_loss:.4f}")
    print(f"Accuracy: {test_acc*100:.1f}%")
    print(f"Recall: {test_recall*100:.1f}%")
    print(f"Precision: {test_precision*100:.1f}%")
    print(f"F1 score: {test_f1*100:.1f}%")



   
    compute_bias_svm(best_model, X_test1.values, Y_test1.values, sex_data)

    # Plot training loss over epochs for final chosen model
    plt.figure(figsize=(8,5))
    plt.title("SVM Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1, len(best_model.training_losses) + 1), best_model.training_losses, label="Training Loss")
    plt.legend()
    plt.show()

    with open("svm.pkl", "wb") as f:
        pickle.dump(best_model, f)

# COMMENT THIS OUT WHEN YOU RUN OTHER MODELS
# train_svm_model()

