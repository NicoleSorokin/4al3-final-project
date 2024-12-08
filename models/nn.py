### NEURAL NETWORK ###

# import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset # for batch training
from sklearn.metrics import accuracy_score, recall_score, f1_score # for evaluating

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import preprocessing as pp

# defining the nn
class diabetes_neural_network(nn.Module) :
  def __init__(self, features) :
    super(diabetes_neural_network, self).__init__()
    # define the fully connected layers
    # play around with these
    self.fc1 = nn.Linear(features, 256)
    self.fc2 = nn.Linear(256, 64)
    self.fc3 = nn.Linear(64, 1)  
    # self.fc4 = nn.Linear(64, 32)
    # self.fc5 = nn.Linear(32, 1)# 0 = no diabetes 1 = prediabetes or diabetes
    self.dropout = nn.Dropout(0.5) # trying regularization
    
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
    # relu
    # x = F.relu(x)
    # # fc
    # x = self.fc4(x)
    # # relu
    # x = F.relu(x)
    # # fc
    # x = self.fc5(x)
    # # relu
    # x = F.relu(x)
    # outputting as a probability
    x = self.dropout(x) # regularization
    x = torch.sigmoid(x)  
    return x
  
def training_nn(model, train_loader, optimizer, loss_func, X_val_tensor, y_val_tensor) :
    # train
    training_losses = []
    # Run the algorithm for this many iterations
    EPOCH = 200
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

        print(cumulative_loss / len(train_loader)) # print the average loss for this epoch
        training_losses.append(cumulative_loss / len(train_loader)) # append the average loss for this epoch to the list

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

        # print the evaluation metrics for every 10th epoch
        if e % 10 == 0 and e != 0:
          print("epoch:", e)
          print("validation accuracy:", val_accuracy)
          print("validation recall:", val_recall)
          print("validation f1 score:", val_f1)

    # plot the validation losses over epochs
    plt.title("Training Loss Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(EPOCH), training_losses, label="Training Loss")
    plt.legend()
    plt.show()

def evaluating_nn(model, X_test_tensor, y_test_tensor) :
    # evaluate on held out test set
    model.eval() # sets the model to evaluation mode
    with torch.no_grad(): # no gradient computation during evaluation
        test_outputs = model(X_test_tensor) # predict the probabilities (between 0 and 1)
        test_predictions = (test_outputs >= 0.5).float() # threshold at 0.5 to get binary predictions
        y_test_numpy = y_test_tensor.numpy() # convert tensor to numpy array

        # evaluation metrics
        test_accuracy = accuracy_score(y_test_numpy, test_predictions)
        test_recall = recall_score(y_test_numpy, test_predictions)
        test_f1 = f1_score(y_test_numpy, test_predictions)

    print("Test Accuracy:", test_accuracy*100)
    print(f"Test Recall:", test_recall*100)
    print(f"Test F1 Score:", test_f1*100)

def neural_network_model() :
    # load the dataset
    X_train, X_test, y_train, y_test = pp.preprocessing()

    # split training into train and validation-implent kfold cross validation later
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)#adding a dimension to the tensor to make it compatible with the model
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)#adding a dimension to the tensor to make it compatible with the model
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)#adding a dimension to the tensor to make it compatible with the model

    # create dataloaders for batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # get number of features
    num_features = X_train_tensor.shape[1]

    # # initialize model
    model = diabetes_neural_network(num_features)

    # define loss function
    loss_func = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # train the model
    training_nn(model, train_loader, optimizer, loss_func, X_val_tensor, y_val_tensor)

    # evaluate the model
    evaluating_nn(model, X_test_tensor, y_test_tensor)

neural_network_model()