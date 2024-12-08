# Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def preprocessing(percentage):
    file_path='../data/diabetes_binary_health_indicators_BRFSS2015.csv'
    # Load the dataset
    data = pd.read_csv(file_path)
    data = shuffle(data)

    data = data[:int(len(data)*percentage)] # select the number of samples you want to use

    # drop features that have nan
    data = data.dropna()

    # Separate target and features
    target = 'Diabetes_binary'
    X = data.drop(columns=[target])
    y = data[target]

    # Normalize numerical features using Min-Max Scaling
    numerical_columns = ['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Income' , 'Education', 'GenHlth' ] 
    scaler = MinMaxScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    
    def balance_classes(X, y):
        # count samples
        class_counts = y.value_counts()
        max_count = max(class_counts.values)
        
        # X_balanced and y_balanced store the balanced classes
        X_balanced = []
        y_balanced = []

        for cls in class_counts.keys():
            X_class = X[y == cls]
            y_class = y[y == cls]
            
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

    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test