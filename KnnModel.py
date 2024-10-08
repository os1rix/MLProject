import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class BasketballModel:
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.best_knn = None
        self.chosenFeatures = None

    def train_and_validate_model(self, train_data, val_data):
        # Define the features for the model
        self.chosenFeatures = ['TS%', 'USG%', 'MinutesPlayed', 'PointsPerGame', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers']

        # Prepare training and validation data
        X_train = train_data[self.chosenFeatures]
        y_train = train_data['AllStarStatus']
        
        X_val = val_data[self.chosenFeatures]
        y_val = val_data['AllStarStatus']

        # Impute missing values
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        X_train_imputed = pd.DataFrame(self.imputer.fit_transform(X_train), columns=X_train.columns)
        X_val_imputed = pd.DataFrame(self.imputer.transform(X_val), columns=X_val.columns)

        # Standardize the data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_val_scaled = self.scaler.transform(X_val_imputed)

        # Train the KNN model
        self.best_knn = KNeighborsClassifier(n_neighbors=15, weights="distance")
        self.best_knn.fit(X_train_scaled, y_train)
        
        # Predict and validate
        y_val_pred = self.best_knn.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        print("------------------------------")
        print("KNN MODEL TRAINING RESULTS")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("\nClassification Report (Validation Set):")
        print(classification_report(y_val, y_val_pred))

    def test_model(self, test_data):
        # Prepare test data
        X_test = test_data[['TS%', 'USG%', 'MinutesPlayed', 'PointsPerGame', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers']]
        y_test = test_data['AllStarStatus']

        # Impute missing values
        X_test_imputed = pd.DataFrame(self.imputer.transform(X_test), columns=X_test.columns)

        # Standardize the data
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # Make predictions
        y_test_pred = self.best_knn.predict(X_test_scaled)

        # Calculate accuracy
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print("------------------------------")
        print("KNN MODEL TEST RESULTS")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred))

    def plot_accuracy_vs_neighbors(self, train_data, val_data, max_neighbors):
        accuracies = []
        neighbors_range = range(1, max_neighbors + 1)

        for n_neighbors in neighbors_range:
            # Train the KNN model with the current number of neighbors
            self.best_knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
            self.best_knn.fit(self.scaler.transform(self.imputer.transform(train_data[self.chosenFeatures])), train_data['AllStarStatus'])
            
            # Validate the model
            y_val_pred = self.best_knn.predict(self.scaler.transform(self.imputer.transform(val_data[self.chosenFeatures])))
            val_accuracy = accuracy_score(val_data['AllStarStatus'], y_val_pred)
            accuracies.append(val_accuracy)

        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.plot(neighbors_range, accuracies, marker='o')
        plt.title('KNN Accuracy vs Number of Neighbors')
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Validation Accuracy')
        plt.xticks(neighbors_range)
        plt.grid()
        plt.show()

    def plot_confusion_matrix(self, test_data):
        # Prepare test data
        X_test = test_data[self.chosenFeatures]
        y_test = test_data['AllStarStatus']

        # Impute missing values
        X_test_imputed = pd.DataFrame(self.imputer.transform(X_test), columns=X_test.columns)

        # Standardize the data
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # Make predictions
        y_test_pred = self.best_knn.predict(X_test_scaled)

        # Calculate confusion matrix
        confusion_knn = confusion_matrix(y_test, y_test_pred)

        # Plotting the confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_knn, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not All-Star', 'Reserve', 'Starter'], 
                    yticklabels=['Not All-Star', 'Reserve', 'Starter'])
        plt.title('Confusion Matrix for KNN Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
