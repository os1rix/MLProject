import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class BasketballModel:
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.best_rf = None
        self.chosenFeatures = None

    def train_and_validate_model(self, train_data, val_data):
        # Define the features for the model
        originalFeatures = ["Age", "TS%", "3PAr", "USG%", "FG%", "3P%", "2P%", "FT%", "MinutesPlayed", "PointsPerGame", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers", "PersonalFouls"]
        self.chosenFeatures = ["TS%", "USG%", "MinutesPlayed", "PointsPerGame", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"]

        # Prepare training and validation data
        X_train = train_data[self.chosenFeatures]
        y_train = train_data["AllStarStatus"]
        
        X_val = val_data[self.chosenFeatures]
        y_val = val_data["AllStarStatus"]

        # Impute missing values
        self.imputer = SimpleImputer(strategy="constant", fill_value=0)
        X_train_imputed = pd.DataFrame(self.imputer.fit_transform(X_train), columns=X_train.columns)
        X_val_imputed = pd.DataFrame(self.imputer.transform(X_val), columns=X_val.columns)

        # Standardize the data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_val_scaled = self.scaler.transform(X_val_imputed)

        # Train the Random Forest model
        self.best_rf = RandomForestClassifier(criterion="gini",  max_depth=10, max_leaf_nodes=20,random_state=42)
        self.best_rf.fit(X_train_scaled, y_train)
        
        # Predict and validate
        y_val_pred = self.best_rf.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print("------------------------------")
        print("RANDOM FOREST MODEL TRAINING RESULTS")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("\nClassification Report (Validation Set):")
        print(classification_report(y_val, y_val_pred))

    def test_model(self, test_data):
        # Prepare test data
        X_test = test_data[["TS%", "USG%", "MinutesPlayed", "PointsPerGame", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"]]
        y_test = test_data["AllStarStatus"]

        # Impute missing values
        X_test_imputed = pd.DataFrame(self.imputer.transform(X_test), columns=X_test.columns)

        # Standardize the data
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # Make predictions
        y_test_pred = self.best_rf.predict(X_test_scaled)

        # Calculate accuracy
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print("------------------------------")
        print("RANDOM FOREST MODEL TEST RESULTS")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred))

    def plot_confusion_matrix(self, test_data):
        # Prepare test data
        X_test = test_data[self.chosenFeatures]
        y_test = test_data["AllStarStatus"]

        # Impute missing values
        X_test_imputed = pd.DataFrame(self.imputer.transform(X_test), columns=X_test.columns)

        # Standardize the data
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # Make predictions
        y_test_pred = self.best_rf.predict(X_test_scaled)

        # Calculate confusion matrix
        confusion_rf = confusion_matrix(y_test, y_test_pred)

        # Plotting the confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_rf, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Not All-Star", "Reserve", "Starter"], 
                    yticklabels=["Not All-Star", "Reserve", "Starter"])
        plt.title("Confusion Matrix for Random Forest Model")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def plot_accuracy_vs_max_depth(self, train_data, val_data, max_depths):
        accuracies_depth = []

        # Evaluate for max_depth
        for depth in max_depths:
            self.best_rf = RandomForestClassifier(criterion="gini", n_estimators=100, max_depth=depth, random_state=42)
            self.best_rf.fit(self.scaler.transform(self.imputer.transform(train_data[self.chosenFeatures])), train_data["AllStarStatus"])
            
            y_val_pred = self.best_rf.predict(self.scaler.transform(self.imputer.transform(val_data[self.chosenFeatures])))
            val_accuracy = accuracy_score(val_data["AllStarStatus"], y_val_pred)
            accuracies_depth.append(val_accuracy)

        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.plot(max_depths, accuracies_depth, marker="o")
        plt.title("Random Forest Accuracy vs Max Depth")
        plt.xlabel("Max Depth")
        plt.ylabel("Validation Accuracy")
        plt.xticks(max_depths)
        plt.grid()
        plt.show()

    def plot_accuracy_vs_max_leaf_nodes(self, train_data, val_data, max_leaf_nodes):
        accuracies_leaf = []

        # Evaluate for max_leaf_nodes
        for leaf_nodes in max_leaf_nodes:
            self.best_rf = RandomForestClassifier(criterion="gini", n_estimators=100, max_leaf_nodes=leaf_nodes, random_state=42)
            self.best_rf.fit(self.scaler.transform(self.imputer.transform(train_data[self.chosenFeatures])), train_data["AllStarStatus"])
            
            y_val_pred = self.best_rf.predict(self.scaler.transform(self.imputer.transform(val_data[self.chosenFeatures])))
            val_accuracy = accuracy_score(val_data["AllStarStatus"], y_val_pred)
            accuracies_leaf.append(val_accuracy)

        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.plot(max_leaf_nodes, accuracies_leaf, marker="o")
        plt.title("Random Forest Accuracy vs Max Leaf Nodes")
        plt.xlabel("Max Leaf Nodes")
        plt.ylabel("Validation Accuracy")
        plt.xticks(max_leaf_nodes)
        plt.grid()
        plt.show()
