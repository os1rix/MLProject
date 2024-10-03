import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

class BasketballModel:
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.best_knn = None

    def train_and_validate_model(self, train_data, val_data):
        features = ['Age', 'TS%', '3PAr', 'USG%', 'FG%', '3P%', '2P%', 'FT%', 'MinutesPlayed', 'PointsPerGame', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'PersonalFouls']
        
        X_train = train_data[features]
        y_train = train_data['AllStarStatus']
        
        X_val = val_data[features]
        y_val = val_data['AllStarStatus']

        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        X_train_imputed = pd.DataFrame(self.imputer.fit_transform(X_train), columns=X_train.columns)
        X_val_imputed = pd.DataFrame(self.imputer.transform(X_val), columns=X_val.columns)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_val_scaled = self.scaler.transform(X_val_imputed)

        self.best_knn = KNeighborsClassifier(n_neighbors=7)
        self.best_knn.fit(X_train_scaled, y_train)
        
        y_val_pred = self.best_knn.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("\nClassification Report (Validation Set):")
        print(classification_report(y_val, y_val_pred))

    def predict_all_star_status(self, features_dict):
        # Convert the dictionary to a DataFrame
        input_df = pd.DataFrame([features_dict])
        
        required_features = ['Age', 'TS%', '3PAr', 'USG%', 'FG%', '3P%', '2P%', 'FT%', 'MinutesPlayed', 'PointsPerGame', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'PersonalFouls']
        for feature in required_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df[required_features]
        
        input_imputed = pd.DataFrame(self.imputer.transform(input_df), columns=input_df.columns)
        
        input_scaled = self.scaler.transform(input_imputed)
        
        prediction = self.best_knn.predict(input_scaled)
        
        status_map = {0: "Not All-Star", 1: "Reserve", 2: "Starter"}
        return status_map[prediction[0]]
