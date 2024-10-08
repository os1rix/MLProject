import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import r_regression

class Pearson:
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.best_rf = None

    def calculate_coefficients(self, train_data, val_data):
        # Features were chosen according to the coefficients" ranking
        chosenFeatures = ["TS%", "USG%", "MinutesPlayed", "PointsPerGame", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"]

        # Prepare training data
        X_train = train_data[chosenFeatures]
        y_train = train_data["AllStarStatus"]

        # Impute missing values and standardize the data
        self.imputer = SimpleImputer(strategy="constant", fill_value=0)
        X_train_imputed = pd.DataFrame(self.imputer.fit_transform(X_train), columns=X_train.columns)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)

        # Calculate and print Pearson correlation coefficients
        pearson_coefficients = r_regression(X_train_scaled, y_train)
        
        # Create a DataFrame to display the feature names and their Pearson coefficients
        pearson_df = pd.DataFrame({
            "Feature": chosenFeatures,
            "Pearson Coefficient": pearson_coefficients
        })
        
        # Sort by absolute value of the coefficients for better insight
        pearson_df = pearson_df.reindex(pearson_df["Pearson Coefficient"].abs().sort_values(ascending=False).index)
        print("------------------------------")
        print("\nPearson Correlation Coefficients for Features:")
        print(pearson_df.to_string(index=False))
