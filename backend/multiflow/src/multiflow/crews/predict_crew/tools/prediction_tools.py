# f:\dev\crewai_agent\multiflow\src\multiflow\crews\predict_crew\tools\prediction_tools.py
import pandas as pd
import joblib # Or import pickle
from io import StringIO
from crewai.tools import BaseTool # Corrected import
from pathlib import Path
from typing import Optional, Any # Added for type hinting
from sklearn.base import BaseEstimator, TransformerMixin # Added for Preprocessor
from sklearn.preprocessing import LabelEncoder # Added for Preprocessor
# The correct ThresholdWrapper class from the training notebook.
# This is needed for joblib to deserialize the model correctly.

# Preprocessor class definition provided by the user
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.binary_map = {'Yes': 1, 'No': 0}
        self.categorical_cols = []

    def fit(self, X, y=None):
        X = X.copy()
        X['totalcharges'] = pd.to_numeric(X['totalcharges'], errors='coerce')

        binary_cols = ['partner', 'dependents', 'phoneservice', 'paperlessbilling']
        for col in binary_cols:
            X[col] = X[col].map(self.binary_map)

        self.categorical_cols = [
            col for col in X.select_dtypes(include=['object', 'category']).columns
            if not pd.to_numeric(X[col], errors='coerce').notna().all() and col not in binary_cols
        ]
        
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        X['totalcharges'] = pd.to_numeric(X['totalcharges'], errors='coerce')
        X['totalcharges'] = X['totalcharges'].fillna(X['tenure'] * X['monthlycharges'])

        service_cols_for_sum = ['phoneservice', 'internetservice', 'onlinesecurity',
                                'onlinebackup', 'deviceprotection', 'techsupport',
                                'streamingtv', 'streamingmovies']
        
        services_df = pd.DataFrame()
        for col in service_cols_for_sum:
            if col in X.columns:
                services_df[col] = X[col].replace({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})
                services_df[col] = pd.to_numeric(services_df[col], errors='coerce').fillna(0)

        X['new_totalservices'] = services_df[service_cols_for_sum].sum(axis=1)
        
        X['new_avg_charges'] = X['totalcharges'] / (X['tenure'] + 1e-5)
        X['new_increase'] = X['new_avg_charges'] / (X['monthlycharges'] + 1e-5)
        X['new_avg_service_fee'] = X['monthlycharges'] / (X['new_totalservices'] + 1e-5)
        X['charge_increased'] = (X['monthlycharges'] > X['new_avg_charges']).astype(int)
        X['charge_growth_rate'] = (X['monthlycharges'] - X['new_avg_charges']) / (X['new_avg_charges'] + 1e-5)
        
        X['is_auto_payment'] = X['paymentmethod'].astype(str).apply(
            lambda x: int('automatic' in x.lower() or 'bank' in x.lower())
        )

        binary_cols_transform = ['partner', 'dependents', 'phoneservice', 'paperlessbilling'] # Renamed to avoid conflict with fit's binary_cols if scope was an issue
        for col in binary_cols_transform:
            if col in X.columns:
                X[col] = X[col].map(self.binary_map)

        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).apply(lambda x: self.label_encoders[col].transform([x])[0] if x in self.label_encoders[col].classes_ else -1)
        
        return X.values

class ThresholdWrapper:
    def __init__(self, pipeline, threshold=0.5):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(self, X):
        probas = self.pipeline.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

class ChurnPredictionTool(BaseTool):
    name: str = "Customer Churn Predictor"
    description: str = (
        "Predicts customer churn based on CSV data. "
        "Input must be a string containing CSV data with a header row."
    )
    model_path: Path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "models" / "churn_pipeline_full.pkl"  # Corrected path
    model: Optional[Any] = None # Added type hint
    # !!! IMPORTANT: Define the expected feature columns and their order/types here !!!
    # This should match the training data used for churn_pipeline_full.pkl
    # Example:
    # expected_features = ['age', 'gender', 'service_calls', 'monthly_charges', 'total_charges'] 
    expected_features: list[str] = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", 
        "tenure", "PhoneService", "MultipleLines", "InternetService", 
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", 
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.expected_features:
            # This check will be more effective once the user provides the features.
            # For now, it will always pass if expected_features remains empty during initialization.
            # Consider raising an error or a strong warning if it's still empty after user input is expected.
            print("Warning: 'expected_features' list is empty. Please define it in ChurnPredictionTool based on the model's training data.")
        self._load_model()

    def _load_model(self):
        if self.model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            try:
                # Assuming the model was saved with joblib. If it was raw pickle, use pickle.load
                self.model = joblib.load(self.model_path)
                print(f"Churn prediction model loaded successfully from {self.model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load churn prediction model: {e}")

    def _run(self, csv_data_string: str) -> str:
        print(f"DEBUG prediction_tools.py: ChurnPredictionTool._run received: {csv_data_string[:250]}...") # Print first 250 chars
        if self.model is None:
            return "Error: Churn prediction model is not loaded."
        
        if not self.expected_features:
            return "Error: 'expected_features' list is not defined in the tool. Cannot process data."

        try:
            # Use StringIO to treat the string as a file
            csv_file_like_object = StringIO(csv_data_string)
            df = pd.read_csv(csv_file_like_object)

            # --- Data Preprocessing ---
            # 1. Check for required columns (use self.expected_features)
            missing_cols = [col for col in self.expected_features if col not in df.columns]
            if missing_cols:
                return f"Error: Input CSV is missing the following required columns: {', '.join(missing_cols)}"
            
            # 2. Ensure columns are in the correct order (if your pipeline is sensitive to it)
            #    and select only the expected features.
            df_processed = df[self.expected_features].copy()

            # Store customerID for output, then drop it from features sent to model
            customer_ids = None
            # The actual column name for customer ID in the input df might be 'customerID' or something else
            # if the user provides a CSV not strictly matching expected_features casing but has the ID.
            # We prioritize 'customerID' as it's in expected_features.
            # If 'customerID' is in df.columns, we use it. Otherwise, we try to find a column that was renamed to 'customerid'.
            original_customer_id_col_name = 'customerID' # Default from expected_features
            if original_customer_id_col_name in df_processed.columns:
                customer_ids = df_processed[original_customer_id_col_name].copy()
                df_processed.drop(columns=[original_customer_id_col_name], inplace=True)
            
            # Rename columns to match the model's internal Preprocessor expectations (mostly lowercase)
            rename_map = {
                "customerID": "customerid",
                "SeniorCitizen": "seniorcitizen",
                "Partner": "partner",
                "Dependents": "dependents",
                "PhoneService": "phoneservice",
                "MultipleLines": "multiplelines",
                "InternetService": "internetservice",
                "OnlineSecurity": "onlinesecurity",
                "OnlineBackup": "onlinebackup",
                "DeviceProtection": "deviceprotection",
                "TechSupport": "techsupport",
                "StreamingTV": "streamingtv",
                "StreamingMovies": "streamingmovies",
                "Contract": "contract",
                "PaperlessBilling": "paperlessbilling",
                "PaymentMethod": "paymentmethod",
                "MonthlyCharges": "monthlycharges",
                "TotalCharges": "totalcharges"
                # 'gender' and 'tenure' are already lowercase and match common expectations
            }
            # Only rename columns that exist in df_processed
            # Ensure we don't try to rename 'customerID' again if it was already handled/dropped
            columns_to_rename = {k: v for k, v in rename_map.items() if k in df_processed.columns and k != original_customer_id_col_name}
            df_processed.rename(columns=columns_to_rename, inplace=True)

            # 3. Add any other specific preprocessing steps here that your model expects
            #    (e.g., type conversions, scaling if not part of the .pkl pipeline)
            #    This depends heavily on how churn_pipeline_full.pkl was constructed.
            #    For now, we assume the .pkl file handles most preprocessing.

            # --- Prediction ---
            predictions = self.model.predict(df_processed)
            # Assuming binary classification (0 or 1). Adapt if it's probabilities or multi-class.
            
            # Add predictions back to the DataFrame for a nice output format
            df_processed['churn_prediction'] = predictions 
            
            result_summary = []
            for index, row in df_processed.iterrows():
                identifier = f"Row {index+1}"
                if customer_ids is not None and index < len(customer_ids):
                    identifier = f"Customer ID {customer_ids.iloc[index]}"
                elif 'customerid' in df.columns: # Fallback if customer_ids wasn't populated but 'customerid' somehow exists in original df
                     identifier = f"Customer ID {df.loc[index, 'customerid']}"
                elif 'customerID' in df.columns: # Fallback for original casing in df
                     identifier = f"Customer ID {df.loc[index, 'customerID']}"
                
                prediction_label = "Churn" if row['churn_prediction'] == 1 else "Not Churn"
                result_summary.append(f"{identifier}: Prediction = {prediction_label} (Raw: {row['churn_prediction']})")

            return "\n".join(result_summary) if result_summary else "No predictions made."

        except pd.errors.EmptyDataError:
            return "Error: Provided CSV data is empty."
        except pd.errors.ParserError:
            return "Error: Could not parse the provided CSV data. Please ensure it's valid CSV format."
        except Exception as e:
            return f"Error during prediction: {e}"

# To make it available for import
__all__ = ["ChurnPredictionTool", "ThresholdWrapper", "Preprocessor"]
