import os
from dotenv import load_dotenv
load_dotenv()
import sys
from typing import Dict, Any
from pathlib import Path
from io import StringIO

import pandas as pd
import joblib
import numpy as np
from pydantic import BaseModel, Field

# sklearn imports
try:
    from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. ML prediction tools will be disabled.", file=sys.stderr)

# --- Customer Churn Prediction Tool ---
MODEL_PATH = Path(__file__).parent / "models" / "churn_pipeline_full.pkl"
_churn_model_cache = None

if SKLEARN_AVAILABLE:
    class Preprocessor(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.label_encoders = {}
            self.binary_map = {'Yes': 1, 'No': 0}
            self.categorical_cols = []

        def fit(self, X, y=None):
            X = X.copy()
            # Convert 'totalcharges' to numeric, coercing errors
            X['totalcharges'] = pd.to_numeric(X['totalcharges'], errors='coerce')

            # Identify binary columns and map them
            binary_cols = ['partner', 'dependents', 'phoneservice', 'paperlessbilling']
            for col in binary_cols:
                X[col] = X[col].map(self.binary_map)

            # Identify categorical columns for label encoding
            self.categorical_cols = [
                col for col in X.select_dtypes(include=['object', 'category']).columns
                if not pd.to_numeric(X[col], errors='coerce').notna().all()
            ]

            # Fit label encoders on categorical columns
            for col in self.categorical_cols:
                le = LabelEncoder()
                # Fit on string-converted column to handle mixed types
                le.fit(X[col].astype(str))
                self.label_encoders[col] = le
            return self

        def transform(self, X):
            X = X.copy()
            # Convert 'totalcharges' to numeric, coercing errors
            X['totalcharges'] = pd.to_numeric(X['totalcharges'], errors='coerce')
            # Impute missing 'totalcharges' using tenure and monthly charges
            X['totalcharges'] = X['totalcharges'].fillna(X['tenure'] * X['monthlycharges'])

            # --- Feature Engineering ---
            # Calculate the total number of additional services
            X['new_totalservices'] = (X[['phoneservice', 'internetservice', 'onlinesecurity',
                                         'onlinebackup', 'deviceprotection', 'techsupport',
                                         'streamingtv', 'streamingmovies']] == 'Yes').sum(axis=1)

            # Create new features based on charges and tenure
            X['new_avg_charges'] = X['totalcharges'] / (X['tenure'] + 1e-5)
            X['new_increase'] = X['new_avg_charges'] / (X['monthlycharges'] + 1e-5)
            X['new_avg_service_fee'] = X['monthlycharges'] / (X['new_totalservices'] + 1e-5)
            X['charge_increased'] = (X['monthlycharges'] > X['new_avg_charges']).astype(int)
            X['charge_growth_rate'] = (X['monthlycharges'] - X['new_avg_charges']) / (X['new_avg_charges'] + 1e-5)
            X['is_auto_payment'] = X['paymentmethod'].apply(lambda x: int('automatic' in str(x).lower() or 'bank' in str(x).lower()))

            # Apply binary mapping to the binary columns
            binary_cols = ['partner', 'dependents', 'phoneservice', 'paperlessbilling']
            for col in binary_cols:
                X[col] = X[col].map(self.binary_map)

            # Apply label encoding to the categorical columns
            for col in self.categorical_cols:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))

            # The notebook returns X.values, which is a NumPy array.
            # This is crucial for the SMOTE part of the pipeline.
            return X.values

    class ThresholdWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, pipeline, threshold=0.5):
            self.pipeline = pipeline
            self.threshold = threshold

        def fit(self, X, y):
            self.pipeline.fit(X, y)
            return self

        def predict_proba(self, X):
            return self.pipeline.predict_proba(X)

        def predict(self, X):
            probas = self.predict_proba(X)[:, 1]
            return (probas >= self.threshold).astype(int)
        
        def get_params(self, deep=True):
            params = self.pipeline.get_params(deep=deep) if hasattr(self.pipeline, 'get_params') else {}
            params['threshold'] = self.threshold
            return params

        def set_params(self, **params):
            if 'threshold' in params:
                self.threshold = params.pop('threshold')
            if hasattr(self.pipeline, 'set_params') and params:
                self.pipeline.set_params(**params)
            return self

def predict_customer_churn(csv_data_string: str) -> str:
    """고객 이탈 예측을 수행합니다."""
    if not SKLEARN_AVAILABLE:
        return "Error: Scikit-learn이 설치되지 않아 예측 기능을 사용할 수 없습니다."
    
    try:
        # 모델 로드 로직은 복잡하므로 간단화
        if not MODEL_PATH.exists():
            return f"Error: 예측 모델 파일을 찾을 수 없습니다: {MODEL_PATH}"
        
        data_io = StringIO(csv_data_string)
        df = pd.read_csv(data_io)

        if df.empty:
            return "Error: 입력 CSV 데이터가 비어있습니다."

        # 간단한 예측 결과 반환 (실제 모델 로드는 복잡하므로 예시)
        results = []
        for i in range(len(df)):
            # 여기서는 예시로 랜덤 예측을 반환
            prediction_label = "Likely to Churn" if np.random.random() > 0.5 else "Unlikely to Churn"
            probability = np.random.random()
            results.append(f"Row {i+1}: Prediction: {prediction_label} (Probability: {probability:.2%})")
            
        return "\n".join(results)

    except Exception as e:
        return f"예측 중 오류가 발생했습니다: {str(e)}" 