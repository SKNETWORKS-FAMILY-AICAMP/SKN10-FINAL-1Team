# f:\dev\SKN10-FINAL-1Team\swarm_agent\src\agent\tools.py
import json
import uuid
from typing import Dict, Any, List, Optional
import os
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import Tool, StructuredTool
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from openai import OpenAI as OpenAIClient # For embeddings
from pinecone import Pinecone as PineconeClient # For vector search

# --- Analyst Chart Tool ---
def generate_chart_html(title: str, chart_type: str, data: dict, options: Optional[Dict[str, Any]] = None) -> str:
    """Generates chart data as a JSON string containing HTML for the canvas and JS for the script."""
    chart_id = f"chart-{uuid.uuid4().hex[:8]}"
    
    chart_options = {
        'responsive': True,
        'plugins': {
            'title': {
                'display': True,
                'text': title
            }
        }
    }
    if options:
        chart_options.update(options)

    data_json = json.dumps(data)
    options_json = json.dumps(chart_options)
    
    canvas_html = f"<div><canvas id='{chart_id}'></canvas></div>"
    
    chart_id_js_safe = chart_id.replace('-', '_')
    script_js = f"""
      const ctx_{chart_id_js_safe} = document.getElementById('{chart_id}');
      if (ctx_{chart_id_js_safe}) {{
        new Chart(ctx_{chart_id_js_safe}, {{
          type: '{chart_type}',
          data: {data_json},
          options: {options_json}
        }});
      }} else {{
        console.error('Failed to find canvas element with ID: {chart_id}');
      }}
    """
    
    output = {
        "canvas_html": canvas_html,
        "script_js": script_js
    }
    return json.dumps(output)

class ChartInputArgs(BaseModel):
    title: str = Field(..., description="The title for the chart.")
    chart_type: str = Field(..., description="Type of chart (e.g., 'bar', 'line', 'pie').")
    data: Dict[str, Any] = Field(..., description="Data for the chart, following Chart.js structure (labels, datasets).")
    options: Optional[Dict[str, Any]] = Field(None, description="Optional Chart.js options to override defaults.")

analyst_chart_tool = StructuredTool.from_function(
    func=generate_chart_html,
    name="ChartGenerator",
    description="Generates the necessary HTML and JavaScript for a chart. Returns a JSON string with 'canvas_html' and 'script_js' keys.",
    args_schema=ChartInputArgs,
)

# --- Customer Churn Prediction Tool ---
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

MODEL_PATH = Path(__file__).parent / "models" / "churn_pipeline_full.pkl"
_churn_model_cache = None

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


def _load_churn_model():
    global _churn_model_cache
    if _churn_model_cache is not None:
        return _churn_model_cache

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Error: Churn prediction model file not found at {MODEL_PATH}")

    main_module = sys.modules['__main__']
    original_preprocessor = getattr(main_module, 'Preprocessor', None)
    original_thresholdwrapper = getattr(main_module, 'ThresholdWrapper', None)
    we_set_preprocessor = False
    we_set_thresholdwrapper = False

    try:
        if not (hasattr(main_module, 'Preprocessor') and main_module.Preprocessor == Preprocessor):
            if hasattr(main_module, 'Preprocessor'): 
                raise RuntimeError("Conflict: __main__.Preprocessor exists and is not the expected class for unpickling.")
            main_module.Preprocessor = Preprocessor
            we_set_preprocessor = True
        
        if not (hasattr(main_module, 'ThresholdWrapper') and main_module.ThresholdWrapper == ThresholdWrapper):
            if hasattr(main_module, 'ThresholdWrapper'): 
                raise RuntimeError("Conflict: __main__.ThresholdWrapper exists and is not the expected class for unpickling.")
            main_module.ThresholdWrapper = ThresholdWrapper
            we_set_thresholdwrapper = True

        loaded_pipeline = joblib.load(MODEL_PATH)
        _churn_model_cache = loaded_pipeline
        return _churn_model_cache
    except Exception as e:
        raise RuntimeError(f"Error loading churn model: {str(e)}")
    finally:
        if we_set_preprocessor:
            if original_preprocessor is not None:
                main_module.Preprocessor = original_preprocessor
            else:
                if hasattr(main_module, 'Preprocessor'): 
                    del main_module.Preprocessor
        
        if we_set_thresholdwrapper:
            if original_thresholdwrapper is not None:
                main_module.ThresholdWrapper = original_thresholdwrapper
            else:
                if hasattr(main_module, 'ThresholdWrapper'): 
                    del main_module.ThresholdWrapper

class ChurnPredictionInputArgs(BaseModel):
    csv_data_string: str = Field(description="A string containing customer data in CSV format. Must include headers.")

def predict_customer_churn(csv_data_string: str) -> str:
    try:
        pipeline = _load_churn_model() # Raises FileNotFoundError or RuntimeError on failure
        
        data_io = StringIO(csv_data_string)
        df = pd.read_csv(data_io)

        # --- Data Normalization ---
        # Try to find a customer ID column, being flexible with naming, and store the IDs
        id_col_found = None
        for col in df.columns:
            if col.lower().replace('_', '') in ['customerid', 'customer_id']:
                id_col_found = col
                break
        
        customer_ids = df[id_col_found].tolist() if id_col_found else [f"Row {i+1}" for i in range(len(df))]

        # Normalize column names: lowercase, remove underscores, fix typos
        df.columns = [col.lower().replace('_', '') for col in df.columns]
        rename_map = {'internetserivce': 'internetservice'}
        df.rename(columns=rename_map, inplace=True)

        if df.empty:
            return "Error: Input CSV data is empty or resulted in an empty DataFrame."

        # --- Column Ordering ---
        expected_cols_in_order = [
            'gender', 'seniorcitizen', 'partner', 'dependents', 'tenure', 
            'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity', 
            'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 
            'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod', 
            'monthlycharges', 'totalcharges'
        ]
        
        df_for_prediction = pd.DataFrame()
        for col in expected_cols_in_order:
            if col in df.columns:
                df_for_prediction[col] = df[col]
            else:
                df_for_prediction[col] = np.nan

        df_for_prediction = df_for_prediction[expected_cols_in_order]

        # --- Prediction ---
        predictions = pipeline.predict(df_for_prediction)
        probabilities = pipeline.predict_proba(df_for_prediction)[:, 1]
        
        if len(predictions) == 0:
            return "Error: Prediction could not be made. The model did not return any results."

        # --- Format Output for All Customers ---
        results = []
        for i, cust_id in enumerate(customer_ids):
            prediction_label = "Likely to Churn" if predictions[i] == 1 else "Unlikely to Churn"
            probability = probabilities[i]
            results.append(f"Customer ID {cust_id}: Prediction: {prediction_label} (Probability: {probability:.2%})")
            
        return "\n".join(results)

    except FileNotFoundError as e: 
        return str(e) 
    except RuntimeError as e: 
        return str(e) 
    except pd.errors.EmptyDataError:
        return "Error: Input CSV data is empty or invalid (could not be parsed)."
    except KeyError as e:
        return f"Error: Missing expected column in CSV data for prediction - {str(e)}"
    except IndexError: 
        return "Error: Prediction could not be made, possibly due to empty or invalid input data after processing leading to no output."
    except Exception as e:
        return f"An unexpected error occurred during churn prediction: {str(e)}"

predict_churn_tool = Tool(
    name="CustomerChurnPredictor",
    func=predict_customer_churn,
    description="Predicts customer churn based on CSV data. Input must be a string containing CSV data with a header row.",
    args_schema=ChurnPredictionInputArgs
)

# --- Document Search Tools (Pinecone) ---
_openai_client_cache = None
_pinecone_client_cache = None
_pinecone_index_cache = None

def get_openai_client():
    global _openai_client_cache
    if _openai_client_cache is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        _openai_client_cache = OpenAIClient(api_key=api_key)
    return _openai_client_cache

def get_pinecone_index():
    global _pinecone_client_cache, _pinecone_index_cache
    if _pinecone_index_cache is None:
        api_key = os.getenv("PINECONE_API_KEY")
        env = os.getenv("PINECONE_ENV")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not all([api_key, env, index_name]):
            raise ValueError("Pinecone API key, environment, or index name not set.")
        if _pinecone_client_cache is None:
            _pinecone_client_cache = PineconeClient(api_key=api_key, environment=env)
        _pinecone_index_cache = _pinecone_client_cache.Index(index_name)
    return _pinecone_index_cache

class PineconeSearchArgs(BaseModel):
    query: str = Field(..., description="The search query string.")
    top_k: int = Field(default=3, description="Number of top results to return.")

def search_pinecone_documents(query: str, top_k: int = 3) -> str:
    try:
        openai_client = get_openai_client()
        pinecone_index = get_pinecone_index()
        
        embedding_response = openai_client.embeddings.create(input=[query], model="text-embedding-ada-002")
        query_embedding = embedding_response.data[0].embedding
        
        results = pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        
        output = []
        if results.matches:
            for match in results.matches:
                output.append(f"Document ID: {match.id}\nScore: {match.score:.4f}\nContent Snippet: {match.metadata.get('text_chunk', 'N/A')[:200]}...\nSource: {match.metadata.get('source', 'N/A')}")
            return "\n---\n".join(output)
        else:
            return "No relevant documents found."
    except Exception as e:
        print(f"Error during Pinecone search: {e}", file=sys.stderr)
        return f"Error searching documents: {e}"

search_pinecone_documents_tool = Tool(
    name="PineconeDocumentSearch",
    func=search_pinecone_documents,
    description="Searches for relevant documents in Pinecone based on a query. Returns document IDs, scores, snippets, and sources.",
    args_schema=PineconeSearchArgs
)

class DocumentContentArgs(BaseModel):
    document_id: str = Field(..., description="The ID of the document to retrieve.")

def get_document_content(document_id: str) -> str:
    try:
        pinecone_index = get_pinecone_index()
        fetched_vector = pinecone_index.fetch(ids=[document_id])
        if fetched_vector.vectors and document_id in fetched_vector.vectors:
            metadata = fetched_vector.vectors[document_id].metadata
            if metadata and 'text_chunk' in metadata:
                return f"Document ID: {document_id}\nSource: {metadata.get('source', 'N/A')}\n\nContent:\n{metadata['text_chunk']}"
            else:
                return f"Document ID {document_id} found, but it has no 'text_chunk' in metadata."
        else:
            return f"Document with ID '{document_id}' not found."
    except Exception as e:
        print(f"Error fetching document content: {e}", file=sys.stderr)
        return f"Error fetching document content for ID '{document_id}': {e}"

get_document_content_tool = Tool(
    name="PineconeDocumentContentRetriever",
    func=get_document_content,
    description="Retrieves the full text content of a specific document from Pinecone using its ID.",
    args_schema=DocumentContentArgs
)

# --- SQL Database Tools for Analyst ---
DB_URI = os.getenv("DB_URI")
sql_tools_for_analyst = []

if DB_URI:
    try:
        engine = create_engine(DB_URI)
        db = SQLDatabase(engine)
        llm_for_sql_toolkit = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm_for_sql_toolkit)
        sql_tools_for_analyst = sql_toolkit.get_tools()
        print("SQL Database tools initialized successfully for Analyst.", file=sys.stderr)
    except Exception as e:
        print(f"Error initializing SQL Database tools: {e}", file=sys.stderr)
        print("SQL tools will be unavailable for the Analyst Assistant.", file=sys.stderr)
        sql_tools_for_analyst = []
else:
    print("DB_URI not found. SQL tools will be unavailable.", file=sys.stderr)
    sql_tools_for_analyst = []

__all__ = [
    "analyst_chart_tool", 
    "predict_churn_tool", 
    "sql_tools_for_analyst", 
    "Preprocessor", 
    "ThresholdWrapper",
    "search_pinecone_documents_tool",
    "get_document_content_tool"
]
