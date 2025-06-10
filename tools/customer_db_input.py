import pandas as pd
import psycopg2
import uuid
from decimal import Decimal, InvalidOperation
import os
from urllib.parse import urlparse
from tqdm import tqdm # Import tqdm

# --- Configuration ---
# Provided PostgreSQL URL: postgresql://myuser:hinton1234@35.170.244.126:5432/mydatabase
DATABASE_URL = "postgresql://myuser:hinton1234@35.170.244.126:5432/mydatabase"
CSV_FILE_PATH = r"d:\dev\SKN10-FINAL-1Team\tools\customer_data (1).csv"
TABLE_NAME = "telecom_customers" # Assuming 'knowledge' is the app name

# Parse database URL
parsed_url = urlparse(DATABASE_URL)
DB_CONFIG = {
    "dbname": parsed_url.path[1:], # Remove leading '/'
    "user": parsed_url.username,
    "password": parsed_url.password,
    "host": parsed_url.hostname,
    "port": parsed_url.port
}

def to_bool(value):
    """Converts various string/numeric representations to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) # 0 is False, others True
    if isinstance(value, str):
        val_lower = value.strip().lower()
        if val_lower in ['yes', 'true', 't', '1']:
            return True
        elif val_lower in ['no', 'false', 'f', '0']:
            return False
    return False # Default for unhandled or empty strings if not nullable

def to_decimal(value, default=Decimal('0.00')):
    """Converts a value to Decimal, handling empty strings or errors."""
    if isinstance(value, Decimal):
        return value
    try:
        # Remove any whitespace that might cause issues
        cleaned_value = str(value).strip()
        if not cleaned_value: # Handle empty strings
            return default
        return Decimal(cleaned_value)
    except (InvalidOperation, ValueError, TypeError):
        return default

def main():
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully read {len(df)} rows from {CSV_FILE_PATH}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print(f"Successfully connected to database '{DB_CONFIG['dbname']}' on {DB_CONFIG['host']}.")

        # Delete existing data from the table
        try:
            print(f"Deleting existing data from {TABLE_NAME}...")
            cur.execute(f"DELETE FROM {TABLE_NAME};")
            conn.commit() # Commit the deletion
            print(f"Successfully deleted all data from {TABLE_NAME}.")
        except Exception as e_delete:
            print(f"Error deleting data from {TABLE_NAME}: {e_delete}")
            conn.rollback() # Rollback if deletion fails
            return # Exit if we can't clear the table

        # --- Column Mapping (CSV_Column_Name: Model_Field_Name) ---
        # Adjust these mappings if your CSV column names differ.
        # This assumes common Telco Churn dataset column names.
        column_mapping = {
            'customerID': 'customer_id',
            'gender': 'gender',
            'SeniorCitizen': 'senior_citizen', # Often 0 or 1
            'Partner': 'partner',
            'Dependents': 'dependents',
            'tenure': 'tenure',
            'PhoneService': 'phone_service',
            'MultipleLines': 'multiple_lines',
            'InternetService': 'internet_serivce', # Note: model has 'internet_serivce'
            'OnlineSecurity': 'online_security',
            'OnlineBackup': 'online_backup',
            'DeviceProtection': 'device_protection',
            'TechSupport': 'tech_support',
            'StreamingTV': 'streaming_tv',
            'StreamingMovies': 'streaming_movies',
            'Contract': 'contract',
            'PaperlessBilling': 'paperless_billing',
            'PaymentMethod': 'payment_method',
            'MonthlyCharges': 'monthly_charges',
            'TotalCharges': 'total_charges',
            'Churn': 'churn'
        }

        insert_count = 0
        print(f"Starting data insertion into {TABLE_NAME}...")
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Inserting records"):
            record = {}
            record['id'] = uuid.uuid4() # Generate new UUID for each record

            for csv_col, model_col in column_mapping.items():
                if csv_col not in row:
                    print(f"Warning: Column '{csv_col}' not found in CSV row {index}. Skipping this field.")
                    # Handle missing columns: assign None or default, or skip
                    # For now, we'll let it be missing in `record`, SQL insert will fail if NOT NULL
                    # Or, assign a default if appropriate for your schema, e.g., record[model_col] = None
                    continue 
                
                val = row[csv_col]

                if model_col in ['senior_citizen', 'partner', 'dependents', 'phone_service', 'paperless_billing', 'churn']:
                    record[model_col] = to_bool(val)
                elif model_col == 'tenure':
                    record[model_col] = int(val) if pd.notna(val) else 0
                elif model_col in ['monthly_charges', 'total_charges']:
                    record[model_col] = to_decimal(val)
                elif model_col == 'customer_id': 
                    record[model_col] = str(val).strip() if pd.notna(val) else ''
                else: # CharFields
                    record[model_col] = str(val).strip() if pd.notna(val) else ''
            
            # Ensure all model fields are present, even if not in CSV or mapping, to avoid SQL errors
            # This is a basic check; more robust handling might be needed based on your model's NOT NULL constraints
            model_fields = [
                'id', 'customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
                'tenure', 'phone_service', 'multiple_lines', 'internet_serivce',
                'online_security', 'online_backup', 'device_protection', 'tech_support',
                'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing',
                'payment_method', 'monthly_charges', 'total_charges', 'churn'
            ]
            for field in model_fields:
                if field not in record:
                    # Assign a default based on type or if nullable. For simplicity, using common defaults.
                    if field in ['senior_citizen', 'partner', 'dependents', 'phone_service', 'paperless_billing', 'churn']:
                        record[field] = False 
                    elif field == 'tenure':
                        record[field] = 0
                    elif field in ['monthly_charges', 'total_charges']:
                        record[field] = Decimal('0.00')
                    else:
                        record[field] = '' # For CharFields
                    print(f"Warning: Field '{field}' was missing for row {index}, defaulted to '{record[field]}'.")


            # Construct SQL INSERT statement
            columns = ', '.join(record.keys())
            placeholders = ', '.join(['%s'] * len(record))
            sql = f"INSERT INTO {TABLE_NAME} ({columns}) VALUES ({placeholders})"

            # Prepare values for execution, ensuring UUIDs are strings
            values_to_insert = []
            for key in record.keys(): # Ensure order matches columns
                value = record[key]
                if isinstance(value, uuid.UUID):
                    values_to_insert.append(str(value))
                else:
                    values_to_insert.append(value)
            
            savepoint_name = f"sp_{index}"
            try:
                cur.execute(f"SAVEPOINT {savepoint_name}")
                cur.execute(sql, values_to_insert)
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                insert_count += 1
            except Exception as e_insert:
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                print(f"Error inserting record for csv_row_index {index}, customer_id {record.get('customer_id', 'UNKNOWN')}: {e_insert}")
                print(f"Problematic record data: {record}")
                # Optionally, log to a file or take other actions

        conn.commit() # Commit the main transaction with all successful inserts
        print(f"Successfully attempted to insert all records. Total successful inserts: {insert_count} into {TABLE_NAME}.")

    except psycopg2.Error as e_db:
        print(f"Database error: {e_db}")
        if conn:
            conn.rollback()
    except Exception as e_main:
        print(f"An unexpected error occurred: {e_main}")
    finally:
        if conn:
            cur.close()
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()
