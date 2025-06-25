# pipeline_orchestrator.py
import subprocess
import sys
import os
import uuid

def upload_and_get_key(file_path, bucket_name='hinton-csv-upload'):
    """
    Uploads a file to S3 and returns the object key. This is the first step of the
    user-facing workflow, designed to be fast.

    Args:
        file_path (str): The local path to the CSV file to be uploaded.
        bucket_name (str): The S3 bucket to use.

    Returns:
        str: The S3 object key for the uploaded file.
        str: An error message if the upload fails, otherwise None.
    """
    base_dir = os.path.dirname(__file__)
    
    if not os.path.exists(file_path):
        error_message = f"Error: Input file not found at {file_path}"
        print(error_message, file=sys.stderr)
        return None, error_message

    try:
        print("--- [1/1] Uploading file to S3...")
        upload_script_path = os.path.join(base_dir, 'upload_script.py')
        upload_command = [
            sys.executable, upload_script_path,
            "--file", file_path,
            "--bucket", bucket_name
        ]
        upload_result = subprocess.run(upload_command, capture_output=True, text=True, check=True, encoding='utf-8')
        s3_object_key = upload_result.stdout.strip()
        if not s3_object_key:
            raise Exception(f"Upload script did not return an S3 object key. Error: {upload_result.stderr}")
        print(f"Upload successful. S3 Key: {s3_object_key}")
        return s3_object_key, None

    except subprocess.CalledProcessError as e:
        error_message = f"Upload script failed.\nCommand: {' '.join(e.cmd)}\nStderr: {e.stderr}"
        print(error_message, file=sys.stderr)
        return None, error_message
    except Exception as e:
        print(f"An unexpected error occurred during upload: {e}", file=sys.stderr)
        return None, str(e)

def run_analysis_pipeline(file_path, bucket_name='hinton-csv-upload'):
    """
    Orchestrates the S3 upload, download, and LLM analysis pipeline using robust subprocess calls.
    This function is designed to be safely called from any Python application, including Django.

    Args:
        file_path (str): The local path to the CSV file to be analyzed.
        bucket_name (str): The S3 bucket to use for the process.

    Returns:
        str: The final analysis result from the LLM.
        str: An error message if any step fails, otherwise None.
    """
    base_dir = os.path.dirname(__file__)
    
    # --- Validate file existence ---
    if not os.path.exists(file_path):
        error_message = f"Error: Input file not found at {file_path}"
        print(error_message, file=sys.stderr)
        return None, error_message

    try:
        # --- Step 1: Upload and get S3 Object Key ---
        print("--- [1/3] Uploading file to S3...")
        upload_script_path = os.path.join(base_dir, 'upload_script.py')
        upload_command = [
            sys.executable, upload_script_path, 
            "--file", file_path, 
            "--bucket", bucket_name
        ]
        upload_result = subprocess.run(upload_command, capture_output=True, text=True, check=True, encoding='utf-8')
        s3_object_key = upload_result.stdout.strip()
        if not s3_object_key:
            raise Exception(f"Upload script did not return an S3 object key. Error: {upload_result.stderr}")
        print(f"Upload successful. S3 Key: {s3_object_key}")

        # --- Step 2: Download and get Temporary File Path ---
        print("--- [2/3] Downloading file from S3...")

        # Generate a unique path for the temporary downloaded file
        temp_file_name = f"temp_download_{uuid.uuid4()}.csv"
        temp_file_path = os.path.join(base_dir, temp_file_name)

        download_script_path = os.path.join(base_dir, 'download_script.py')
        download_command = [
            sys.executable, download_script_path,
            "--key", s3_object_key,
            "--bucket", bucket_name,
            "--output_path", temp_file_path
        ]
        
        # The download script saves the file to the path and does not print to stdout.
        # We avoid capturing output to prevent potential deadlocks with pipes.
        subprocess.run(download_command, check=True, encoding='utf-8')
        
        if not os.path.exists(temp_file_path):
            raise Exception(f"Download script ran but did not create the file at {temp_file_path}.")
        
        print(f"Download successful. Temp file: {temp_file_path}")

        # --- Step 3: Run LLM Analysis and get Result ---
        print("--- [3/3] Running LLM Analysis...")
        llm_script_path = os.path.join(base_dir, 'llm_script.py')
        llm_command = [sys.executable, llm_script_path, "--file", temp_file_path]
        # Use PIPE for stdout/stderr to avoid blocking on large outputs
        llm_result = subprocess.run(llm_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, encoding='utf-8')
        analysis_result = llm_result.stdout
        print("Analysis successful.")

        return analysis_result, None

    except subprocess.CalledProcessError as e:
        error_message = f"A step in the pipeline failed.\n"
        error_message += f"Command: {' '.join(e.cmd)}\n"
        error_message += f"Return Code: {e.returncode}\n"
        error_message += f"Stdout: {e.stdout}\n"
        error_message += f"Stderr: {e.stderr}"
        print(error_message, file=sys.stderr)
        return None, error_message
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return None, str(e)

if __name__ == '__main__':
    # This is an example of how to call the function.
    # To run this, execute: python pipeline_orchestrator.py
    local_csv_file = 'test_data.csv' # Make sure this file exists
    print(f"Starting pipeline for file: {local_csv_file}")
    
    final_result, error = run_analysis_pipeline(local_csv_file)
    
    if error:
        print("\n--- PIPELINE FAILED ---")
        print(error)
    else:
        print("\n--- PIPELINE SUCCESSFUL ---")
        print("\n[Final Analysis Result]")
        print(final_result)
