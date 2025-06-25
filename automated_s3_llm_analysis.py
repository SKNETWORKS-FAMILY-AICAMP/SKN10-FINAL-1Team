import os
import uuid
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="S3-LLM Automation Orchestrator")
    parser.add_argument('--file-path', type=str, default='test_data.csv', help='Path to the file to upload')
    parser.add_argument('--bucket-name', type=str, default='hinton-csv-upload', help='S3 bucket name')
    args = parser.parse_args()

    # --- 1. Upload ---
    print("\n--- 1. S3 File Upload ---"); sys.stdout.flush()
    file_path = args.file_path
    bucket_name = args.bucket_name

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    upload_command = [sys.executable, "upload_script.py", "--file", file_path, "--bucket", bucket_name]
    print(f"Executing: {' '.join(upload_command)}"); sys.stdout.flush()

    upload_stdout_file = f"temp_{uuid.uuid4()}.out"
    upload_stderr_file = f"temp_{uuid.uuid4()}.err"

    try:
        with open(upload_stdout_file, 'w') as f_out, open(upload_stderr_file, 'w') as f_err:
            process = subprocess.Popen(upload_command, stdout=f_out, stderr=f_err)
            process.wait()

        with open(upload_stderr_file, 'r') as f_err:
            stderr = f_err.read()
        if process.returncode != 0:
            print("\nUpload script failed.", file=sys.stderr)
            print(f"STDERR:\n{stderr}", file=sys.stderr)
            sys.exit(1)

        with open(upload_stdout_file, 'r') as f_out:
            upload_stdout = f_out.read()
    finally:
        if os.path.exists(upload_stdout_file): os.remove(upload_stdout_file)
        if os.path.exists(upload_stderr_file): os.remove(upload_stderr_file)

    output_lines = upload_stdout.strip().split('\n')
    if len(output_lines) < 2:
        print("\nCould not parse upload script output.", file=sys.stderr)
        print(f"STDOUT: {upload_stdout}", file=sys.stderr)
        sys.exit(1)

    s3_url, object_name = output_lines[0], output_lines[1]
    print(f"Upload successful. URL: {s3_url}")

    # --- 2. Download ---
    print("\n--- 2. S3 File Download ---"); sys.stdout.flush()
    temp_file_path = f"temp_{uuid.uuid4()}.csv"
    download_command = [sys.executable, "download_script.py", "--bucket", bucket_name, "--key", object_name, "--output", temp_file_path]
    print(f"Executing: {' '.join(download_command)}"); sys.stdout.flush()

    download_stderr_file = f"temp_{uuid.uuid4()}.err"
    try:
        with open(download_stderr_file, 'w') as f_err:
            process = subprocess.Popen(download_command, stdout=subprocess.DEVNULL, stderr=f_err)
            process.wait()

        with open(download_stderr_file, 'r') as f_err:
            stderr = f_err.read()
        if process.returncode != 0:
            print("\nDownload script failed.", file=sys.stderr)
            print(f"STDERR:\n{stderr}", file=sys.stderr)
            sys.exit(1)
    finally:
        if os.path.exists(download_stderr_file): os.remove(download_stderr_file)

    if not os.path.exists(temp_file_path):
        print("\nDownload script seemed to succeed, but output file not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Download successful. File at: {temp_file_path}")

    # --- 3. LLM Analysis ---
    print("\n--- 3. LLM Analysis ---"); sys.stdout.flush()
    llm_command = [sys.executable, "llm_script.py", "--file", temp_file_path]
    print(f"Executing: {' '.join(llm_command)}"); sys.stdout.flush()

    llm_stdout_file = f"temp_{uuid.uuid4()}.out"
    llm_stderr_file = f"temp_{uuid.uuid4()}.err"

    try:
        with open(llm_stdout_file, 'w') as f_out, open(llm_stderr_file, 'w') as f_err:
            process = subprocess.Popen(llm_command, stdout=f_out, stderr=f_err)
            process.wait(timeout=300)

        with open(llm_stderr_file, 'r') as f_err:
            stderr = f_err.read()
        if process.returncode != 0:
            print("\nLLM analysis script failed.", file=sys.stderr)
            print(f"STDERR:\n{stderr}", file=sys.stderr)
            sys.exit(1)

        with open(llm_stdout_file, 'r') as f_out:
            analysis_output = f_out.read()
    except subprocess.TimeoutExpired:
        print("\nLLM analysis script timed out.", file=sys.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(llm_stdout_file): os.remove(llm_stdout_file)
        if os.path.exists(llm_stderr_file): os.remove(llm_stderr_file)

    print("\n--- 4. LLM Analysis Result ---"); sys.stdout.flush()
    print(analysis_output)

    with open("analysis_result.txt", "w", encoding="utf-8") as f:
        f.write(analysis_output)
    print("\nAnalysis result saved to analysis_result.txt")

if __name__ == "__main__":
    main()



