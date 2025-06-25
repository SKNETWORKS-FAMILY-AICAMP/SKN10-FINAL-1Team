# upload_script.py
import boto3
import os
import sys
import argparse
import uuid
from dotenv import load_dotenv

def upload_file(file_path, bucket_name):
    """
    Uploads a file to S3 and prints the URL and object key to stdout.
    """
    try:
        # Load .env from the same directory or backend subdirectory
        dotenv_path = os.path.join(os.path.dirname(__file__), 'backend', '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
        else:
            load_dotenv()

        object_name = f"{uuid.uuid4()}-{os.path.basename(file_path)}"

        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )
        
        s3_client.upload_file(file_path, bucket_name, object_name)
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
        
        # Print the results to stdout, one per line
        print(object_name)
        
        sys.exit(0)

    except Exception as e:
        print(f"Error during upload: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S3 Uploader Script")
    parser.add_argument("--file", required=True, help="Path to the file to upload")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    args = parser.parse_args()
    
    upload_file(args.file, args.bucket)
