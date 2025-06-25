import os
import boto3
from botocore.exceptions import NoCredentialsError
import uuid
from dotenv import load_dotenv

def upload_file_to_s3(file_path, bucket_name, object_name=None):
    """
    S3 버킷에 파일을 업로드하고 URL을 반환합니다.
    """
    # AWS 자격 증명 설정 (환경 변수에서 가져옴)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )
    
    # 객체 이름이 지정되지 않은 경우 파일 이름 사용
    if object_name is None:
        object_name = os.path.basename(file_path)
    
    # 고유한 파일 이름 생성 (충돌 방지)
    object_name = f"{uuid.uuid4()}-{object_name}"
    
    try:
        # 파일 업로드
        s3_client.upload_file(file_path, bucket_name, object_name)
        
        # S3 URL 생성
        url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
        print(f"파일이 성공적으로 업로드되었습니다: {url}")
        return url
    except NoCredentialsError:
        print("AWS 자격 증명을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"S3 업로드 중 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    # .env 파일에서 환경 변수 로드
    load_dotenv('backend/.env')
    
    # 환경 변수 확인
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    region = os.environ.get('AWS_REGION')
    bucket = os.environ.get('AWS_S3_BUCKET_NAME', 'hinton-csv-upload')
    
    print(f"AWS_ACCESS_KEY_ID: {access_key[:4] + '****' + access_key[-4:] if access_key else 'Not set'}")
    print(f"AWS_SECRET_ACCESS_KEY: {'*' * 12}")
    print(f"AWS_REGION: {region}")
    print(f"AWS_S3_BUCKET_NAME: {bucket}")
    
    # 테스트 파일 경로
    file_path = "test_data.csv"
    
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
    else:
        # S3에 파일 업로드
        upload_file_to_s3(file_path, bucket)
