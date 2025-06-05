import csv
import random

def add_customerid_to_csv(input_filename="sample_users_30.csv", output_filename="sample_users_30_with_ids.csv"):
    """
    CSV 파일을 읽어 'customerid' 컬럼을 추가하고, 각 행에 5자리 무작위 ID를 할당한 후
    새로운 CSV 파일로 저장합니다.

    Args:
        input_filename (str): 원본 CSV 파일 이름
        output_filename (str): customerid가 추가된 내용을 저장할 새 CSV 파일 이름
    """
    rows = []
    generated_ids = set()

    try:
        with open(input_filename, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # 헤더 읽기
            rows.append(['customerid'] + header) # 새 헤더 생성

            for row in reader:
                if not row: # 빈 줄 처리
                    rows.append([])
                    continue

                while True:
                    # 5자리 무작위 숫자 ID 생성 (10000 ~ 99999)
                    new_id = random.randint(10000, 99999)
                    if new_id not in generated_ids:
                        generated_ids.add(new_id)
                        break
                rows.append([str(new_id)] + row)

        with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(rows)

        print(f"'{output_filename}' 파일이 성공적으로 생성되었습니다.")
        print(f"총 {len(rows) -1} 개의 데이터 행에 ID가 할당되었습니다 (헤더 제외).")

    except FileNotFoundError:
        print(f"오류: '{input_filename}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    # 현재 작업 디렉토리에 sample_users_30.csv 파일이 있다고 가정합니다.
    # 필요시 파일 경로를 수정하세요.
    input_csv_path = "c:\\development\\github\\SKN10-FINAL-1Team\\sample_users_30.csv"
    output_csv_path = "c:\\development\\github\\SKN10-FINAL-1Team\\sample_users_30_with_ids.csv"
    
    add_customerid_to_csv(input_csv_path, output_csv_path)