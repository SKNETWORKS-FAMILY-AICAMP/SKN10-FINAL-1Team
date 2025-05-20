
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import MinMaxScaler

# 경로 설정
excel_path = "사내_문서_검색_및_코드_설명_배포_기업_리스트.xlsx"
icon_dir = "icons"

# 데이터 로드
df = pd.read_excel(excel_path)

# 아이콘 파일명 매핑
icon_files = [f for f in os.listdir(icon_dir) if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg'}]

def match_icon(service_name, icons, cutoff=0.5):
    import difflib
    name_key = service_name.lower().replace(" ", "")
    matches = difflib.get_close_matches(name_key, [os.path.splitext(f)[0].lower().replace(" ", "") for f in icons], n=1, cutoff=cutoff)
    if matches:
        for icon in icons:
            if os.path.splitext(icon)[0].lower().replace(" ", "") == matches[0]:
                return icon
    return None

df["icon_file"] = df["서비스 이름"].apply(lambda name: match_icon(name, icon_files))
df = df.dropna(subset=["icon_file"])

# 정확성/대중성 점수 임의 부여
def assign_scores(row):
    name = row["서비스 이름"].lower()
    if "refinder" in name or "hancom" in name:
        return 85, 90
    elif "alli" in name:
        return 80, 75
    elif "robi" in name:
        return 88, 70
    elif "goover" in name:
        return 70, 65
    elif "konan" in name:
        return 92, 85
    elif "perplexity" in name:
        return 95, 95
    elif "liner" in name:
        return 80, 90
    elif "docseek" in name:
        return 75, 60
    elif "copilot" in name:
        return 89, 92
    else:
        return 70, 60

df[["Accuracy", "Popularity"]] = df.apply(assign_scores, axis=1, result_type="expand")

# 정규화 점수 (0~100)
scaler = MinMaxScaler(feature_range=(0, 100))
scores_scaled = scaler.fit_transform(df[["Accuracy", "Popularity"]])
df["Rel_Accuracy"] = scores_scaled[:, 0]
df["Rel_Popularity"] = scores_scaled[:, 1]

# Perplexity 중복 제거
df = df.drop_duplicates(subset=["서비스 이름"], keep="first")
df = pd.concat([df[~df["서비스 이름"].str.contains("Perplexity", case=False)], 
                df[df["서비스 이름"].str.contains("Perplexity", case=False)].head(1)])

# 아이콘 사이즈 조절 함수
def custom_icon_size(name):
    if "amazon" in name.lower():
        return 0.12
    else:
        return 0.15

# 시각화 시작
fig, ax = plt.subplots(figsize=(12, 10))
ax.axhline(50, color='gray', linewidth=1, linestyle='--')
ax.axvline(50, color='gray', linewidth=1, linestyle='--')
ax.set_xticks([])
ax.set_yticks([])

# 아이콘 및 텍스트 배치
for _, row in df.iterrows():
    icon_path = os.path.join(icon_dir, row["icon_file"])
    try:
        zoom = custom_icon_size(row["서비스 이름"])
        img = mpimg.imread(icon_path)
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (row["Rel_Accuracy"], row["Rel_Popularity"]), frameon=False)
        ax.add_artist(ab)
        ax.text(row["Rel_Accuracy"], row["Rel_Popularity"] - 6, row["서비스 이름"], ha="center", fontsize=9)
    except Exception as e:
        print(f"이미지 오류: {row['icon_file']} - {e}")

# 외부 레이블 추가
ax.text(-15, 50, "Low Accuracy", fontsize=12, va="center", ha="center", rotation=90, fontweight="bold")
ax.text(115, 50, "High Accuracy", fontsize=12, va="center", ha="center", rotation=270, fontweight="bold")
ax.text(50, 115, "High Popularity", fontsize=12, ha="center", va="bottom", fontweight="bold")
ax.text(50, -15, "Low Popularity", fontsize=12, ha="center", va="top", fontweight="bold")

# 축 범위
ax.set_xlim(-10, 110)
ax.set_ylim(-10, 110)

# 저장
output_path = "final_service_quadrant_outside_labels.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print("✅ 그래프 저장 완료:", output_path)
