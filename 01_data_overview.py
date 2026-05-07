"""01. 데이터 기본 구조 / 품질 진단"""
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 200)

df = pd.read_csv('RIA_최종통합피쳐셋.csv', encoding='utf-8-sig')
df['사고일시'] = pd.to_datetime(df['사고일시'], errors='coerce')

print("=" * 80)
print(f"[1] shape: {df.shape}")
print(f"[2] 기간: {df['사고일시'].min()}  ~  {df['사고일시'].max()}")
print(f"[3] 사고일시 결측: {df['사고일시'].isna().sum()}")

print("\n" + "=" * 80)
print("[4] 컬럼별 dtype + 결측 + 유니크 수")
info = pd.DataFrame({
    'dtype': df.dtypes.astype(str),
    'nulls': df.isna().sum(),
    'null_pct': (df.isna().mean() * 100).round(2),
    'nunique': df.nunique(),
})
print(info.to_string())

print("\n" + "=" * 80)
print("[5] 카테고리형 변수 분포")
for col in ['시도구분', '날씨', '공사대분류', '공종(소분류)', '위험등급']:
    vc = df[col].value_counts(dropna=False)
    print(f"\n--- {col} (n={df[col].nunique()}) ---")
    print(vc.head(15).to_string())

print("\n" + "=" * 80)
print("[6] 핵심 수치형 변수 분포")
num_cols = ['평균기온(°C)', '최고기온(°C)', '최저기온(°C)', '일강수량(mm)', '평균 풍속(m/s)',
            'ATD_7d', 'TSI_3d', '누적강수_7d', '누적풍속_7d', '기온변동성_7d',
            '폭염_연속_7d', '한파_연속_7d', '강수_연속_7d', '강풍_연속_7d',
            '날씨_위험도_복합', 'magnitude_score', 'C_RPI',
            '사망자', '부상자', '총재해자', '외국인비율', '고령자비율', '중대재해']
print(df[num_cols].describe().T.round(3).to_string())

print("\n" + "=" * 80)
print("[7] 타깃 라벨 균형")
print(f"중대재해 비율: {df['중대재해'].mean():.4f}  ({df['중대재해'].sum()} / {len(df)})")
print(f"사망자>=1 비율: {(df['사망자']>=1).mean():.4f}")
print("\n위험등급 분포:")
print(df['위험등급'].value_counts(normalize=True).round(4).to_string())

print("\n" + "=" * 80)
print("[8] 사고 유형(0/1 플래그) 발생률")
acc_types = ['추락_낙하', '붕괴_도괴', '구조물_균열', '침수_수해', '장비_충돌', '화재_폭발', '기타']
for c in acc_types:
    print(f"  {c}: {df[c].mean():.4f}  ({int(df[c].sum())}건)")
