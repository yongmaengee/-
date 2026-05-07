"""02. 날씨 카테고리·계절성 vs 사고 빈도/심각도"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 (있으면 사용)
for f in ['NanumGothic', 'Noto Sans CJK KR', 'AppleGothic', 'Malgun Gothic']:
    try:
        mpl.rcParams['font.family'] = f
        break
    except Exception:
        pass
mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('RIA_최종통합피쳐셋.csv', encoding='utf-8-sig')
df['사고일시'] = pd.to_datetime(df['사고일시'])
df['연'] = df['사고일시'].dt.year
df['월'] = df['사고일시'].dt.month
df['시'] = df['사고일시'].dt.hour
df['요일'] = df['사고일시'].dt.dayofweek  # 0=Mon
df['계절'] = df['월'].map(lambda m: '봄(3-5)' if 3<=m<=5 else '여름(6-8)' if 6<=m<=8
                       else '가을(9-11)' if 9<=m<=11 else '겨울(12-2)')
df['주의이상'] = df['위험등급'].isin(['주의', '경계']).astype(int)

n_total = len(df)

def severity_table(g):
    return pd.DataFrame({
        '건수': g.size(),
        '비중(%)': (g.size() / n_total * 100).round(2),
        '평균C_RPI': g['C_RPI'].mean().round(2),
        '중대재해율(%)': (g['중대재해'].mean() * 100).round(2),
        '평균사망자': g['사망자'].mean().round(3),
        '평균총재해자': g['총재해자'].mean().round(3),
        '주의이상률(%)': (g['주의이상'].mean() * 100).round(2),
    })

print("=" * 80)
print("[A] 날씨 카테고리별 사고 심각도")
g = df.groupby('날씨')
out = severity_table(g).sort_values('중대재해율(%)', ascending=False)
print(out.to_string())

print("\n" + "=" * 80)
print("[B] 계절별 사고 심각도")
g = df.groupby('계절')
print(severity_table(g).reindex(['봄(3-5)', '여름(6-8)', '가을(9-11)', '겨울(12-2)']).to_string())

print("\n" + "=" * 80)
print("[C] 월별 사고 심각도")
g = df.groupby('월')
print(severity_table(g).to_string())

print("\n" + "=" * 80)
print("[D] 시간대별 사고 심각도 (top 10)")
g = df.groupby('시')
out = severity_table(g)
print(out.sort_values('건수', ascending=False).head(12).to_string())

print("\n" + "=" * 80)
print("[E] 요일별 사고 심각도")
labels = ['월', '화', '수', '목', '금', '토', '일']
g = df.groupby('요일')
out = severity_table(g)
out.index = [labels[i] for i in out.index]
print(out.to_string())

print("\n" + "=" * 80)
print("[F] 시도구분별 (상위 10)")
g = df.groupby('시도구분')
out = severity_table(g).sort_values('건수', ascending=False)
print(out.head(10).to_string())

print("\n" + "=" * 80)
print("[G] 공사대분류별 사고 심각도")
g = df.groupby('공사대분류')
print(severity_table(g).sort_values('중대재해율(%)', ascending=False).to_string())

print("\n" + "=" * 80)
print("[H] 공종(소분류) 상위 10 - 중대재해율 순")
g = df.groupby('공종(소분류)').filter(lambda x: len(x) >= 200).groupby('공종(소분류)')
out = severity_table(g).sort_values('중대재해율(%)', ascending=False)
print(out.head(15).to_string())

# 시각화 1: 월×날씨 히트맵 (사고 빈도)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
piv = pd.crosstab(df['월'], df['날씨'])
im = axes[0].imshow(piv.values, aspect='auto', cmap='YlOrRd')
axes[0].set_xticks(range(len(piv.columns)))
axes[0].set_xticklabels(piv.columns, rotation=45)
axes[0].set_yticks(range(len(piv.index)))
axes[0].set_yticklabels(piv.index)
axes[0].set_title('월 × 날씨 사고 건수')
plt.colorbar(im, ax=axes[0])
for i in range(len(piv.index)):
    for j in range(len(piv.columns)):
        axes[0].text(j, i, piv.values[i, j], ha='center', va='center', fontsize=8)

# 월×날씨 중대재해율
piv2 = df.pivot_table(index='월', columns='날씨', values='중대재해', aggfunc='mean') * 100
im2 = axes[1].imshow(piv2.values, aspect='auto', cmap='Reds')
axes[1].set_xticks(range(len(piv2.columns)))
axes[1].set_xticklabels(piv2.columns, rotation=45)
axes[1].set_yticks(range(len(piv2.index)))
axes[1].set_yticklabels(piv2.index)
axes[1].set_title('월 × 날씨 중대재해율(%)')
plt.colorbar(im2, ax=axes[1])
for i in range(piv2.shape[0]):
    for j in range(piv2.shape[1]):
        v = piv2.values[i, j]
        if not np.isnan(v):
            axes[1].text(j, i, f"{v:.1f}", ha='center', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('plot_02_weather_month.png', dpi=110, bbox_inches='tight')
plt.close()
print("\n[저장] plot_02_weather_month.png")
