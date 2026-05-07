"""04. 사고 유형별 × 누적 날씨/공종 민감도 프로파일링 — 이슈리스트 룰 후보"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

for f in ['NanumGothic', 'Noto Sans CJK KR', 'AppleGothic', 'Malgun Gothic']:
    try:
        mpl.rcParams['font.family'] = f
        break
    except Exception:
        pass
mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('RIA_최종통합피쳐셋.csv', encoding='utf-8-sig')
df['사고일시'] = pd.to_datetime(df['사고일시'])
df['월'] = df['사고일시'].dt.month

acc_types = ['추락_낙하', '붕괴_도괴', '구조물_균열', '침수_수해', '장비_충돌', '화재_폭발']
weather_feats = ['최고기온(°C)', '최저기온(°C)', '일강수량(mm)', '평균 풍속(m/s)',
                 '누적강수_7d', '누적풍속_7d', '기온변동성_7d',
                 '폭염_연속_7d', '한파_연속_7d', '강수_연속_7d',
                 'ATD_7d', 'TSI_3d']

# A. 사고유형별 평균 날씨조건
print("=" * 100)
print("[A] 사고 유형별 평균 날씨조건 (해당 유형=1 발생 표본만)")
rows = []
for t in acc_types:
    sub = df[df[t] == 1]
    row = {'사고유형': t, '건수': len(sub)}
    for w in weather_feats:
        row[w] = round(sub[w].mean(), 2)
    rows.append(row)
# 전체 평균 추가
row = {'사고유형': '전체평균', '건수': len(df)}
for w in weather_feats:
    row[w] = round(df[w].mean(), 2)
rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

# B. 표준화된 차이 — 어떤 사고가 어떤 날씨에서 더 자주 터지는가
print("\n" + "=" * 100)
print("[B] 표준화 차이 ((해당사고 평균 - 전체 평균) / 전체 표준편차)")
print("    → 양수: 그 사고는 해당 날씨조건이 더 강할 때 더 자주 발생  /  음수: 반대")
rows = []
all_mean = df[weather_feats].mean()
all_std = df[weather_feats].std()
for t in acc_types:
    sub = df[df[t] == 1]
    diff = ((sub[weather_feats].mean() - all_mean) / all_std).round(3)
    diff['사고유형'] = t
    diff['건수'] = len(sub)
    rows.append(diff)
res = pd.DataFrame(rows).set_index('사고유형')
cols = ['건수'] + weather_feats
print(res[cols].to_string())

# C. 날씨 카테고리별 사고유형 발생률 (행별 정규화)
print("\n" + "=" * 100)
print("[C] 날씨 카테고리별 각 사고유형 발생률(%) (행=날씨, 열=사고유형)")
rows = []
for w in df['날씨'].unique():
    sub = df[df['날씨'] == w]
    row = {'날씨': w, '건수': len(sub)}
    for t in acc_types:
        row[t] = round(sub[t].mean() * 100, 2)
    rows.append(row)
print(pd.DataFrame(rows).sort_values('건수', ascending=False).to_string(index=False))

# D. 월별 사고유형 발생률 — 시즌 패턴
print("\n" + "=" * 100)
print("[D] 월별 사고유형 발생률(%)")
piv = df.groupby('월')[acc_types].mean().round(3) * 100
piv.columns = [c + '(%)' for c in piv.columns]
print(piv.round(2).to_string())

# E. 공종별 × 사고유형 (상위 공종만)
print("\n" + "=" * 100)
print("[E] 주요 공종별 사고유형 발생률(%) — 표본 ≥ 500건")
big = df.groupby('공종(소분류)').filter(lambda x: len(x) >= 500)
piv = big.groupby('공종(소분류)')[acc_types].mean() * 100
piv['n'] = big.groupby('공종(소분류)').size()
piv = piv.sort_values('n', ascending=False).round(2)
print(piv.to_string())

# F. 누적 날씨 임계 × 공종 조합 — 이슈리스트 룰 후보 (Lift 기준)
print("\n" + "=" * 100)
print("[F] [공종 × 누적날씨 임계] 조합 — 중대재해율 / 사고유형률 Lift  (n>=100, Lift>=1.3)")
print("=" * 100)
base_severe = df['중대재해'].mean()
big_works = df['공종(소분류)'].value_counts()
big_works = big_works[big_works >= 800].index.tolist()

conditions = [
    ('폭염_연속_7d>=2', df['폭염_연속_7d'] >= 2),
    ('폭염_연속_7d>=3', df['폭염_연속_7d'] >= 3),
    ('한파_연속_7d>=1', df['한파_연속_7d'] >= 1),
    ('한파_연속_7d>=2', df['한파_연속_7d'] >= 2),
    ('강수_연속_7d>=3', df['강수_연속_7d'] >= 3),
    ('누적강수_7d>=100', df['누적강수_7d'] >= 100),
    ('일강수량>=30', df['일강수량(mm)'] >= 30),
    ('평균풍속>=4', df['평균 풍속(m/s)'] >= 4),
    ('최고기온>=33', df['최고기온(°C)'] >= 33),
    ('최저기온<=-5', df['최저기온(°C)'] <= -5),
    ('기온변동성>=4', df['기온변동성_7d'] >= 4),
]

issue_list = []
for cond_name, mask in conditions:
    for w in big_works:
        wmask = (df['공종(소분류)'] == w) & mask
        n = wmask.sum()
        if n < 80:
            continue
        sub = df[wmask]
        rate = sub['중대재해'].mean()
        lift = rate / base_severe if base_severe > 0 else 0
        if lift >= 1.3 and rate >= 0.05:
            issue_list.append({
                '조건': cond_name, '공종': w, 'n': int(n),
                '중대재해율(%)': round(rate * 100, 2), 'Lift': round(lift, 2),
                '주_사고유형': max(acc_types, key=lambda t: sub[t].mean()),
            })

if issue_list:
    res = pd.DataFrame(issue_list).sort_values(['Lift', 'n'], ascending=[False, False])
    print(res.to_string(index=False))
else:
    print("  (조건/공종 조합에서 Lift>=1.3, n>=80 케이스 없음)")

# G. 시각화: 사고유형 × 누적날씨 표준화차이 히트맵
fig, ax = plt.subplots(figsize=(14, 5))
mat = res = []
for t in acc_types:
    sub = df[df[t] == 1]
    diff = ((sub[weather_feats].mean() - all_mean) / all_std).values
    mat.append(diff)
mat = np.array(mat)
im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax.set_yticks(range(len(acc_types)))
ax.set_yticklabels(acc_types)
ax.set_xticks(range(len(weather_feats)))
ax.set_xticklabels(weather_feats, rotation=45, ha='right')
plt.colorbar(im, ax=ax, label='표준화 차이 (z-score)')
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                color='white' if abs(v) > 0.3 else 'black', fontsize=8)
ax.set_title('사고 유형 × 누적/현재 날씨 — 표준화 평균 차이')
plt.tight_layout()
plt.savefig('plot_04_accident_weather_zscore.png', dpi=110, bbox_inches='tight')
plt.close()
print("\n[저장] plot_04_accident_weather_zscore.png")
