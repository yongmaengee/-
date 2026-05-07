"""03. 누적 날씨 피처 vs 위험등급/중대재해 — 모델 입력 피처 후보 식별"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

for f in ['NanumGothic', 'Noto Sans CJK KR', 'AppleGothic', 'Malgun Gothic']:
    try:
        mpl.rcParams['font.family'] = f
        break
    except Exception:
        pass
mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('RIA_최종통합피쳐셋.csv', encoding='utf-8-sig')
df['주의이상'] = df['위험등급'].isin(['주의', '경계']).astype(int)

# 모든 누적/날씨 수치 피처
weather_cum = ['평균기온(°C)', '최고기온(°C)', '최저기온(°C)', '일강수량(mm)', '평균 풍속(m/s)',
               'ATD_7d', 'TSI_3d', '누적강수_7d', '누적풍속_7d', '기온변동성_7d',
               '폭염_연속_7d', '한파_연속_7d', '강수_연속_7d',
               '날씨_위험도_복합', 'magnitude_score',
               'risk_heat_process', 'risk_rain_progress', 'risk_wind_scaffold', 'risk_cold_concrete',
               'weather_scale_interact', 'heat_foreign_exposure', 'heat_elderly_exposure']

print("=" * 100)
print("[A] 누적 날씨 피처 vs 중대재해(0/1) 점이상관 + 두 그룹 평균 비교")
rows = []
for c in weather_cum:
    x0 = df.loc[df['중대재해'] == 0, c]
    x1 = df.loc[df['중대재해'] == 1, c]
    r, p = stats.pointbiserialr(df['중대재해'], df[c])
    lift = (x1.mean() - x0.mean()) / (x0.std() + 1e-9)  # 표준화 차이 (Cohen's d 근사)
    rows.append({
        'feature': c,
        '정상_평균': round(x0.mean(), 3),
        '중대_평균': round(x1.mean(), 3),
        '표준화차이': round(lift, 3),
        '점이상관r': round(r, 4),
        'p값': f"{p:.2e}",
    })
res = pd.DataFrame(rows).sort_values('점이상관r', key=lambda s: s.abs(), ascending=False)
print(res.to_string(index=False))

print("\n" + "=" * 100)
print("[B] 누적 날씨 피처 vs 주의이상(주의+경계) 점이상관")
rows = []
for c in weather_cum:
    r, p = stats.pointbiserialr(df['주의이상'], df[c])
    rows.append({'feature': c, '점이상관r': round(r, 4), 'p값': f"{p:.2e}"})
res2 = pd.DataFrame(rows).sort_values('점이상관r', key=lambda s: s.abs(), ascending=False)
print(res2.to_string(index=False))

print("\n" + "=" * 100)
print("[C] 누적 피처 vs C_RPI 스피어만 상관")
rows = []
for c in weather_cum:
    r, p = stats.spearmanr(df[c], df['C_RPI'])
    rows.append({'feature': c, '스피어만r': round(r, 4), 'p값': f"{p:.2e}"})
res3 = pd.DataFrame(rows).sort_values('스피어만r', key=lambda s: s.abs(), ascending=False)
print(res3.to_string(index=False))

# 분위 구간별 위험률 — 누적 피처가 어떤 구간에서 위험이 급증하는지
print("\n" + "=" * 100)
print("[D] 핵심 누적 피처의 분위 구간(quartile)별 중대재해율(%) — 비선형 효과 탐색")
key_feats = ['누적강수_7d', '폭염_연속_7d', '강수_연속_7d', '한파_연속_7d',
             '기온변동성_7d', 'ATD_7d', 'TSI_3d', '누적풍속_7d',
             '날씨_위험도_복합', 'magnitude_score', 'C_RPI']
for c in key_feats:
    try:
        if df[c].nunique() < 8:
            grouped = df.groupby(c).agg(건수=('중대재해', 'size'), 중대재해율=('중대재해', 'mean'),
                                       주의이상률=('주의이상', 'mean')).reset_index()
            grouped['중대재해율(%)'] = (grouped['중대재해율'] * 100).round(2)
            grouped['주의이상률(%)'] = (grouped['주의이상률'] * 100).round(2)
            print(f"\n--- {c} (정수 카운트) ---")
            print(grouped[[c, '건수', '중대재해율(%)', '주의이상률(%)']].to_string(index=False))
        else:
            df['_q'] = pd.qcut(df[c], q=5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
            grouped = df.groupby('_q', observed=True).agg(
                건수=('중대재해', 'size'),
                범위_min=(c, 'min'), 범위_max=(c, 'max'),
                중대재해율=('중대재해', 'mean'),
                주의이상률=('주의이상', 'mean'),
                평균C_RPI=('C_RPI', 'mean')).reset_index()
            grouped['중대재해율(%)'] = (grouped['중대재해율'] * 100).round(2)
            grouped['주의이상률(%)'] = (grouped['주의이상률'] * 100).round(2)
            grouped['평균C_RPI'] = grouped['평균C_RPI'].round(2)
            print(f"\n--- {c} ---")
            print(grouped[['_q', '건수', '범위_min', '범위_max', '중대재해율(%)', '주의이상률(%)', '평균C_RPI']]
                  .to_string(index=False))
    except Exception as e:
        print(f"  {c}: 처리 실패 {e}")
df.drop(columns=['_q'], inplace=True, errors='ignore')

# 임계값 기반: "위험 신호" 후보
print("\n" + "=" * 100)
print("[E] 단일 임계값 룰 — 'IF X >= 임계 THEN 중대재해율'  (베이스라인 4.11%)")
print("    Lift = 임계조건의 중대재해율 / 베이스라인")
base = df['중대재해'].mean()
rules = [
    ('폭염_연속_7d',     [1, 2, 3]),
    ('한파_연속_7d',     [1, 2, 3]),
    ('강수_연속_7d',     [2, 3, 4, 5]),
    ('누적강수_7d',     [50, 100, 150, 200]),
    ('기온변동성_7d',    [3, 4, 5]),
    ('TSI_3d',         [13, 15, 17]),
    ('일강수량(mm)',    [10, 30, 50]),
    ('평균 풍속(m/s)',  [3, 4, 5]),
    ('최고기온(°C)',    [30, 33, 35]),
    ('최저기온(°C)',    [-5, -10]),
    ('날씨_위험도_복합', [10, 13, 15]),
    ('magnitude_score', [0.4, 0.5, 0.6]),
]
for col, thrs in rules:
    print(f"\n--- {col} ---")
    for t in thrs:
        sub = df[df[col] >= t]
        n = len(sub)
        if n == 0:
            continue
        rate = sub['중대재해'].mean()
        warn = sub['주의이상'].mean()
        print(f"  >= {t:>6}: n={n:>6}  중대재해율={rate*100:5.2f}%  Lift={rate/base:4.2f}x  주의이상률={warn*100:5.2f}%")

# 시각화: 분위별 중대재해율 막대그래프
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
plot_feats = ['누적강수_7d', '폭염_연속_7d', '한파_연속_7d', '강수_연속_7d',
              '기온변동성_7d', 'ATD_7d', 'TSI_3d', '날씨_위험도_복합', 'magnitude_score']
for ax, c in zip(axes.flatten(), plot_feats):
    try:
        if df[c].nunique() < 8:
            g = df.groupby(c)['중대재해'].mean() * 100
            ax.bar(g.index.astype(str), g.values, color='salmon')
        else:
            df['_q'] = pd.qcut(df[c], q=5, duplicates='drop')
            g = df.groupby('_q', observed=True)['중대재해'].mean() * 100
            ax.bar(range(len(g)), g.values, color='salmon')
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels([f"Q{i+1}" for i in range(len(g))])
            df.drop(columns=['_q'], inplace=True)
        ax.axhline(base * 100, color='gray', linestyle='--', label=f'평균 {base*100:.2f}%')
        ax.set_title(c)
        ax.set_ylabel('중대재해율(%)')
        ax.legend(fontsize=8)
    except Exception as e:
        ax.set_title(f"{c} (failed)")
plt.tight_layout()
plt.savefig('plot_03_cumulative_weather.png', dpi=110, bbox_inches='tight')
plt.close()
print("\n[저장] plot_03_cumulative_weather.png")
