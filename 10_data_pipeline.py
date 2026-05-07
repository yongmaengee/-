"""Phase 0 — 데이터 파이프라인 + sanity check
- 평년 근사 (옵션 A): 7년 doy 평균
- 14일 윈도우 추출
- Negative sampling (시도/공종 prior)
- 라벨 통합 (메인 = 중대재해 binary)
"""
import pandas as pd
import numpy as np
import os

W_PATH = '/home/nuri5/바탕화면/공모전/national_weather_avg_2019_2026 2.csv'
R_PATH = '/home/nuri5/바탕화면/공모전/RIA_최종통합피쳐셋.csv'
OUT_DIR = '/home/nuri5/바탕화면/공모전/cache'
os.makedirs(OUT_DIR, exist_ok=True)

T_WINDOW = 14
NEG_RATIO = 3
RNG = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────
# 1. 외부 일별 날씨 + 평년편차 (doy 평균 기반, 옵션 A)
# ─────────────────────────────────────────────────────────────────
w = pd.read_csv(W_PATH, encoding='cp949')
w['일시'] = pd.to_datetime(w['일시'])
w = w.sort_values('일시').reset_index(drop=True)
w['doy'] = w['일시'].dt.dayofyear

# 7일 이동평균을 캘린더-doy 평균에 적용 → 노이즈 줄인 평년 근사
clim = w.groupby('doy')[['평균기온(°C)', '최고기온(°C)', '최저기온(°C)',
                         '일강수량(mm)', '평균 풍속(m/s)']].mean()
# doy를 인덱스로 양 끝을 wrap해서 7d 이동평균
clim_smooth = pd.concat([clim.iloc[-7:], clim, clim.iloc[:7]]).rolling(7, center=True, min_periods=1).mean().iloc[7:-7]
clim_smooth.columns = ['tavg_clim', 'tmax_clim', 'tmin_clim', 'rain_clim', 'wind_clim']
w = w.merge(clim_smooth.reset_index(), on='doy', how='left')

# 평년편차 + 보조 derived feature
w['Δtavg'] = w['평균기온(°C)'] - w['tavg_clim']
w['Δtmax'] = w['최고기온(°C)'] - w['tmax_clim']
w['Δtmin'] = w['최저기온(°C)'] - w['tmin_clim']
w['Δrain'] = w['일강수량(mm)'] - w['rain_clim']
w['Δwind'] = w['평균 풍속(m/s)'] - w['wind_clim']
w['일교차']    = w['최고기온(°C)'] - w['최저기온(°C)']
w['tavg_5dma'] = w['평균기온(°C)'].rolling(5, min_periods=1).mean()
w['doy_sin']   = np.sin(2 * np.pi * w['doy'] / 365)
w['doy_cos']   = np.cos(2 * np.pi * w['doy'] / 365)
w['year_idx']  = w['일시'].dt.year - 2019

print("=" * 80)
print(f"[1] 일별 시계열 shape: {w.shape}  ({w['일시'].min().date()} ~ {w['일시'].max().date()})")
print(f"\n[2] 평년편차 분포 (옵션 A: 7년 doy 평균 기반)")
print(w[['Δtavg', 'Δtmax', 'Δtmin', 'Δrain', 'Δwind']].describe().round(2).T.to_string())

# 연도별 평년편차 평균 — trend 확인
print(f"\n[3] 연도별 평균 평년편차 (양수 = 평년보다 더움/많음)")
yearly = w.groupby(w['일시'].dt.year)[['Δtavg', 'Δtmax', 'Δtmin', 'Δrain']].mean().round(3)
print(yearly.to_string())

w.to_parquet(os.path.join(OUT_DIR, 'weather_daily.parquet'))

# ─────────────────────────────────────────────────────────────────
# 2. RIA 사고 데이터 + 시간/요일 + 14일 윈도우 인덱스
# ─────────────────────────────────────────────────────────────────
ria = pd.read_csv(R_PATH, encoding='utf-8-sig')
ria['사고일시'] = pd.to_datetime(ria['사고일시'])
ria['date'] = ria['사고일시'].dt.normalize()
ria['시간'] = ria['사고일시'].dt.hour
ria['요일'] = ria['사고일시'].dt.dayofweek

# 윈도우 시작 가능 일자 (학습 시작 + T-1일 이후만 사용)
min_window_date = w['일시'].min() + pd.Timedelta(days=T_WINDOW - 1)
ria_valid = ria[ria['date'] >= min_window_date].copy()
print(f"\n[4] 사고 데이터 shape: {ria.shape}, T={T_WINDOW}d 윈도우 가능: {ria_valid.shape}")
print(f"  중대재해율: {ria_valid['중대재해'].mean():.4f}")

# 14일 윈도우 샘플 출력 (sanity check)
sample = ria_valid.iloc[100]
sd = sample['date']
window = w[(w['일시'] >= sd - pd.Timedelta(days=T_WINDOW-1)) & (w['일시'] <= sd)]
print(f"\n[5] 14d 윈도우 샘플 (사고={sd.date()}, 시도={sample['시도구분']}, 공종={sample['공종(소분류)']})")
print(window[['일시', '평균기온(°C)', 'Δtavg', '일강수량(mm)', '일교차']].round(2).to_string(index=False))

# ─────────────────────────────────────────────────────────────────
# 3. Negative sampling — 무사고 일자 × 시도/공종 prior
# ─────────────────────────────────────────────────────────────────
sido_p     = ria_valid['시도구분'].value_counts(normalize=True)
gongjong_p = ria_valid['공종(소분류)'].value_counts(normalize=True)
gongjong2gongdae = (ria_valid.groupby('공종(소분류)')['공사대분류']
                             .agg(lambda x: x.value_counts().index[0]).to_dict())

accident_dates = set(ria_valid['date'].dt.date.unique())
valid_dates = pd.date_range(min_window_date, w['일시'].max())
no_accident_dates = np.array([d for d in valid_dates if d.date() not in accident_dates])
print(f"\n[6] Negative pool")
print(f"  학습기간 일자: {len(valid_dates)}, 사고발생: {len(accident_dates)}, 무사고: {len(no_accident_dates)}")

N_neg = len(ria_valid) * NEG_RATIO
# 시간/공간 negative 5:5 혼합
N_time = N_neg // 2   # 같은 시도×공종, 다른 (무사고) 날짜
N_space = N_neg - N_time  # 같은 (사고)날짜, 다른 시도×공종
# 단 우리 데이터는 시도-기상 무관계라 사실상 둘다 doy 매칭 효과만 다름
# → 둘 다 무사고 날짜에서 뽑되 prior만 다르게

neg_dates = RNG.choice(no_accident_dates, size=N_neg)
neg_sido = RNG.choice(sido_p.index.values, size=N_neg, p=sido_p.values)
neg_gongjong = RNG.choice(gongjong_p.index.values, size=N_neg, p=gongjong_p.values)

neg = pd.DataFrame({
    'date': pd.to_datetime(neg_dates).normalize(),
    '시도구분': neg_sido,
    '공종(소분류)': neg_gongjong,
})
neg['공사대분류'] = neg['공종(소분류)'].map(gongjong2gongdae)
# 시간/요일/공정율은 학습 마진얼 분포에서 샘플링 (혹은 결측 유지)
neg['시간'] = RNG.choice(ria_valid['시간'].values, size=N_neg)
neg['요일'] = neg['date'].dt.dayofweek
neg['소규모현장']  = np.nan
neg['고위험공종']  = 0
neg['공정율_수치']  = np.nan
neg['label']      = 0
neg['source']     = 'neg'
neg['중대재해']    = 0

# ─────────────────────────────────────────────────────────────────
# 4. 라벨 통합 (메인 = 중대재해 binary)
# ─────────────────────────────────────────────────────────────────
pos_cols = ['date', '시도구분', '공사대분류', '공종(소분류)',
            '시간', '요일', '소규모현장', '고위험공종', '공정율_수치', '중대재해']
pos = ria_valid[pos_cols].copy()
pos['label'] = pos['중대재해']
pos['source'] = 'pos'

dataset = pd.concat([pos, neg], ignore_index=True)
print(f"\n[7] 통합 데이터셋")
print(f"  positive (사고): {len(pos)},  label=1: {pos['label'].sum()}  ({pos['label'].mean()*100:.2f}%)")
print(f"  negative (무사고): {len(neg)},  label=1: {neg['label'].sum()}  (0.0%)")
print(f"  전체 양성률: {dataset['label'].mean()*100:.4f}%  → 클래스 가중치 ~{(1-dataset['label'].mean())/dataset['label'].mean():.1f}")

# ─────────────────────────────────────────────────────────────────
# 5. 시계열 split (2019-07~2024-12 학습 / 2025 검증)
# ─────────────────────────────────────────────────────────────────
SPLIT_DATE = pd.Timestamp('2025-01-01')
dataset['split'] = np.where(dataset['date'] < SPLIT_DATE, 'train', 'val')
print(f"\n[8] 시계열 split")
print(dataset.groupby(['split', 'source'])['label'].agg(['size', 'sum', 'mean']).round(4))

dataset.to_parquet(os.path.join(OUT_DIR, 'dataset.parquet'))
print(f"\n[저장] {OUT_DIR}/weather_daily.parquet,  dataset.parquet")
