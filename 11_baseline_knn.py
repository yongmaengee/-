"""Phase 1 — doy-filtered KNN retrieval 베이스라인 (모델 학습 X)
- 사고 케이스 40K만 사용 (negative 폐기)
- 14일 윈도우 요약 + 현장 컨텍스트 → KNN
- doy ±15일 안에서만 후보 검색
- val 케이스의 라벨 = top-K train 이웃의 라벨 평균
- 모델이 이 점수보다 잘 나와야 가치 있음
"""
import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.spatial.distance import cdist

OUT = '/home/nuri5/바탕화면/공모전/cache'
T_WIN = 14
DOY_BUCKET = 15
K_LIST = [5, 20, 50]
SPLIT = pd.Timestamp('2025-01-01')

# ───────────────────────────────────────────────────────
# 1. 일별 시계열 로드 (캐시) + 사고 데이터
# ───────────────────────────────────────────────────────
w = pd.read_parquet(f'{OUT}/weather_daily.parquet')
w_idx = w.set_index('일시').sort_index()

ria = pd.read_csv('/home/nuri5/바탕화면/공모전/RIA_최종통합피쳐셋.csv', encoding='utf-8-sig')
ria['사고일시'] = pd.to_datetime(ria['사고일시'])
ria['date'] = ria['사고일시'].dt.normalize()
ria['시간'] = ria['사고일시'].dt.hour
ria['요일'] = ria['사고일시'].dt.dayofweek
ria['시간bin'] = pd.cut(ria['시간'], bins=[-1, 5, 11, 17, 23],
                         labels=['새벽', '오전', '오후', '저녁']).astype(str)

# ───────────────────────────────────────────────────────
# 2. 사고일자별 14일 윈도우 요약 통계
# ───────────────────────────────────────────────────────
W_FEATS = ['평균기온(°C)', '최고기온(°C)', '최저기온(°C)', '일강수량(mm)',
           '평균 풍속(m/s)', 'Δtavg', 'Δtmax', 'Δtmin', 'Δrain', '일교차']

def summarize_window(end_date):
    start = end_date - pd.Timedelta(days=T_WIN - 1)
    win = w_idx.loc[start:end_date, W_FEATS]
    if len(win) < T_WIN:
        return None
    s = {}
    for c in W_FEATS:
        s[f'{c}_mean'] = win[c].mean()
        s[f'{c}_max'] = win[c].max()
        s[f'{c}_min'] = win[c].min()
    s['rain_sum'] = win['일강수량(mm)'].sum()
    s['hot_days']  = int((win['최고기온(°C)'] >= 33).sum())
    s['cold_days'] = int((win['최저기온(°C)'] <= -5).sum())
    s['rain_days'] = int((win['일강수량(mm)'] >= 10).sum())
    s['windy_days'] = int((win['평균 풍속(m/s)'] >= 4).sum())
    s['Δtavg_hot_days'] = int((win['Δtavg'] > 1).sum())
    s['Δtavg_cold_days'] = int((win['Δtavg'] < -1).sum())
    s['Δtavg_trend'] = win['Δtavg'].iloc[-7:].mean() - win['Δtavg'].iloc[:7].mean()
    return s

t0 = time.time()
unique_dates = ria['date'].unique()
date_summary = {}
for d in unique_dates:
    s = summarize_window(d)
    if s is not None:
        date_summary[d] = s
print(f"[1] {len(date_summary)}개 일자 윈도우 요약 ({time.time()-t0:.1f}s)")

ws_df = pd.DataFrame.from_dict(date_summary, orient='index')
ws_df.index.name = 'date'
ws_df = ws_df.reset_index()
df = ria.merge(ws_df, on='date', how='inner')
print(f"[2] 윈도우 매칭 후 사고 케이스: {df.shape}")

# ───────────────────────────────────────────────────────
# 3. 학습/검증 split
# ───────────────────────────────────────────────────────
df = df.sort_values('사고일시').reset_index(drop=True)
df['split'] = np.where(df['date'] < SPLIT, 'train', 'val')

num_cols = [c for c in df.columns if any(s in c for s in ['_mean', '_max', '_min', '_days', 'rain_sum', 'Δtavg_trend'])]
num_cols += ['공정율_수치']
cat_cols = ['공종(소분류)', '시도구분', '공사대분류', '시간bin', '요일', '소규모현장', '고위험공종']

# 결측 처리
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].astype(str).fillna('missing')

train = df[df['split'] == 'train'].reset_index(drop=True)
val = df[df['split'] == 'val'].reset_index(drop=True)
print(f"\n[3] Split — train {len(train)} (중대재해율 {train['중대재해'].mean():.4f})  /  val {len(val)} (중대재해율 {val['중대재해'].mean():.4f})")

# ───────────────────────────────────────────────────────
# 4. Feature matrix (numeric z-score + one-hot 카테고리, 카테고리 가중↓)
# ───────────────────────────────────────────────────────
scaler = StandardScaler().fit(train[num_cols])
X_num_tr = scaler.transform(train[num_cols])
X_num_va = scaler.transform(val[num_cols])

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(train[cat_cols])
X_cat_tr = ohe.transform(train[cat_cols]) * 0.5  # 카테고리 가중 절반 (numeric과 균형)
X_cat_va = ohe.transform(val[cat_cols]) * 0.5

X_tr = np.hstack([X_num_tr, X_cat_tr]).astype(np.float32)
X_va = np.hstack([X_num_va, X_cat_va]).astype(np.float32)
y_tr = train['중대재해'].values
y_va = val['중대재해'].values
print(f"[4] feature dim={X_tr.shape[1]}  (numeric {len(num_cols)} + cat onehot {X_cat_tr.shape[1]})")

# doy
tr_doy = train['date'].dt.dayofyear.values
va_doy = val['date'].dt.dayofyear.values

# ───────────────────────────────────────────────────────
# 5. doy ±15일 마스크 + KNN
# ───────────────────────────────────────────────────────
print(f"\n[5] KNN, doy bucket ±{DOY_BUCKET}일")
all_preds = {K: np.zeros(len(val), dtype=np.float32) for K in K_LIST}

batch = 400
t0 = time.time()
for i in range(0, len(val), batch):
    j = min(i + batch, len(val))
    D = cdist(X_va[i:j], X_tr, metric='euclidean')
    doy_diff = np.abs(va_doy[i:j, None] - tr_doy[None, :])
    doy_diff = np.minimum(doy_diff, 365 - doy_diff)
    D[doy_diff > DOY_BUCKET] = np.inf
    for K in K_LIST:
        topk = np.argpartition(D, min(K, D.shape[1]-1), axis=1)[:, :K]
        nn_lab = y_tr[topk]  # (B, K)
        all_preds[K][i:j] = nn_lab.mean(axis=1)
print(f"  완료 ({time.time()-t0:.1f}s)")

# ───────────────────────────────────────────────────────
# 6. 평가
# ───────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("[6] 평가 — val 2025년 사고 케이스 6,196건")
base = y_va.mean()
print(f"  베이스라인 (always predict {base:.4f}): PR-AUC = {base:.4f}")
print()
print(f"{'K':>4} | {'PR-AUC':>7} | {'ROC-AUC':>8} | {'R@5%':>7} {'L@5%':>5} | {'R@10%':>7} {'L@10%':>5} | {'R@20%':>7} {'L@20%':>5}")
print("-" * 90)

for K in K_LIST:
    p = all_preds[K]
    pr = average_precision_score(y_va, p)
    roc = roc_auc_score(y_va, p)
    line = f"{K:>4} | {pr:>7.4f} | {roc:>8.4f} |"
    for pct in [5, 10, 20]:
        n_alarm = int(len(y_va) * pct / 100)
        top_idx = np.argsort(p)[::-1][:n_alarm]
        recall = y_va[top_idx].sum() / y_va.sum()
        rate = y_va[top_idx].mean()
        lift = rate / base
        line += f" {recall:>6.3f} {lift:>5.2f}x|"
    print(line)

# ───────────────────────────────────────────────────────
# 7. 결과 저장
# ───────────────────────────────────────────────────────
df.to_parquet(f'{OUT}/accidents_with_features.parquet')
np.savez(f'{OUT}/baseline_knn_preds.npz',
         val_y=y_va, **{f'pred_K{K}': all_preds[K] for K in K_LIST})
print(f"\n[저장] {OUT}/accidents_with_features.parquet, baseline_knn_preds.npz")
