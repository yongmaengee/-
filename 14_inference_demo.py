"""Phase 3 (v2) — 추론 데모: 모델 4-head + Isotonic 캘리브레이션 + §4 룰 체크리스트"""
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression

ROOT = '/home/nuri5/바탕화면/공모전'
CACHE = f'{ROOT}/cache'

# ───── §4 룰 정의 ───────────────────────────────────
RULES = [
    ('강수누적-철골 추락위험',
     lambda r: r['공종(소분류)'] == '철골공사' and r['강수_연속_7d'] >= 3,
     2.67, '7일 연속 강수 + 고소작업. 비계 결로/미끄럼 점검, 안전대 이중 결속.'),
    ('폭염누적-토공 붕괴위험',
     lambda r: r['공종(소분류)'] == '토공사' and r['폭염_연속_7d'] >= 2,
     2.26, '폭염 2일+ 누적 + 토공. 사면 균열·지반 약화 점검, 작업자 휴식 강화.'),
    ('강수누적-기타 붕괴위험',
     lambda r: r['공종(소분류)'] == '기타' and r['누적강수_7d'] >= 100,
     2.00, '7일 누적 강수 100mm↑. 굴착면·법면 안정성 재점검 필수.'),
    ('강수누적-기타(2)',
     lambda r: r['공종(소분류)'] == '기타' and r['강수_연속_7d'] >= 3,
     1.96, '강수 3일 연속. 침수 자재·전선 재배치, 배수 점검.'),
    ('한파-철골 동결위험',
     lambda r: r['공종(소분류)'] == '철골공사' and r['최저기온(°C)'] <= -5,
     1.87, '저온 노출. 강재 취성 파괴 위험, 용접부 점검.'),
    ('강수누적-토공 붕괴위험',
     lambda r: r['공종(소분류)'] == '토공사' and r['강수_연속_7d'] >= 3,
     1.83, '연속 강수로 지반 포화. 사면 변위 모니터링.'),
    ('강수누적-기계설비',
     lambda r: r['공종(소분류)'] == '기계설비공사' and r['누적강수_7d'] >= 100,
     1.83, '누적 강수 + 기계설비. 누전·감전 방호 확인.'),
    ('한파-토공',
     lambda r: r['공종(소분류)'] == '토공사' and r['최저기온(°C)'] <= -5,
     1.70, '한파 + 토공. 동결-융해 반복, 사면 변형 점검.'),
    ('강수누적-철골(2)',
     lambda r: r['공종(소분류)'] == '철골공사' and r['누적강수_7d'] >= 100,
     1.68, '7일 누적 100mm↑ + 철골. 비계 침하·결로 점검.'),
    ('기온변동-해체철거',
     lambda r: r['공종(소분류)'] == '해체 및 철거공사' and r['기온변동성_7d'] >= 4,
     1.54, '주간 기온변동 큼 + 해체철거. 구조물 응력 변화 위험.'),
    ('강풍-철근콘크리트',
     lambda r: r['공종(소분류)'] == '철근콘크리트공사' and r['평균 풍속(m/s)'] >= 4,
     1.38, '강풍 + 콘크리트. 거푸집·고소 자재 비산, 양생 영향.'),
]

# ───── 모델 정의 (학습과 동일) ────────────────────────
ckpt = torch.load(f'{CACHE}/model/ria_model.pt', weights_only=False)
cfg, vocab = ckpt['config'], ckpt['vocab']
mean, std = np.array(ckpt['mean']), np.array(ckpt['std'])

class RiskInstinctModel(nn.Module):
    def __init__(self, vocab_sizes, num_cont=3, T=14, F=10, d=64, n_enc=2, n_dec=1, n_heads=4):
        super().__init__()
        self.weather_proj = nn.Linear(F, d)
        self.pos_emb = nn.Parameter(torch.zeros(T, d))
        enc_layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4*d, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc)
        self.cat_embs = nn.ModuleList([nn.Embedding(s + 2, d // 4) for s in vocab_sizes])
        ctx_dim = (d // 4) * len(vocab_sizes) + num_cont
        self.ctx_proj = nn.Linear(ctx_dim, d)
        dec_layer = nn.TransformerDecoderLayer(d, n_heads, dim_feedforward=4*d, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec)
        self.heads = nn.ModuleDict({k: nn.Linear(d, 1) for k in ['중대재해','다중사상','외국인피해','고령자피해']})
    def forward(self, win, cat_ids, num):
        x = self.weather_proj(win) + self.pos_emb.unsqueeze(0)
        kv = self.encoder(x)
        embs = [emb(cat_ids[:, i]) for i, emb in enumerate(self.cat_embs)]
        ctx = torch.cat(embs + [num], dim=-1)
        q = self.ctx_proj(ctx).unsqueeze(1)
        out = self.decoder(q, kv).squeeze(1)
        return {k: head(out).squeeze(-1) for k, head in self.heads.items()}, out

vocab_sizes = [len(vocab[c]) for c in cfg['cat_cols']]
model = RiskInstinctModel(vocab_sizes, T=cfg['T_WIN'], F=cfg['F_DIM'], d=cfg['D_MODEL'],
                           n_enc=cfg['N_ENC'], n_dec=cfg['N_DEC'], n_heads=cfg['N_HEADS'])
model.load_state_dict(ckpt['model_state'])
model.eval()

# ───── 데이터 ────────────────────────────────────
w = pd.read_parquet(f'{CACHE}/weather_daily.parquet').sort_values('일시').reset_index(drop=True)
w['doy_sin'] = np.sin(2*np.pi * w['일시'].dt.dayofyear / 365)
w['doy_cos'] = np.cos(2*np.pi * w['일시'].dt.dayofyear / 365)
WF = cfg['WEATHER_FEATS']
weather_arr = w[WF + ['doy_sin', 'doy_cos']].values.astype(np.float32).copy()
weather_arr[:, :len(WF)] = (weather_arr[:, :len(WF)] - mean) / (std + 1e-6)
date2idx = {d: i for i, d in enumerate(w['일시'].dt.normalize())}

ria = pd.read_csv(f'{ROOT}/RIA_최종통합피쳐셋.csv', encoding='utf-8-sig')
ria['사고일시'] = pd.to_datetime(ria['사고일시'])
ria['date'] = ria['사고일시'].dt.normalize()
ria['시간'] = ria['사고일시'].dt.hour
ria['요일'] = ria['사고일시'].dt.dayofweek
ria['시간bin'] = pd.cut(ria['시간'], bins=[-1,5,11,17,23], labels=[0,1,2,3]).astype(int)
ria['공정율_수치'] = ria['공정율_수치'].fillna(50)
ria['소규모현장'] = ria['소규모현장'].fillna(0).astype(float)
ria['공사대분류'] = ria['공사대분류'].astype(str)
ria['시도구분'] = ria['시도구분'].astype(str)
ria['공종(소분류)'] = ria['공종(소분류)'].astype(str)

val = ria[ria['사고일시'] >= pd.Timestamp('2025-01-01')].reset_index(drop=True)
# val 절반은 calibration용, 절반은 demo
val_cal = val.iloc[::2].reset_index(drop=True)
val_demo = val.iloc[1::2].reset_index(drop=True)
print(f"[데이터] val_cal={len(val_cal)}, val_demo={len(val_demo)}")

def predict_batch(df_subset):
    HEADS = ['중대재해','다중사상','외국인피해','고령자피해']
    out = {h: [] for h in HEADS}
    with torch.no_grad():
        for i in range(0, len(df_subset), 256):
            chunk = df_subset.iloc[i:i+256]
            wins, cats, nums = [], [], []
            for _, row in chunk.iterrows():
                end = date2idx[row['date']]
                wins.append(weather_arr[end - cfg['T_WIN'] + 1: end + 1])
                cats.append([vocab[c].get(str(row[c]), 0) for c in cfg['cat_cols']])
                nums.append([row['공정율_수치'], row['소규모현장'], row['고위험공종']])
            wins = torch.from_numpy(np.stack(wins)).float()
            cats = torch.tensor(cats, dtype=torch.long)
            nums = torch.tensor(nums, dtype=torch.float32)
            logits, _ = model(wins, cats, nums)
            for h in HEADS:
                out[h].append(torch.sigmoid(logits[h]).numpy())
    return {h: np.concatenate(v) for h, v in out.items()}

# ───── 캘리브레이션 (Isotonic) ───────────────────────
print("[캘리브레이션] val 절반으로 isotonic regressor 학습 중...")
preds_cal = predict_batch(val_cal)
labels_cal = {
    '중대재해':   val_cal['중대재해'].values,
    '다중사상':   (val_cal['총재해자'] >= 2).astype(int).values,
    '외국인피해': (val_cal['외국인재해자'] >= 1).astype(int).values,
    '고령자피해': (val_cal['고령재해자'] >= 1).astype(int).values,
}
calibrators = {}
print(f"  Head별 (raw 평균 → 캘리브레이션 후 평균):")
for h in preds_cal:
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(preds_cal[h], labels_cal[h])
    calibrators[h] = ir
    raw_mean = preds_cal[h].mean()
    cal_mean = ir.predict(preds_cal[h]).mean()
    base = labels_cal[h].mean()
    print(f"   {h}: {raw_mean:.3f} → {cal_mean:.3f}  (실제 base {base:.3f})")

# ───── Demo 케이스 선정: 룰 활성 + 중대재해 케이스 포함 ───
np.random.seed(13)
def has_active_rule(row):
    return any(cond(row) for _, cond, _, _ in RULES)
val_demo['has_rule'] = val_demo.apply(has_active_rule, axis=1)

picks = []
# 1) 룰 활성 + 중대재해
sub = val_demo[(val_demo['has_rule']) & (val_demo['중대재해'] == 1)]
if len(sub):
    picks.append(sub.sample(1, random_state=13))
# 2) 룰 활성 + 비중대
sub = val_demo[(val_demo['has_rule']) & (val_demo['중대재해'] == 0)]
if len(sub):
    picks.append(sub.sample(2, random_state=13))
# 3) 룰 비활성 + 중대재해
sub = val_demo[(~val_demo['has_rule']) & (val_demo['중대재해'] == 1)]
if len(sub):
    picks.append(sub.sample(1, random_state=13))
# 4) 룰 비활성 + 비중대
sub = val_demo[(~val_demo['has_rule']) & (val_demo['중대재해'] == 0)]
if len(sub):
    picks.append(sub.sample(1, random_state=13))
samples = pd.concat(picks).reset_index(drop=True)

# 캘리브레이션된 예측
preds_demo = predict_batch(samples)
preds_cal_demo = {h: calibrators[h].predict(preds_demo[h]) for h in preds_demo}

# ───── 출력 ─────────────────────────────────────
print("\n" + "=" * 100)
print("  RISK INSTINCT ALERT — 추론 데모 (캘리브레이션 적용)")
print("=" * 100)

for i, row in samples.iterrows():
    print(f"\n[Case {i+1}] {row['사고일시']}, {row['시도구분']}")
    print(f"  공사:    {row['공사대분류']} / {row['공종(소분류)']}  (공정율 {row['공정율_수치']:.0f}%, 고위험공종={row['고위험공종']})")
    print(f"  당일:    평균 {row['평균기온(°C)']:.1f}°C / 최고 {row['최고기온(°C)']:.1f} / 최저 {row['최저기온(°C)']:.1f} / 강수 {row['일강수량(mm)']:.1f}mm / 풍속 {row['평균 풍속(m/s)']:.1f}m/s")
    print(f"  7일누적: 강수 {row['누적강수_7d']:.1f}mm  /  폭염연속 {row['폭염_연속_7d']}일  /  한파연속 {row['한파_연속_7d']}일  /  강수연속 {row['강수_연속_7d']}일")

    print(f"\n  ▶ 모델 예측 확률 (캘리브레이션 후):")
    for h in ['중대재해', '다중사상', '외국인피해', '고령자피해']:
        v = preds_cal_demo[h][i]
        base = labels_cal[h].mean()
        lift = v / max(base, 1e-9)
        bar = '█' * int(min(v, 1.0) * 30)
        flag = ' ⚠ HIGH' if lift >= 2.0 else (' ⚠' if lift >= 1.5 else '')
        print(f"      {h:<8}: {v*100:5.2f}%  (base {base*100:.1f}%, Lift {lift:.2f}x)  {bar}{flag}")

    print(f"\n  ▶ 활성 이슈 체크리스트 (§4 룰):")
    issues = [(name, lift, advice) for name, cond, lift, advice in RULES if cond(row)]
    if issues:
        for name, lift, advice in sorted(issues, key=lambda x: -x[1]):
            print(f"      ⚠ {name}  (단변량 Lift {lift:.2f}x)")
            print(f"         → {advice}")
    else:
        print(f"      (활성 이슈 없음 — 알려진 단변량 룰 미적용)")

    print(f"\n  ▶ 실제 결과: 사망 {row['사망자']}, 부상 {row['부상자']}, 외국인 {row['외국인재해자']}, 고령 {row['고령재해자']}, 위험등급 {row['위험등급']}")
    print("-" * 100)

print("\n" + "=" * 100)
print(f"  Phase 1 KNN PR-AUC 0.042 → Phase 2 모델 0.078 (+87%)  /  Top5% Lift 1.32x → 3.25x")
print(f"  학습된 룰 {len(RULES)}개 + 모델 4 head + Isotonic 캘리브레이션  /  파라미터 175K")
print("=" * 100)
