"""Phase 4 — What-if 시나리오 입력 → 실시간 알람 출력
- alert(date, 시도, 대분류, 공종, 공정율, 시간, 소규모, 고위험) 함수 호출
- 외부 일별 기상에서 14d 윈도우 + 누적 피처 자동 재계산
- 모델 4-head + 캘리브레이션 + §4 룰 체크 통합
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression

ROOT = '/home/nuri5/바탕화면/공모전'
CACHE = f'{ROOT}/cache'

# ───── 모델/룰 정의 (14와 동일) ────────────────────
RULES = [
    ('강수누적-철골 추락위험',
     lambda r: r['공종'] == '철골공사' and r['강수_연속_7d'] >= 3,
     2.67, '7일 연속 강수 + 고소작업. 비계 결로/미끄럼 점검, 안전대 이중 결속.'),
    ('폭염누적-토공 붕괴위험',
     lambda r: r['공종'] == '토공사' and r['폭염_연속_7d'] >= 2,
     2.26, '폭염 2일+ 누적 + 토공. 사면 균열·지반 약화 점검.'),
    ('강수누적-기타 붕괴위험',
     lambda r: r['공종'] == '기타' and r['누적강수_7d'] >= 100,
     2.00, '7일 누적 강수 100mm↑. 굴착면·법면 안정성 점검.'),
    ('강수누적-기타(2)',
     lambda r: r['공종'] == '기타' and r['강수_연속_7d'] >= 3,
     1.96, '강수 3일 연속. 침수 자재·전선 재배치, 배수 점검.'),
    ('한파-철골 동결위험',
     lambda r: r['공종'] == '철골공사' and r['최저기온'] <= -5,
     1.87, '저온 노출. 강재 취성 파괴 위험, 용접부 점검.'),
    ('강수누적-토공 붕괴위험',
     lambda r: r['공종'] == '토공사' and r['강수_연속_7d'] >= 3,
     1.83, '연속 강수로 지반 포화. 사면 변위 모니터링.'),
    ('강수누적-기계설비',
     lambda r: r['공종'] == '기계설비공사' and r['누적강수_7d'] >= 100,
     1.83, '누적 강수 + 기계설비. 누전·감전 방호 확인.'),
    ('한파-토공',
     lambda r: r['공종'] == '토공사' and r['최저기온'] <= -5,
     1.70, '한파 + 토공. 동결-융해 반복 주의.'),
    ('강수누적-철골(2)',
     lambda r: r['공종'] == '철골공사' and r['누적강수_7d'] >= 100,
     1.68, '7일 누적 100mm↑ + 철골. 비계 침하·결로 점검.'),
    ('기온변동-해체철거',
     lambda r: r['공종'] == '해체 및 철거공사' and r['기온변동성_7d'] >= 4,
     1.54, '주간 기온변동 큼 + 해체철거. 구조물 응력 변화 위험.'),
    ('강풍-철근콘크리트',
     lambda r: r['공종'] == '철근콘크리트공사' and r['풍속'] >= 4,
     1.38, '강풍 + 콘크리트. 거푸집·자재 비산, 양생 영향.'),
]

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

# ───── 외부 일별 기상 (정규화 전) + 정규화 버전 ─────
w = pd.read_parquet(f'{CACHE}/weather_daily.parquet').sort_values('일시').reset_index(drop=True)
w['doy_sin'] = np.sin(2*np.pi * w['일시'].dt.dayofyear / 365)
w['doy_cos'] = np.cos(2*np.pi * w['일시'].dt.dayofyear / 365)
WF = cfg['WEATHER_FEATS']
weather_norm = w[WF + ['doy_sin', 'doy_cos']].values.astype(np.float32).copy()
weather_norm[:, :len(WF)] = (weather_norm[:, :len(WF)] - mean) / (std + 1e-6)
date2idx = {d: i for i, d in enumerate(w['일시'].dt.normalize())}

# ───── 캘리브레이션 (val 절반으로 fit) ─────────────
ria = pd.read_csv(f'{ROOT}/RIA_최종통합피쳐셋.csv', encoding='utf-8-sig')
ria['사고일시'] = pd.to_datetime(ria['사고일시'])
ria['date'] = ria['사고일시'].dt.normalize()
ria['시간'] = ria['사고일시'].dt.hour
ria['요일'] = ria['사고일시'].dt.dayofweek
ria['시간bin'] = pd.cut(ria['시간'], bins=[-1,5,11,17,23], labels=[0,1,2,3]).astype(int)
ria['공정율_수치'] = ria['공정율_수치'].fillna(50)
ria['소규모현장'] = ria['소규모현장'].fillna(0).astype(float)
val_cal = ria[ria['사고일시'] >= pd.Timestamp('2025-01-01')].iloc[::2].reset_index(drop=True)

def predict_rows(rows):
    HEADS = ['중대재해','다중사상','외국인피해','고령자피해']
    out = {h: [] for h in HEADS}
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            chunk = rows.iloc[i:i+256]
            wins, cats, nums = [], [], []
            for _, r in chunk.iterrows():
                end = date2idx[r['date']]
                wins.append(weather_norm[end - cfg['T_WIN'] + 1: end + 1])
                cats.append([vocab[c].get(str(r[c]), 0) for c in cfg['cat_cols']])
                nums.append([r['공정율_수치'], r['소규모현장'], r['고위험공종']])
            wins = torch.from_numpy(np.stack(wins)).float()
            cats = torch.tensor(cats, dtype=torch.long)
            nums = torch.tensor(nums, dtype=torch.float32)
            logits, _ = model(wins, cats, nums)
            for h in HEADS:
                out[h].append(torch.sigmoid(logits[h]).numpy())
    return {h: np.concatenate(v) for h, v in out.items()}

cal_preds = predict_rows(val_cal)
cal_labels = {
    '중대재해':   val_cal['중대재해'].values,
    '다중사상':   (val_cal['총재해자'] >= 2).astype(int).values,
    '외국인피해': (val_cal['외국인재해자'] >= 1).astype(int).values,
    '고령자피해': (val_cal['고령재해자'] >= 1).astype(int).values,
}
calibrators, base_rates = {}, {}
for h in cal_preds:
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(cal_preds[h], cal_labels[h])
    calibrators[h] = ir
    base_rates[h] = cal_labels[h].mean()

# ───── 외부 날씨에서 누적 피처 재계산 ────────────────
def compute_cumulative(end_date_str):
    end = pd.Timestamp(end_date_str)
    win7 = w[(w['일시'] >= end - pd.Timedelta(days=6)) & (w['일시'] <= end)]
    today = w[w['일시'] == end].iloc[0]
    def streak(series, cond):
        s = cond(series)
        cnt = 0
        for v in s.values[::-1]:
            if v: cnt += 1
            else: break
        return cnt
    return {
        '평균기온': today['평균기온(°C)'],
        '최고기온': today['최고기온(°C)'],
        '최저기온': today['최저기온(°C)'],
        '일강수량': today['일강수량(mm)'],
        '풍속': today['평균 풍속(m/s)'],
        'Δtavg': today['Δtavg'],
        '누적강수_7d':   win7['일강수량(mm)'].sum(),
        '폭염_연속_7d': streak(win7['최고기온(°C)'], lambda s: s >= 33),
        '한파_연속_7d': streak(win7['최저기온(°C)'], lambda s: s <= -10),
        '강수_연속_7d': streak(win7['일강수량(mm)'], lambda s: s >= 1),
        '기온변동성_7d': win7['평균기온(°C)'].std(),
    }

# ───── alert() — 단일 시나리오 추론 + 출력 ─────────
def alert(date, 시도, 대분류, 공종, 공정율, 시간, 소규모=0, 고위험=0):
    end = pd.Timestamp(date)
    feats = compute_cumulative(date)
    rule_row = {**feats, '공종': 공종}

    row = pd.Series({
        'date': end, '시도구분': 시도, '공사대분류': 대분류, '공종(소분류)': 공종,
        '시간': 시간, '요일': end.dayofweek,
        '시간bin': int(pd.cut([시간], bins=[-1,5,11,17,23], labels=[0,1,2,3])[0]),
        '공정율_수치': 공정율, '소규모현장': float(소규모), '고위험공종': float(고위험),
    })
    pred_raw = predict_rows(row.to_frame().T)
    pred = {h: float(calibrators[h].predict([pred_raw[h][0]])[0]) for h in pred_raw}

    issues = sorted(
        [(name, lift, advice) for name, cond, lift, advice in RULES if cond(rule_row)],
        key=lambda x: -x[1]
    )

    print("┌" + "─" * 96 + "┐")
    print(f"│ {date} {end.strftime('%a')}, {시도}  |  {대분류} / {공종}  (공정율 {공정율}%, 시간 {시간}시)")
    print("├" + "─" * 96 + "┤")
    print(f"│ 당일 기상:  평균 {feats['평균기온']:5.1f}°C / 최고 {feats['최고기온']:5.1f} / 최저 {feats['최저기온']:5.1f}  / Δtavg {feats['Δtavg']:+.1f}°C")
    print(f"│           강수 {feats['일강수량']:5.1f}mm / 풍속 {feats['풍속']:.1f}m/s")
    print(f"│ 7일 누적:  강수 {feats['누적강수_7d']:.0f}mm / 폭염연속 {feats['폭염_연속_7d']}일 / 한파연속 {feats['한파_연속_7d']}일 / 강수연속 {feats['강수_연속_7d']}일 / 변동성 {feats['기온변동성_7d']:.2f}°C")
    print("├" + "─" * 96 + "┤")
    print(f"│ ▶ 모델 예측 (확률, 캘리브레이션 후):")
    for h in ['중대재해','다중사상','외국인피해','고령자피해']:
        v = pred[h]
        b = base_rates[h]
        lift = v / max(b, 1e-9)
        bar = '█' * int(min(v, 1.0) * 30)
        flag = ' ⚠ HIGH' if lift >= 2.0 else (' ⚠' if lift >= 1.5 else '')
        print(f"│   {h:<8}: {v*100:5.2f}%   (base {b*100:4.1f}%, Lift {lift:.2f}x)  {bar}{flag}")

    print(f"│")
    print(f"│ ▶ 활성 이슈 (§4 단변량 룰):")
    if issues:
        for name, lift, advice in issues:
            print(f"│   ⚠ {name}  (Lift {lift:.2f}x)")
            print(f"│     → {advice}")
    else:
        print(f"│   (활성 룰 없음 — 누적 임계 미달)")
    print("└" + "─" * 96 + "┘")

# ────────────────────────────────────────────────
# 시나리오 6개
# ────────────────────────────────────────────────
print("\n" + "=" * 100)
print("RISK INSTINCT ALERT — What-if 시나리오 데모")
print("=" * 100)

# ① 한여름 폭염 + 토공
alert('2024-08-05', '경기도', '토목', '토공사', 공정율=50, 시간=13)

# ② 한파 + 철골 고소작업
alert('2024-12-23', '강원도', '건축', '철골공사', 공정율=60, 시간=9, 고위험=1)

# ③ 장마 + 굴착(기타공종)
alert('2024-07-15', '서울특별시', '건축', '기타', 공정율=40, 시간=10)

# ④ 강풍 + 콘크리트
alert('2024-04-10', '부산광역시', '건축', '철근콘크리트공사', 공정율=70, 시간=14)

# ⑤ 평범한 봄날 + 마감(도장)
alert('2024-04-22', '서울특별시', '건축', '도장공사', 공정율=90, 시간=14, 소규모=1)

# ⑥ 일요일 점심 직전 + 작업관리 약함
alert('2024-09-08', '경상북도', '토목', '관공사', 공정율=80, 시간=12)

print("\n" + "=" * 100)
print("alert(date, 시도, 대분류, 공종, 공정율=, 시간=, 소규모=0, 고위험=0) 형태로 시나리오 추가 가능")
print("=" * 100)
