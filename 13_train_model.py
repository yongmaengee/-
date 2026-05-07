"""Phase 2 — Multi-task Encoder-Decoder Cross-Attention 모델
- Encoder: 14d 일별 기상 시계열 (8 features) → K, V
- Decoder: 현장 컨텍스트 1 토큰 → Cross-Attention with Enc K/V
- 4 Heads: 중대재해 (가중 5x), 다중사상, 외국인피해, 고령자피해
- 학습/검증: 시계열 split (2019-07~2024 / 2025)
"""
import os, json, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = '/home/nuri5/바탕화면/공모전'
CACHE = f'{ROOT}/cache'

# ───── 설정 ─────────────────────────────────────────
T_WIN     = 14
D_MODEL   = 64
N_HEADS   = 4
N_ENC     = 2
N_DEC     = 1
EPOCHS    = 8
BATCH     = 512
LR        = 1e-3
SPLIT     = pd.Timestamp('2025-01-01')
HEAD_WEIGHTS = {'중대재해': 5.0, '다중사상': 1.0, '외국인피해': 1.0, '고령자피해': 1.0}
torch.manual_seed(42)
np.random.seed(42)

# ───── 데이터 로드 ───────────────────────────────────
w = pd.read_parquet(f'{CACHE}/weather_daily.parquet').sort_values('일시').reset_index(drop=True)
w_idx = w.set_index('일시')

ria = pd.read_csv(f'{ROOT}/RIA_최종통합피쳐셋.csv', encoding='utf-8-sig')
ria['사고일시'] = pd.to_datetime(ria['사고일시'])
ria['date'] = ria['사고일시'].dt.normalize()
ria['시간'] = ria['사고일시'].dt.hour
ria['요일'] = ria['사고일시'].dt.dayofweek
ria['시간bin'] = pd.cut(ria['시간'], bins=[-1, 5, 11, 17, 23], labels=[0, 1, 2, 3]).astype(int)

# 라벨 4종
ria['lab_severe']  = ria['중대재해']
ria['lab_multi']   = (ria['총재해자'] >= 2).astype(int)
ria['lab_foreign'] = (ria['외국인재해자'] >= 1).astype(int)
ria['lab_elderly'] = (ria['고령재해자'] >= 1).astype(int)

# ───── 카테고리 인코딩 ──────────────────────────────
cat_cols = ['공종(소분류)', '시도구분', '공사대분류', '시간bin', '요일']
for c in cat_cols:
    ria[c] = ria[c].astype(str).fillna('missing')

# train만으로 vocab 만들기
train_mask = ria['사고일시'] < SPLIT
vocab = {}
for c in cat_cols:
    uniq = sorted(ria.loc[train_mask, c].unique().tolist())
    vocab[c] = {v: i+1 for i, v in enumerate(uniq)}  # 0 = unknown
    vocab[c]['<unk>'] = 0
for c in cat_cols:
    ria[f'{c}_id'] = ria[c].map(lambda v: vocab[c].get(v, 0))

# 수치 피처
ria['공정율_수치'] = ria['공정율_수치'].fillna(ria.loc[train_mask, '공정율_수치'].median())
ria['소규모현장']  = ria['소규모현장'].fillna(0).astype(float)
ria['고위험공종']  = ria['고위험공종'].astype(float)

# 14d 윈도우 인덱스 사전 생성
WEATHER_FEATS = ['평균기온(°C)', '최고기온(°C)', '최저기온(°C)', '일강수량(mm)',
                 '평균 풍속(m/s)', 'Δtavg', 'Δrain', '일교차']
F_DIM = len(WEATHER_FEATS) + 2  # + doy_sin, doy_cos

# 일별 numpy 배열 생성 (lookup용)
w['doy_sin'] = np.sin(2 * np.pi * w['일시'].dt.dayofyear / 365)
w['doy_cos'] = np.cos(2 * np.pi * w['일시'].dt.dayofyear / 365)
weather_arr = w[WEATHER_FEATS + ['doy_sin', 'doy_cos']].values.astype(np.float32)
date2idx = {d: i for i, d in enumerate(w['일시'].dt.normalize())}

# 정규화 (train 일자만으로 std/mean)
train_dates = ria.loc[train_mask, 'date'].unique()
train_idx_arr = np.array([date2idx[d] for d in train_dates if d in date2idx])
mean = weather_arr[train_idx_arr, :len(WEATHER_FEATS)].mean(axis=0)
std  = weather_arr[train_idx_arr, :len(WEATHER_FEATS)].std(axis=0) + 1e-6
weather_arr_norm = weather_arr.copy()
weather_arr_norm[:, :len(WEATHER_FEATS)] = (weather_arr_norm[:, :len(WEATHER_FEATS)] - mean) / std

# 14d 윈도우 추출 함수
def get_window(date):
    end = date2idx[date]
    start = end - T_WIN + 1
    if start < 0:
        return None
    return weather_arr_norm[start:end+1]

# 사고 케이스에 대해 윈도우 가능 여부 필터
ria['has_window'] = ria['date'].apply(lambda d: d in date2idx and date2idx[d] >= T_WIN - 1)
ria = ria[ria['has_window']].reset_index(drop=True)
print(f"[데이터] 윈도우 가능 케이스: {len(ria)}")

# ───── Dataset ─────────────────────────────────────
class RIADataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.dates = self.df['date'].values
        self.cat_ids = self.df[[f'{c}_id' for c in cat_cols]].values.astype(np.int64)
        self.num = self.df[['공정율_수치', '소규모현장', '고위험공종']].values.astype(np.float32)
        self.labels = self.df[['lab_severe', 'lab_multi', 'lab_foreign', 'lab_elderly']].values.astype(np.float32)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        d = pd.Timestamp(self.dates[i])
        win = get_window(d)
        return (
            torch.from_numpy(win).float(),
            torch.from_numpy(self.cat_ids[i]),
            torch.from_numpy(self.num[i]),
            torch.from_numpy(self.labels[i]),
        )

train_df = ria[ria['사고일시'] < SPLIT].reset_index(drop=True)
val_df   = ria[ria['사고일시'] >= SPLIT].reset_index(drop=True)
print(f"[split] train={len(train_df)}  val={len(val_df)}")

train_ds = RIADataset(train_df)
val_ds   = RIADataset(val_df)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

# ───── 모델 ─────────────────────────────────────────
class RiskInstinctModel(nn.Module):
    def __init__(self, vocab_sizes, num_cont=3, T=14, F=10, d=64, n_enc=2, n_dec=1, n_heads=4):
        super().__init__()
        self.weather_proj = nn.Linear(F, d)
        self.pos_emb = nn.Parameter(torch.zeros(T, d))
        nn.init.normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4*d, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc)

        # 카테고리 임베딩 (각 vocab size + 1 for unknown)
        self.cat_embs = nn.ModuleList([nn.Embedding(s + 2, d // 4) for s in vocab_sizes])
        ctx_dim = (d // 4) * len(vocab_sizes) + num_cont
        self.ctx_proj = nn.Linear(ctx_dim, d)

        # Cross-Attention Decoder (1 query token)
        dec_layer = nn.TransformerDecoderLayer(d, n_heads, dim_feedforward=4*d, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec)

        # 4 heads
        self.heads = nn.ModuleDict({
            '중대재해':   nn.Linear(d, 1),
            '다중사상':   nn.Linear(d, 1),
            '외국인피해': nn.Linear(d, 1),
            '고령자피해': nn.Linear(d, 1),
        })

    def forward(self, win, cat_ids, num):
        # win: (B, T, F), cat_ids: (B, n_cat), num: (B, n_num)
        x = self.weather_proj(win) + self.pos_emb.unsqueeze(0)
        kv = self.encoder(x)  # (B, T, d)

        embs = [emb(cat_ids[:, i]) for i, emb in enumerate(self.cat_embs)]
        ctx = torch.cat(embs + [num], dim=-1)  # (B, ctx_dim)
        q = self.ctx_proj(ctx).unsqueeze(1)    # (B, 1, d)

        out = self.decoder(q, kv)              # (B, 1, d)
        z = out.squeeze(1)                     # (B, d)

        return {k: head(z).squeeze(-1) for k, head in self.heads.items()}, z

vocab_sizes = [len(vocab[c]) for c in cat_cols]
model = RiskInstinctModel(vocab_sizes, T=T_WIN, F=F_DIM, d=D_MODEL,
                           n_enc=N_ENC, n_dec=N_DEC, n_heads=N_HEADS)
n_params = sum(p.numel() for p in model.parameters())
print(f"[모델] params = {n_params:,}")

# ───── 학습 ─────────────────────────────────────────
# 클래스 불균형 처리: pos_weight for BCE
pos_weights = {}
for k, lab in [('중대재해', 'lab_severe'), ('다중사상', 'lab_multi'),
               ('외국인피해', 'lab_foreign'), ('고령자피해', 'lab_elderly')]:
    pos = train_df[lab].sum()
    neg = len(train_df) - pos
    pos_weights[k] = torch.tensor(neg / max(pos, 1), dtype=torch.float32)
print("[클래스 가중] (neg/pos)")
for k, v in pos_weights.items():
    print(f"  {k}: {v.item():.2f}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
HEADS = list(HEAD_WEIGHTS.keys())

def step(model, dl, train=True):
    if train: model.train()
    else: model.eval()
    losses = []
    all_preds = {h: [] for h in HEADS}
    all_labels = {h: [] for h in HEADS}
    with torch.set_grad_enabled(train):
        for win, cat_ids, num, lab in dl:
            logits, _ = model(win, cat_ids, num)
            loss = 0.0
            for i, h in enumerate(HEADS):
                bce = F.binary_cross_entropy_with_logits(
                    logits[h], lab[:, i], pos_weight=pos_weights[h])
                loss = loss + HEAD_WEIGHTS[h] * bce
                all_preds[h].append(torch.sigmoid(logits[h]).detach().numpy())
                all_labels[h].append(lab[:, i].numpy())
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(loss.item())
    metrics = {}
    for h in HEADS:
        p = np.concatenate(all_preds[h])
        y = np.concatenate(all_labels[h])
        if y.sum() > 0 and y.sum() < len(y):
            metrics[h] = {
                'pr_auc': average_precision_score(y, p),
                'roc_auc': roc_auc_score(y, p),
                'base': y.mean(),
            }
    return np.mean(losses), metrics

print("\n[학습 시작]")
print(f"{'epoch':>5} | {'tr_loss':>7} | {'val_loss':>8} | {'중대 PR':>8} {'중대 ROC':>9} | {'다중 PR':>8} | {'외국 PR':>8} | {'고령 PR':>8}")
print("-" * 100)
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, _ = step(model, train_dl, train=True)
    val_loss, val_m = step(model, val_dl, train=False)
    sched.step()
    line = f"{epoch:>5} | {tr_loss:>7.4f} | {val_loss:>8.4f} |"
    line += f" {val_m['중대재해']['pr_auc']:>7.4f} {val_m['중대재해']['roc_auc']:>8.4f} |"
    line += f" {val_m['다중사상']['pr_auc']:>7.4f} |"
    line += f" {val_m['외국인피해']['pr_auc']:>7.4f} |"
    line += f" {val_m['고령자피해']['pr_auc']:>7.4f} | ({time.time()-t0:.0f}s)"
    print(line)

# ───── 최종 평가 ─────────────────────────────────────
print("\n[최종 검증]")
val_loss, val_m = step(model, val_dl, train=False)
print(f"{'head':>10} | {'base':>5} | {'PR-AUC':>7} | {'ROC':>5} | {'R@5%':>5} {'L@5%':>5} | {'R@10%':>6} {'L@10%':>5} | {'R@20%':>6} {'L@20%':>5}")
print("-" * 100)
preds_save = {}
for h in HEADS:
    # 한 번 더 추론해서 정렬용 score 모음
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for win, cat_ids, num, lab in val_dl:
            logits, _ = model(win, cat_ids, num)
            all_p.append(torch.sigmoid(logits[h]).numpy())
            all_y.append(lab[:, HEADS.index(h)].numpy())
    p = np.concatenate(all_p)
    y = np.concatenate(all_y)
    base = y.mean()
    line = f"{h:>10} | {base:>5.3f} | {val_m[h]['pr_auc']:>7.4f} | {val_m[h]['roc_auc']:>5.3f} |"
    for pct in [5, 10, 20]:
        n_alarm = int(len(y) * pct / 100)
        top_idx = np.argsort(p)[::-1][:n_alarm]
        recall = y[top_idx].sum() / max(y.sum(), 1)
        rate = y[top_idx].mean()
        lift = rate / max(base, 1e-9)
        line += f" {recall:>5.3f} {lift:>4.2f}x|"
    print(line)
    preds_save[h] = (p, y)

# ───── 저장 ──────────────────────────────────────────
os.makedirs(f'{CACHE}/model', exist_ok=True)
torch.save({
    'model_state': model.state_dict(),
    'vocab': vocab,
    'mean': mean.tolist(),
    'std': std.tolist(),
    'config': {
        'T_WIN': T_WIN, 'D_MODEL': D_MODEL, 'N_HEADS': N_HEADS,
        'N_ENC': N_ENC, 'N_DEC': N_DEC, 'F_DIM': F_DIM,
        'cat_cols': cat_cols, 'WEATHER_FEATS': WEATHER_FEATS,
        'HEAD_WEIGHTS': HEAD_WEIGHTS,
    },
}, f'{CACHE}/model/ria_model.pt')
np.savez(f'{CACHE}/model/val_preds.npz', **{f'{h}_p': p for h, (p, _) in preds_save.items()},
                                          **{f'{h}_y': y for h, (_, y) in preds_save.items()})
print(f"\n[저장] {CACHE}/model/ria_model.pt, val_preds.npz")
