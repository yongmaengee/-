"""Phase 5 — 운영 루프: KMA fetch → 모델 추론 → Slack 알람
- mock 모드: 외부 CSV 마지막 일자를 "오늘" 으로 사용
- live 모드: KMA apihub 호출 + Slack chat.postMessage
- 토글: USE_KMA_LIVE / USE_SLACK_LIVE
"""
import os, sys, json, time
from datetime import datetime, timedelta
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression

ROOT = '/home/nuri5/바탕화면/공모전'
CACHE = f'{ROOT}/cache'

# ───── 설정 (env var fallback) ─────────────────────
KMA_KEY       = os.environ.get('KMA_KEY', '')        # data.go.kr 일반인증키
KMA_ENDPOINT  = 'https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
SLACK_TOKEN   = os.environ.get('SLACK_TOKEN', '')    # xoxb-... (chat:write 시)
SLACK_CHANNEL = os.environ.get('SLACK_CHANNEL', '#general')
SLACK_WEBHOOK = os.environ.get('SLACK_WEBHOOK', '')  # https://hooks.slack.com/services/...
USE_KMA_LIVE   = os.environ.get('USE_KMA_LIVE', '0') == '1'
USE_SLACK_LIVE = os.environ.get('USE_SLACK_LIVE', '0') == '1'

# 6개 대표 ASOS 지점 (전국 평균 근사)
KMA_STATIONS = {108: '서울', 159: '부산', 143: '대구', 112: '인천', 156: '광주', 133: '대전'}

# 모니터링 대상 현장 리스트 (시연)
SITES = [
    ('현장 A', '서울특별시', '건축', '철근콘크리트공사', 50, 14, 0, 0),
    ('현장 B', '경기도', '토목', '토공사', 30, 10, 0, 0),
    ('현장 C', '부산광역시', '건축', '철골공사', 70, 9, 0, 1),
    ('현장 D', '강원도', '토목', '해체 및 철거공사', 80, 13, 0, 1),
    ('현장 E', '경상북도', '토목', '관공사', 80, 12, 0, 0),
    ('현장 F', '인천광역시', '건축', '도장공사', 90, 14, 1, 0),
]

ALERT_LIFT_THRESHOLD = 1.5  # 알람 발동 기준

# ────────────────────────────────────────────────
# 1. KMA fetch (data.go.kr OpenAPI) + 캐시 갱신
# ────────────────────────────────────────────────
def _f(v):
    """KMA 응답 빈 문자열을 NaN으로 변환"""
    if v is None or v == '' or v == ' ':
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan

def fetch_kma_single_day(date_str):
    """단일 날짜 — 6개 대표 지점 fetch 후 평균 반환 (Series 형태)."""
    rows = []
    for stn, name in KMA_STATIONS.items():
        params = urllib.parse.urlencode({
            'serviceKey': KMA_KEY, 'numOfRows': '1', 'pageNo': '1',
            'dataType': 'JSON', 'dataCd': 'ASOS', 'dateCd': 'DAY',
            'startDt': date_str, 'endDt': date_str, 'stnIds': str(stn),
        })
        url = f'{KMA_ENDPOINT}?{params}'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode('utf-8'))
            items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
            if not items:
                continue
            it = items[0]
            rows.append({
                '평균기온(°C)':   _f(it.get('avgTa')),
                '최고기온(°C)':   _f(it.get('maxTa')),
                '최저기온(°C)':   _f(it.get('minTa')),
                '일강수량(mm)':   _f(it.get('sumRn')) if it.get('sumRn') else 0.0,
                '평균 풍속(m/s)': _f(it.get('avgWs')),
            })
        except Exception as e:
            print(f"    WARN stn={stn}({name}): {e}")
    if not rows:
        raise RuntimeError(f"KMA fetch 실패: {date_str}")
    df = pd.DataFrame(rows)
    return pd.Series({
        '일시': pd.Timestamp(date_str),
        '평균기온(°C)':   df['평균기온(°C)'].mean(),
        '최고기온(°C)':   df['최고기온(°C)'].mean(),
        '최저기온(°C)':   df['최저기온(°C)'].mean(),
        '일강수량(mm)':   df['일강수량(mm)'].mean(),
        '평균 풍속(m/s)': df['평균 풍속(m/s)'].mean(),
    })

def update_weather_cache(yesterday):
    """캐시 last_date+1 ~ yesterday 까지 KMA에서 fetch해 append + 파생컬럼 재계산."""
    w = pd.read_parquet(f'{CACHE}/weather_daily.parquet').sort_values('일시').reset_index(drop=True)
    last = w['일시'].max()
    if last >= yesterday:
        print(f"  캐시 최신 ({last.date()}) — fetch 생략")
        return w

    # 기존 캐시의 doy 평년값 (climatology)
    clim = w.groupby(w['일시'].dt.dayofyear)[['평균기온(°C)', '최고기온(°C)',
                                              '최저기온(°C)', '일강수량(mm)',
                                              '평균 풍속(m/s)']].mean()
    clim.columns = ['tavg_clim', 'tmax_clim', 'tmin_clim', 'rain_clim', 'wind_clim']

    new_rows = []
    cur = last + pd.Timedelta(days=1)
    while cur <= yesterday:
        try:
            row = fetch_kma_single_day(cur.strftime('%Y%m%d'))
            new_rows.append(row)
        except Exception as e:
            print(f"    SKIP {cur.date()}: {e}")
        cur += pd.Timedelta(days=1)
    if not new_rows:
        return w

    new = pd.DataFrame(new_rows)
    new['doy'] = new['일시'].dt.dayofyear
    new = new.merge(clim, left_on='doy', right_index=True, how='left')
    new['Δtavg'] = new['평균기온(°C)'] - new['tavg_clim']
    new['Δtmax'] = new['최고기온(°C)'] - new['tmax_clim']
    new['Δtmin'] = new['최저기온(°C)'] - new['tmin_clim']
    new['Δrain'] = new['일강수량(mm)'] - new['rain_clim']
    new['Δwind'] = new['평균 풍속(m/s)'] - new['wind_clim']
    new['일교차'] = new['최고기온(°C)'] - new['최저기온(°C)']
    new['doy_sin'] = np.sin(2 * np.pi * new['doy'] / 365)
    new['doy_cos'] = np.cos(2 * np.pi * new['doy'] / 365)
    new['year_idx'] = new['일시'].dt.year - 2019

    # 컬럼 정렬 후 concat
    new = new.reindex(columns=w.columns.tolist())
    out = pd.concat([w, new], ignore_index=True).sort_values('일시').reset_index(drop=True)
    out['tavg_5dma'] = out['평균기온(°C)'].rolling(5, min_periods=1).mean()
    out.to_parquet(f'{CACHE}/weather_daily.parquet')
    print(f"  KMA fetch +{len(new_rows)}일 → 캐시 갱신 (~{out['일시'].max().date()})")
    return out

def fetch_weather_mock(date):
    """date = pd.Timestamp. 외부 캐시에서 해당 날짜 row 반환 (없으면 가장 최근)"""
    w = pd.read_parquet(f'{CACHE}/weather_daily.parquet').sort_values('일시')
    target = pd.Timestamp(date).normalize()
    sub = w[w['일시'] <= target]
    if len(sub) == 0:
        sub = w
    return sub.iloc[-1]

def reload_weather_globals():
    """update_weather_cache 후 in-memory 시계열 재로드"""
    global w, weather_norm, date2idx
    w = pd.read_parquet(f'{CACHE}/weather_daily.parquet').sort_values('일시').reset_index(drop=True)
    w['doy_sin'] = np.sin(2 * np.pi * w['일시'].dt.dayofyear / 365)
    w['doy_cos'] = np.cos(2 * np.pi * w['일시'].dt.dayofyear / 365)
    weather_norm = w[WF + ['doy_sin', 'doy_cos']].values.astype(np.float32).copy()
    weather_norm[:, :len(WF)] = (weather_norm[:, :len(WF)] - mean) / (std + 1e-6)
    date2idx = {d: i for i, d in enumerate(w['일시'].dt.normalize())}

def get_today():
    """USE_KMA_LIVE 토글에 따라 분기"""
    today = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=1)  # 어제 (KMA 1일 지연)
    if USE_KMA_LIVE:
        try:
            update_weather_cache(today)
            reload_weather_globals()
            return fetch_weather_mock(today)  # 갱신된 캐시에서 어제 row 반환
        except Exception as e:
            print(f"[KMA live 실패 → mock fallback] {e}")
    return fetch_weather_mock(today)

# ────────────────────────────────────────────────
# 2. 모델 + 캘리브레이션 + 룰 로드 (14_inference_demo와 동일)
# ────────────────────────────────────────────────
RULES = [
    ('강수누적-철골 추락위험',
     lambda r: r['공종'] == '철골공사' and r['강수_연속_7d'] >= 3,
     2.67, '7일 연속 강수 + 고소작업. 비계 결로/미끄럼 점검, 안전대 이중 결속.'),
    ('폭염누적-토공 붕괴위험',
     lambda r: r['공종'] == '토공사' and r['폭염_연속_7d'] >= 2,
     2.26, '폭염 2일+ 누적. 사면 균열·지반 약화 점검, 작업자 휴식 강화.'),
    ('강수누적-기타 붕괴위험',
     lambda r: r['공종'] == '기타' and r['누적강수_7d'] >= 100,
     2.00, '누적 강수 100mm↑. 굴착면·법면 안정성 점검.'),
    ('강수누적-기타(2)',
     lambda r: r['공종'] == '기타' and r['강수_연속_7d'] >= 3,
     1.96, '강수 3일 연속. 침수 자재·전선 재배치, 배수 점검.'),
    ('한파-철골 동결위험',
     lambda r: r['공종'] == '철골공사' and r['최저기온'] <= -5,
     1.87, '저온 노출. 강재 취성 파괴, 용접부 점검.'),
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
     1.38, '강풍 + 콘크리트. 거푸집·자재 비산 주의.'),
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

w = pd.read_parquet(f'{CACHE}/weather_daily.parquet').sort_values('일시').reset_index(drop=True)
w['doy_sin'] = np.sin(2*np.pi * w['일시'].dt.dayofyear / 365)
w['doy_cos'] = np.cos(2*np.pi * w['일시'].dt.dayofyear / 365)
WF = cfg['WEATHER_FEATS']
weather_norm = w[WF + ['doy_sin', 'doy_cos']].values.astype(np.float32).copy()
weather_norm[:, :len(WF)] = (weather_norm[:, :len(WF)] - mean) / (std + 1e-6)
date2idx = {d: i for i, d in enumerate(w['일시'].dt.normalize())}

# 캘리브레이션 (val 절반)
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

# ────────────────────────────────────────────────
# 3. 누적 피처 재계산
# ────────────────────────────────────────────────
def compute_cumulative(end_date):
    win7 = w[(w['일시'] >= pd.Timestamp(end_date) - pd.Timedelta(days=6)) & (w['일시'] <= pd.Timestamp(end_date))]
    today = w[w['일시'] == pd.Timestamp(end_date)].iloc[0]
    def streak(series, cond):
        s = cond(series)
        cnt = 0
        for v in s.values[::-1]:
            if v: cnt += 1
            else: break
        return cnt
    return {
        '평균기온': float(today['평균기온(°C)']), '최고기온': float(today['최고기온(°C)']),
        '최저기온': float(today['최저기온(°C)']), '일강수량': float(today['일강수량(mm)']),
        '풍속':     float(today['평균 풍속(m/s)']), 'Δtavg': float(today['Δtavg']),
        '누적강수_7d':   float(win7['일강수량(mm)'].sum()),
        '폭염_연속_7d': int(streak(win7['최고기온(°C)'], lambda s: s >= 33)),
        '한파_연속_7d': int(streak(win7['최저기온(°C)'], lambda s: s <= -10)),
        '강수_연속_7d': int(streak(win7['일강수량(mm)'], lambda s: s >= 1)),
        '기온변동성_7d': float(win7['평균기온(°C)'].std()),
    }

# ────────────────────────────────────────────────
# 4. 현장별 알람 생성
# ────────────────────────────────────────────────
def evaluate_site(site_id, sido, daebun, gongjong, gongjeong, hour, sogyumo, gowiheom, cumf, today_date):
    rule_row = {**cumf, '공종': gongjong}
    end = pd.Timestamp(today_date)
    row = pd.Series({
        'date': end, '시도구분': sido, '공사대분류': daebun, '공종(소분류)': gongjong,
        '시간': hour, '요일': end.dayofweek,
        '시간bin': int(pd.cut([hour], bins=[-1,5,11,17,23], labels=[0,1,2,3])[0]),
        '공정율_수치': gongjeong, '소규모현장': float(sogyumo), '고위험공종': float(gowiheom),
    })
    pred_raw = predict_rows(row.to_frame().T)
    pred = {h: float(calibrators[h].predict([pred_raw[h][0]])[0]) for h in pred_raw}
    issues = sorted(
        [(name, lift, advice) for name, cond, lift, advice in RULES if cond(rule_row)],
        key=lambda x: -x[1]
    )
    max_model_lift = max(pred[h] / max(base_rates[h], 1e-9) for h in pred)
    triggered = (max_model_lift >= ALERT_LIFT_THRESHOLD) or (len(issues) > 0)
    return {
        'site_id': site_id, 'sido': sido, 'daebun': daebun, 'gongjong': gongjong,
        'gongjeong': gongjeong, 'hour': hour,
        'pred': pred, 'issues': issues, 'triggered': triggered,
        'max_model_lift': max_model_lift,
    }

# ────────────────────────────────────────────────
# 5. Slack 메시지 포맷 + 송신
# ────────────────────────────────────────────────
def format_slack_message(today_date, cumf, alerts):
    today_str = pd.Timestamp(today_date).strftime('%Y-%m-%d (%a)')
    lines = []
    lines.append(f"🚨 *RIA Risk Instinct Alert — {today_str}*")
    lines.append("")
    lines.append(f"📊 *전국 평균 기상*")
    lines.append(f"  • 평균 {cumf['평균기온']:.1f}°C / 최고 {cumf['최고기온']:.1f} / 최저 {cumf['최저기온']:.1f}  (Δtavg {cumf['Δtavg']:+.1f}°C)")
    lines.append(f"  • 강수 {cumf['일강수량']:.1f}mm / 풍속 {cumf['풍속']:.1f}m/s")
    lines.append(f"  • 7일누적: 강수 {cumf['누적강수_7d']:.0f}mm / 폭염 {cumf['폭염_연속_7d']}일 / 한파 {cumf['한파_연속_7d']}일 / 강수 {cumf['강수_연속_7d']}일 연속")
    lines.append("")
    n_alert = sum(1 for a in alerts if a['triggered'])
    lines.append(f"🏗️ *현장별 알람 ({n_alert}/{len(alerts)} 발동)*")
    lines.append("")
    for a in alerts:
        if not a['triggered']:
            continue
        lines.append(f"▸ *{a['site_id']}* — {a['sido']} / {a['gongjong']} (공정율 {a['gongjeong']}%)")
        for h in ['중대재해','다중사상','외국인피해','고령자피해']:
            v = a['pred'][h]
            lift = v / max(base_rates[h], 1e-9)
            if lift >= 1.5:
                lines.append(f"   • 모델 — {h} {v*100:.1f}% (Lift {lift:.2f}x ⚠)")
        for name, lift, advice in a['issues']:
            lines.append(f"   • 룰 — {name} (Lift {lift:.2f}x)")
            lines.append(f"     ↳ {advice}")
        lines.append("")
    if n_alert == 0:
        lines.append("_오늘은 모든 현장이 안정 범위입니다._")
    return '\n'.join(lines)

def send_slack_live(message):
    """SLACK_WEBHOOK가 있으면 incoming webhook 방식, 없으면 chat.postMessage(chat:write 스코프 필요)"""
    if SLACK_WEBHOOK:
        body = json.dumps({'text': message}).encode('utf-8')
        req = urllib.request.Request(
            SLACK_WEBHOOK, data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            txt = r.read().decode('utf-8')
        return {'ok': txt.strip() == 'ok', 'method': 'webhook', 'response': txt}
    body = json.dumps({'channel': SLACK_CHANNEL, 'text': message}).encode('utf-8')
    req = urllib.request.Request(
        'https://slack.com/api/chat.postMessage',
        data=body,
        headers={
            'Authorization': f'Bearer {SLACK_TOKEN}',
            'Content-Type': 'application/json; charset=utf-8',
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        out = json.loads(r.read().decode('utf-8'))
    out['method'] = 'chat.postMessage'
    return out

# ────────────────────────────────────────────────
# 6. 메인 루프
# ────────────────────────────────────────────────
def run_once():
    print("=" * 80)
    print(f"  RIA 운영 루프  |  KMA={'live' if USE_KMA_LIVE else 'mock'}  Slack={'live' if USE_SLACK_LIVE else 'dry-run'}")
    print("=" * 80)

    today_row = get_today()
    today_date = pd.Timestamp(today_row['일시']).normalize()
    print(f"\n[1] 오늘 날짜 = {today_date.date()}")
    print(f"    당일 기상: 평균 {today_row['평균기온(°C)']:.1f}°C, 강수 {today_row['일강수량(mm)']:.1f}mm")

    cumf = compute_cumulative(today_date)
    print(f"    7일 누적: 강수 {cumf['누적강수_7d']:.0f}mm, 폭염연속 {cumf['폭염_연속_7d']}일, 한파연속 {cumf['한파_연속_7d']}일")

    print(f"\n[2] 현장 {len(SITES)}개 평가 중...")
    alerts = []
    for site in SITES:
        a = evaluate_site(*site, cumf=cumf, today_date=today_date)
        alerts.append(a)
        flag = '⚠ ALERT' if a['triggered'] else 'ok'
        print(f"    [{flag:>8}] {a['site_id']}: 모델 max Lift {a['max_model_lift']:.2f}x, 룰 {len(a['issues'])}개")

    msg = format_slack_message(today_date, cumf, alerts)
    print(f"\n[3] Slack 메시지 (미리보기):")
    print("-" * 80)
    print(msg)
    print("-" * 80)

    if USE_SLACK_LIVE:
        print(f"\n[4] Slack POST → {SLACK_CHANNEL}")
        try:
            resp = send_slack_live(msg)
            if resp.get('ok'):
                print(f"    ✓ 전송 성공 (ts={resp.get('ts')})")
            else:
                print(f"    ✗ Slack error: {resp.get('error')}")
                if resp.get('error') == 'missing_scope':
                    print(f"    필요 스코프: {resp.get('needed')}, 현재: {resp.get('provided')}")
        except Exception as e:
            print(f"    ✗ HTTP 실패: {e}")
    else:
        print(f"\n[4] Slack DRY-RUN (USE_SLACK_LIVE=1 로 전송 활성화)")

if __name__ == '__main__':
    run_once()
