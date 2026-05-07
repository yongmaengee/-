"""Phase 5c — 카테고리 기반 이슈 체크리스트 (확률값 숨김)
- 모델 출력(Lift) → 위험 카테고리 매핑 (사람이 이해하는 이름)
- 카테고리별 사전검증 액션 코퍼스
- 메시지에서 확률/Lift 노출 X — '이슈 리스트' 정체성 유지
"""
import os, sys, json
from datetime import datetime, timedelta
import urllib.request, urllib.parse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression

ROOT = '/home/nuri5/바탕화면/공모전'
CACHE = f'{ROOT}/cache'

# ───── 설정 ─────
KMA_KEY       = os.environ.get('KMA_KEY', '')        # data.go.kr 일반인증키
KMA_ENDPOINT  = 'https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
SLACK_WEBHOOK = os.environ.get('SLACK_WEBHOOK', '')  # https://hooks.slack.com/services/...
USE_KMA_LIVE   = os.environ.get('USE_KMA_LIVE', '0') == '1'
USE_SLACK_LIVE = os.environ.get('USE_SLACK_LIVE', '0') == '1'

KMA_STATIONS = {108: '서울', 159: '부산', 143: '대구', 112: '인천', 156: '광주', 133: '대전'}
SIDO_PLACEHOLDER = '경기도'

ALERT_LIFT_THRESHOLD = 1.5

# 모니터링 현장 (지역 X, 공종 중심)
SITES = [
    {'id': '철근콘크리트 현장', '공종': '철근콘크리트공사',  '공정율': 50, '시간': 14, '소규모': 0, '고위험': 0},
    {'id': '토공 현장',         '공종': '토공사',             '공정율': 30, '시간': 10, '소규모': 0, '고위험': 0},
    {'id': '철골 고소작업 현장', '공종': '철골공사',           '공정율': 70, '시간': 9,  '소규모': 0, '고위험': 1},
    {'id': '해체·철거 현장',     '공종': '해체 및 철거공사',  '공정율': 80, '시간': 13, '소규모': 0, '고위험': 1},
    {'id': '도장 마감 현장',     '공종': '도장공사',           '공정율': 90, '시간': 14, '소규모': 1, '고위험': 0},
    {'id': '기계설비 현장',     '공종': '기계설비공사',       '공정율': 60, '시간': 11, '소규모': 0, '고위험': 0},
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 위험 카테고리 매핑 — (모델 헤드, 공종) → 카테고리명
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK_CATEGORIES = {
    # 중대재해 (사망) — 공종별로 다른 메커니즘
    ('중대재해', '도장공사'):           '마감 공정 화재·유증기 노출',
    ('중대재해', '철골공사'):           '고소작업 추락·강재 결손',
    ('중대재해', '철근콘크리트공사'):    '거푸집·양생 단계 사고',
    ('중대재해', '토공사'):             '굴착·사면 작업 안전 위험',
    ('중대재해', '해체 및 철거공사'):    '구조물 해체 붕괴',
    ('중대재해', '기계설비공사'):       '기계 설치·전기 작업',
    ('중대재해', '*'):                 '중대 안전사고 발생 가능성',
    # 외국인
    ('외국인피해', '*'):               '다국적 작업자 안전관리',
    # 고령자
    ('고령자피해', '토공사'):          '고령 작업자 열사병·부상',
    ('고령자피해', '철골공사'):        '고령 작업자 고소 사고',
    ('고령자피해', '*'):               '고령 작업자 건강·안전 모니터링',
    # 다중사상
    ('다중사상', '*'):                 '집단 동시작업 사고 확산',
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 카테고리별 액션 코퍼스 (사전 검증)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTIONS_BY_CATEGORY = {
    '마감 공정 화재·유증기 노출': [
        '환기 시스템 가동 상태 확인',
        '인화물·전열기 작업구역 격리',
        '유증기 농도 측정 (LEL 25% 이하 유지)',
        '소화기·비상대피로 동선 점검',
    ],
    '고소작업 추락·강재 결손': [
        '안전대·안전모 결속 상태 점검',
        '비계 결합부·발판 미끄럼 확인',
        '용접부 강재 취성 점검 (저온 시)',
        '고소작업 페어 작업 의무화',
    ],
    '거푸집·양생 단계 사고': [
        '거푸집 결박 상태 점검',
        '양생 시트·자재 결박 확인 (강풍 대비)',
        '콘크리트 타설 인원 동선 분리',
    ],
    '굴착·사면 작업 안전 위험': [
        '사면 균열·변위 모니터링',
        '굴착면 함수율·배수로 점검',
        '폭염·강수 시 작업 시간 분산',
    ],
    '구조물 해체 붕괴': [
        '응력 변화 모니터링',
        '단계 해체 속도 조절',
        '비상 대피로 확보·예행 연습',
    ],
    '기계 설치·전기 작업': [
        '누전 차단기 점검',
        '접지선·습기 보호 커버 확인',
        '전선 노출부 절연 점검',
    ],
    '중대 안전사고 발생 가능성': [
        '작업 전 안전브리핑 (5분 의무)',
        '안전 장비 결속 점검',
        '비상 대응 매뉴얼 숙지 확인',
    ],
    '다국적 작업자 안전관리': [
        '다국어 안전수칙 게시 확인',
        '핵심 안내 통역 인력 배치',
        '보호구 사이즈 적합성 점검 (체형 차이)',
    ],
    '고령 작업자 열사병·부상': [
        '고령 작업자 작업 시간 분산',
        '시간당 10분 휴식 강제',
        '체온·혈압·열사병 증상 모니터링',
    ],
    '고령 작업자 고소 사고': [
        '고령 작업자 고소작업 제외 검토',
        '체력 부담 큰 작업 페어링',
        '근골격계 부상 예방 스트레칭',
    ],
    '고령 작업자 건강·안전 모니터링': [
        '시간당 10분 휴식 강제',
        '혈압·열사병·동상 증상 모니터링',
        '응급 대응 체계 확인',
    ],
    '집단 동시작업 사고 확산': [
        '동시 작업 인원 제한 점검',
        '비상 대피로 확보 및 표시',
        '집단작업 시 안전관리자 상주',
    ],
}

# 룰 기반 (이미 카테고리 형태) — 이름 + 액션
RULES = [
    ('철골 강수누적 추락 위험',
     lambda r: r['공종'] == '철골공사' and r['강수_연속_7d'] >= 3,
     ['비계 결로·미끄럼 점검', '안전대 이중 결속', '고소작업 페어 작업 의무화']),
    ('토공 폭염누적 사면 위험',
     lambda r: r['공종'] == '토공사' and r['폭염_연속_7d'] >= 2,
     ['사면 균열·지반 약화 점검', '시간당 휴식 10분 이상', '폭염 경보 시 옥외작업 중단']),
    ('굴착 강수누적 붕괴 위험',
     lambda r: r['공종'] == '기타' and r['누적강수_7d'] >= 100,
     ['굴착면·법면 안정성 점검', '배수로 점검', '지반 변위 모니터링']),
    ('철골 한파 동결 위험',
     lambda r: r['공종'] == '철골공사' and r['최저기온'] <= -5,
     ['용접부 강재 취성 점검', '안전대 결로 확인', '결빙 발판 제거']),
    ('토공 강수누적 사면 위험',
     lambda r: r['공종'] == '토공사' and r['강수_연속_7d'] >= 3,
     ['사면 변위 모니터링', '굴착 깊이 일시 축소', '지반 함수율 측정']),
    ('기계설비 강수누적 누전 위험',
     lambda r: r['공종'] == '기계설비공사' and r['누적강수_7d'] >= 100,
     ['누전 차단기 점검', '접지선 확인', '습기 보호 커버 부착']),
    ('토공 한파 동결-융해 위험',
     lambda r: r['공종'] == '토공사' and r['최저기온'] <= -5,
     ['동결-융해 사면 점검', '결빙 진입로 제염', '굴착면 모니터링']),
    ('철골 누적강수 비계침하 위험',
     lambda r: r['공종'] == '철골공사' and r['누적강수_7d'] >= 100,
     ['비계 침하 점검', '결로 발판 미끄럼 방지', '체결부 부식 확인']),
    ('해체철거 기온변동 응력 위험',
     lambda r: r['공종'] == '해체 및 철거공사' and r['기온변동성_7d'] >= 4,
     ['구조물 응력 모니터링', '단계 해체 속도 조절', '비상 대피로 확보']),
    ('철근콘크리트 강풍 비산 위험',
     lambda r: r['공종'] == '철근콘크리트공사' and r['풍속'] >= 4,
     ['거푸집·자재 결박 점검', '고소 자재 비산 방지', '양생 시트 보강']),
]

# ───── 모델 + 캘리브레이션 + 캐시 ─────
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
model.load_state_dict(ckpt['model_state']); model.eval()

WF = cfg['WEATHER_FEATS']

def reload_weather():
    global w, weather_norm, date2idx
    w = pd.read_parquet(f'{CACHE}/weather_daily.parquet').sort_values('일시').reset_index(drop=True)
    w['doy_sin'] = np.sin(2*np.pi * w['일시'].dt.dayofyear / 365)
    w['doy_cos'] = np.cos(2*np.pi * w['일시'].dt.dayofyear / 365)
    weather_norm = w[WF + ['doy_sin', 'doy_cos']].values.astype(np.float32).copy()
    weather_norm[:, :len(WF)] = (weather_norm[:, :len(WF)] - mean) / (std + 1e-6)
    date2idx = {d: i for i, d in enumerate(w['일시'].dt.normalize())}
reload_weather()

# 캘리브레이션
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
            for h in HEADS: out[h].append(torch.sigmoid(logits[h]).numpy())
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
    ir = IsotonicRegression(out_of_bounds='clip'); ir.fit(cal_preds[h], cal_labels[h])
    calibrators[h] = ir
    base_rates[h] = cal_labels[h].mean()

# ───── KMA fetch + 캐시 갱신 (16과 동일) ─────
def _f(v):
    if v is None or v == '' or v == ' ': return np.nan
    try: return float(v)
    except: return np.nan

def fetch_kma_single_day(date_str):
    rows = []
    for stn in KMA_STATIONS:
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
            if not items: continue
            it = items[0]
            rows.append({
                '평균기온(°C)': _f(it.get('avgTa')), '최고기온(°C)': _f(it.get('maxTa')),
                '최저기온(°C)': _f(it.get('minTa')),
                '일강수량(mm)': _f(it.get('sumRn')) if it.get('sumRn') else 0.0,
                '평균 풍속(m/s)': _f(it.get('avgWs')),
            })
        except Exception as e:
            print(f"    WARN stn={stn}: {e}")
    if not rows:
        raise RuntimeError(f"KMA fetch 실패: {date_str}")
    df = pd.DataFrame(rows)
    return pd.Series({
        '일시': pd.Timestamp(date_str),
        '평균기온(°C)': df['평균기온(°C)'].mean(),
        '최고기온(°C)': df['최고기온(°C)'].mean(),
        '최저기온(°C)': df['최저기온(°C)'].mean(),
        '일강수량(mm)': df['일강수량(mm)'].mean(),
        '평균 풍속(m/s)': df['평균 풍속(m/s)'].mean(),
    })

def update_weather_cache(yesterday):
    w_local = pd.read_parquet(f'{CACHE}/weather_daily.parquet').sort_values('일시').reset_index(drop=True)
    last = w_local['일시'].max()
    if last >= yesterday:
        print(f"  캐시 최신 ({last.date()})")
        return
    clim = w_local.groupby(w_local['일시'].dt.dayofyear)[['평균기온(°C)','최고기온(°C)','최저기온(°C)','일강수량(mm)','평균 풍속(m/s)']].mean()
    clim.columns = ['tavg_clim','tmax_clim','tmin_clim','rain_clim','wind_clim']
    new_rows = []
    cur = last + pd.Timedelta(days=1)
    while cur <= yesterday:
        try: new_rows.append(fetch_kma_single_day(cur.strftime('%Y%m%d')))
        except Exception as e: print(f"    SKIP {cur.date()}: {e}")
        cur += pd.Timedelta(days=1)
    if not new_rows: return
    new = pd.DataFrame(new_rows); new['doy'] = new['일시'].dt.dayofyear
    new = new.merge(clim, left_on='doy', right_index=True, how='left')
    new['Δtavg'] = new['평균기온(°C)'] - new['tavg_clim']
    new['Δtmax'] = new['최고기온(°C)'] - new['tmax_clim']
    new['Δtmin'] = new['최저기온(°C)'] - new['tmin_clim']
    new['Δrain'] = new['일강수량(mm)'] - new['rain_clim']
    new['Δwind'] = new['평균 풍속(m/s)'] - new['wind_clim']
    new['일교차'] = new['최고기온(°C)'] - new['최저기온(°C)']
    new['doy_sin'] = np.sin(2*np.pi*new['doy']/365); new['doy_cos'] = np.cos(2*np.pi*new['doy']/365)
    new['year_idx'] = new['일시'].dt.year - 2019
    new = new.reindex(columns=w_local.columns.tolist())
    out = pd.concat([w_local, new], ignore_index=True).sort_values('일시').reset_index(drop=True)
    out['tavg_5dma'] = out['평균기온(°C)'].rolling(5, min_periods=1).mean()
    out.to_parquet(f'{CACHE}/weather_daily.parquet')
    print(f"  KMA fetch +{len(new_rows)}일")

def get_today():
    today = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=1)
    if USE_KMA_LIVE:
        try:
            update_weather_cache(today); reload_weather()
        except Exception as e:
            print(f"[KMA fail → cache fallback] {e}")
    sub = w[w['일시'] <= today]
    return sub.iloc[-1] if len(sub) else w.iloc[-1]

def compute_cumulative(end_date):
    win7 = w[(w['일시'] >= pd.Timestamp(end_date) - pd.Timedelta(days=6)) & (w['일시'] <= pd.Timestamp(end_date))]
    today = w[w['일시'] == pd.Timestamp(end_date)].iloc[0]
    def streak(series, cond):
        s = cond(series); cnt = 0
        for v in s.values[::-1]:
            if v: cnt += 1
            else: break
        return cnt
    return {
        '평균기온': float(today['평균기온(°C)']), '최고기온': float(today['최고기온(°C)']),
        '최저기온': float(today['최저기온(°C)']), '일강수량': float(today['일강수량(mm)']),
        '풍속': float(today['평균 풍속(m/s)']), 'Δtavg': float(today['Δtavg']),
        '누적강수_7d': float(win7['일강수량(mm)'].sum()),
        '폭염_연속_7d': int(streak(win7['최고기온(°C)'], lambda s: s >= 33)),
        '한파_연속_7d': int(streak(win7['최저기온(°C)'], lambda s: s <= -10)),
        '강수_연속_7d': int(streak(win7['일강수량(mm)'], lambda s: s >= 1)),
        '기온변동성_7d': float(win7['평균기온(°C)'].std()),
    }

def evaluate_site(site, cumf, today_date):
    end = pd.Timestamp(today_date)
    rule_row = {**cumf, '공종': site['공종']}
    row = pd.Series({
        'date': end, '시도구분': SIDO_PLACEHOLDER, '공사대분류': '건축',
        '공종(소분류)': site['공종'], '시간': site['시간'], '요일': end.dayofweek,
        '시간bin': int(pd.cut([site['시간']], bins=[-1,5,11,17,23], labels=[0,1,2,3])[0]),
        '공정율_수치': site['공정율'], '소규모현장': float(site['소규모']),
        '고위험공종': float(site['고위험']),
    })
    pred_raw = predict_rows(row.to_frame().T)
    pred = {h: float(calibrators[h].predict([pred_raw[h][0]])[0]) for h in pred_raw}

    # 카테고리 수집 (룰 + 모델)
    cats = []
    seen_actions = set()

    # 1) 룰 카테고리 (deterministic)
    for rule_name, cond, actions in RULES:
        if cond(rule_row):
            new_actions = [a for a in actions if a not in seen_actions]
            for a in new_actions: seen_actions.add(a)
            cats.append({'name': rule_name, 'source': 'rule', 'actions': new_actions, 'sort_key': 0})

    # 2) 모델 카테고리 (Lift ≥ 1.5인 헤드만)
    for head in ['중대재해', '다중사상', '외국인피해', '고령자피해']:
        v = pred[head]
        lift = v / max(base_rates[head], 1e-9)
        if lift < ALERT_LIFT_THRESHOLD:
            continue
        cat_name = (RISK_CATEGORIES.get((head, site['공종']))
                    or RISK_CATEGORIES.get((head, '*')))
        if not cat_name:
            continue
        actions = ACTIONS_BY_CATEGORY.get(cat_name, [])
        new_actions = [a for a in actions if a not in seen_actions]
        for a in new_actions: seen_actions.add(a)
        if new_actions:  # 액션이 모두 중복이면 카테고리 생략
            cats.append({'name': cat_name, 'source': 'model', 'actions': new_actions,
                         'sort_key': -lift})

    cats.sort(key=lambda c: c['sort_key'])
    return {'site': site, 'pred': pred, 'cats': cats}

# ───── 체크리스트 메시지 (Lift 노출 X) ─────
def format_checklist(today_date, cumf, evals):
    today_str = pd.Timestamp(today_date).strftime('%Y-%m-%d (%a)')
    lines = []
    lines.append(f"🚨 *RIA Risk Instinct Alert — {today_str}*")
    lines.append("")
    lines.append(f"🌤️ *오늘의 전국 기상*")
    lines.append(f"   • 평균 {cumf['평균기온']:.1f}°C  /  최고 {cumf['최고기온']:.1f}  /  최저 {cumf['최저기온']:.1f}")
    delta_sign = '+' if cumf['Δtavg'] >= 0 else ''
    lines.append(f"   • 평년편차 {delta_sign}{cumf['Δtavg']:.1f}°C  /  강수 {cumf['일강수량']:.1f}mm  /  풍속 {cumf['풍속']:.1f}m/s")
    cumul_parts = []
    if cumf['누적강수_7d'] > 5:    cumul_parts.append(f"누적강수 {cumf['누적강수_7d']:.0f}mm")
    if cumf['폭염_연속_7d'] >= 1: cumul_parts.append(f"폭염 {cumf['폭염_연속_7d']}일 연속")
    if cumf['한파_연속_7d'] >= 1: cumul_parts.append(f"한파 {cumf['한파_연속_7d']}일 연속")
    if cumf['강수_연속_7d'] >= 2: cumul_parts.append(f"강수 {cumf['강수_연속_7d']}일 연속")
    if cumul_parts:
        lines.append(f"   • 7일 누적: {' · '.join(cumul_parts)}")
    lines.append("")
    lines.append(f"📋 *오늘의 안전 이슈 체크리스트*")
    lines.append("")

    triggered = [ev for ev in evals if ev['cats']]
    if not triggered:
        lines.append("   _오늘은 모든 공종이 안정 범위입니다._")
        return '\n'.join(lines)

    # 매그니튜드 큰 현장 먼저 (sort_key 합)
    triggered.sort(key=lambda ev: min(c['sort_key'] for c in ev['cats']))

    for ev in triggered:
        site = ev['site']
        tags = []
        if site['고위험']:  tags.append('🟥 고위험공종')
        if site['소규모']:  tags.append('🟨 소규모현장')
        tag_str = '   ' + '  '.join(tags) if tags else ''
        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"▣ *{site['id']}*   _공정율 {site['공정율']}% · {site['시간']}시 작업_{tag_str}")
        lines.append("")

        for cat in ev['cats']:
            icon = '🔴' if cat['source'] == 'model' else '🟡'
            lines.append(f"   {icon} *{cat['name']}*")
            for a in cat['actions']:
                lines.append(f"      ☐ {a}")
            lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("_🔴 모델 신호  ·  🟡 룰 활성  ·  체크 항목은 작업 전 안전미팅에서 확인_")
    return '\n'.join(lines)

def send_slack(message):
    if not SLACK_WEBHOOK:
        return {'ok': False, 'error': 'SLACK_WEBHOOK 미설정'}
    body = json.dumps({'text': message}).encode('utf-8')
    req = urllib.request.Request(SLACK_WEBHOOK, data=body,
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=10) as r:
        txt = r.read().decode('utf-8')
    return {'ok': txt.strip() == 'ok', 'response': txt}

# ───── main ─────
def run():
    print("=" * 80)
    print(f"  RIA 알람 v3 (카테고리 체크리스트)  |  KMA={'live' if USE_KMA_LIVE else 'mock'}  Slack={'live' if USE_SLACK_LIVE else 'dry'}")
    print("=" * 80)

    today_row = get_today()
    today_date = pd.Timestamp(today_row['일시']).normalize()
    cumf = compute_cumulative(today_date)
    print(f"\n오늘 = {today_date.date()},  평균 {cumf['평균기온']:.1f}°C,  Δtavg {cumf['Δtavg']:+.1f}°C")

    evals = [evaluate_site(s, cumf, today_date) for s in SITES]
    msg = format_checklist(today_date, cumf, evals)
    print("\n" + "─" * 80)
    print(msg)
    print("─" * 80)

    if USE_SLACK_LIVE:
        r = send_slack(msg)
        print(f"\n[Slack] {'✓ 전송 성공' if r.get('ok') else '✗ 실패'}: {r}")
    else:
        print("\n[Slack] DRY-RUN  (USE_SLACK_LIVE=1로 전송 활성화)")

if __name__ == '__main__':
    run()
