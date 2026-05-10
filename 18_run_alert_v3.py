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

ROOT = os.path.dirname(os.path.abspath(__file__))
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 추상 환경 타입 — 특정 공종 대신 작업 환경으로 분류
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENVIRONMENTS = [
    {'id': '야외 굴착·토공',   '공종_proxy': '토공사',           '고위험': 0},
    {'id': '고소·구조물 작업', '공종_proxy': '철골공사',         '고위험': 1},
    {'id': '지하·밀폐 공간',   '공종_proxy': '기계설비공사',     '고위험': 0},
    {'id': '다중 혼재 작업',   '공종_proxy': '철근콘크리트공사', '고위험': 0},
    {'id': '마감·설비·전기',   '공종_proxy': '도장공사',         '고위험': 0},
    {'id': '해체·철거',        '공종_proxy': '해체 및 철거공사', '고위험': 1},
]

# 모델 추론용 대표 파라미터 (공종_proxy 기반)
ENV_MODEL_PARAMS = {
    '야외 굴착·토공':   {'공정율': 50, '시간': 10, '소규모': 0},
    '고소·구조물 작업': {'공정율': 70, '시간': 9,  '소규모': 0},
    '지하·밀폐 공간':   {'공정율': 60, '시간': 11, '소규모': 0},
    '다중 혼재 작업':   {'공정율': 50, '시간': 14, '소규모': 0},
    '마감·설비·전기':   {'공정율': 90, '시간': 14, '소규모': 1},
    '해체·철거':        {'공정율': 80, '시간': 13, '소규모': 0},
}

def detect_scenario(cumf):
    if cumf['누적강수_7d'] >= 80 or cumf['강수_연속_7d'] >= 3: return '집중호우'
    if cumf['폭염_연속_7d'] >= 2:  return '장기가뭄·폭염'
    if cumf['풍속'] >= 6:           return '태풍'
    if cumf['한파_연속_7d'] >= 2:  return '한파'
    return '정상'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 위험 카테고리 매핑 — (모델 헤드, 공종) → 카테고리명
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK_CATEGORIES = {
    # 중대재해 (사망) — 공종별로 다른 메커니즘
    ('중대재해', '도장공사'):           '도장·마감 공정 화재 및 유증기 노출',
    ('중대재해', '철골공사'):           '철골 고소작업 추락 및 부재 취약',
    ('중대재해', '철근콘크리트공사'):    '거푸집·동바리 및 타설 단계 사고',
    ('중대재해', '토공사'):             '굴착·사면 붕괴 및 장비 동선 위험',
    ('중대재해', '해체 및 철거공사'):    '해체 순서 오류 및 구조물 붕괴',
    ('중대재해', '기계설비공사'):       '기계설비 설치 및 전기 작업 위험',
    ('중대재해', '*'):                 '중대 안전사고 발생 가능성',
    # 외국인
    ('외국인피해', '*'):               '다국적 작업자 안전관리',
    # 고령자
    ('고령자피해', '토공사'):          '고령 작업자 옥외작업 부담',
    ('고령자피해', '철골공사'):        '고령 작업자 고소작업 부담',
    ('고령자피해', '*'):               '고령 작업자 건강·안전 모니터링',
    # 다중사상
    ('다중사상', '철근콘크리트공사'):   '타설·양생 구간 동시작업 사고 확산',
    ('다중사상', '해체 및 철거공사'):   '해체 구간 작업자 대피 동선 위험',
    ('다중사상', '*'):                 '집단 동시작업 사고 확산',
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 카테고리별 액션 코퍼스 (사전 검증)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTIONS_BY_CATEGORY = {
    '도장·마감 공정 화재 및 유증기 노출': [
        '도장·방수 작업구역 환기 상태 확인',
        '인화성 자재와 용접·절단 작업 동선 분리',
        '밀폐공간 유증기 농도 측정 및 기록',
        '소화기 위치와 비상대피 동선 재확인',
    ],
    '철골 고소작업 추락 및 부재 취약': [
        '안전대 이중걸이와 생명줄 체결 상태 확인',
        '비계·작업발판 결로, 침하, 흔들림 점검',
        '저온·강풍 시 양중 및 고소작업 순서 재검토',
        '용접부와 접합부 균열·취성 징후 확인',
    ],
    '거푸집·동바리 및 타설 단계 사고': [
        '거푸집·동바리 체결부와 수평재 누락 확인',
        '타설 구간 작업자·장비 동선 분리',
        '강풍 시 양생포·자재 결속 상태 보강',
        '타설 전 변형·처짐 계측값 확인',
    ],
    '굴착·사면 붕괴 및 장비 동선 위험': [
        '사면 균열·배부름·용수 발생 여부 점검',
        '굴착면 배수로와 집수정 기능 확인',
        '장비 회전반경 내 보행자 접근 통제',
        '강수·폭염 시 굴착 깊이와 작업시간 조정',
    ],
    '해체 순서 오류 및 구조물 붕괴': [
        '해체 순서와 임시 지지 계획 재확인',
        '잔존 구조물 균열·기울어짐 육안 점검',
        '낙하물 방호구역과 출입통제선 재설정',
        '비상 대피로와 신호수 배치 확인',
    ],
    '기계설비 설치 및 전기 작업 위험': [
        '임시전기 분전반 누전차단기 시험',
        '접지선 체결과 케이블 피복 손상 확인',
        '우천 후 전선·콘센트 습기 보호 조치',
        '양중 장비와 작업자 신호체계 확인',
    ],
    '중대 안전사고 발생 가능성': [
        '작업 전 안전브리핑 (5분 의무)',
        '당일 고위험 작업 허가서 재확인',
        '보호구 착용과 체결 상태 상호 점검',
        '비상 연락망과 대피 집결지 공유',
    ],
    '다국적 작업자 안전관리': [
        '다국어 TBM 자료와 위험구역 표지 확인',
        '작업 전 핵심 위험요인 통역 전달',
        '보호구 사이즈와 착용법 현장 확인',
        '신규·단기 투입 인력의 작업범위 제한',
    ],
    '고령 작업자 옥외작업 부담': [
        '고령 작업자 옥외 연속작업 시간 제한',
        '폭염·한파 시간대 작업 전환 검토',
        '체온·혈압·탈수 증상 모니터링',
        '장비 유도·신호 업무 우선 배치 검토',
    ],
    '고령 작업자 고소작업 부담': [
        '고령 작업자 고소작업 투입 적정성 재검토',
        '사다리·발판 작업 시 보조자 동행',
        '무거운 자재 운반 작업 분산',
        '근골격계 부담 작업 전 스트레칭 실시',
    ],
    '고령 작업자 건강·안전 모니터링': [
        '건강 이상자 사전 확인 및 작업 조정',
        '시간당 휴식과 수분·보온 조치 확인',
        '혈압·열사병·동상 증상 모니터링',
        '응급 대응 담당자와 이송 동선 확인',
    ],
    '타설·양생 구간 동시작업 사고 확산': [
        '타설 구간 동시 투입 인원 제한',
        '펌프카·레미콘 차량 유도자 배치',
        '작업층 하부 출입통제 확인',
        '비상정지 신호와 대피 동선 공유',
    ],
    '해체 구간 작업자 대피 동선 위험': [
        '해체 반경 내 동시작업 금지구역 설정',
        '상하부 작업자 간 무전 신호체계 확인',
        '분진·소음으로 인한 경보 전달 가능성 점검',
        '대피로 적치물 제거',
    ],
    '집단 동시작업 사고 확산': [
        '동시 작업 인원과 작업면 간격 제한',
        '비상 대피로 확보 및 표시',
        '집단작업 시 안전관리자 상주',
        '작업 전 정지 신호와 대피 신호 통일',
    ],
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 날씨 시나리오 × 환경 타입 → 주요 이슈 (선언적 테이블)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENV_SCENARIO_ACTIONS = {
    # ── 야외 굴착·토공 ────────────────────────────
    ('야외 굴착·토공', '집중호우'): [
        {'name': '사면·굴착면 붕괴', 'actions': [
            '사면 균열·용수 즉시 점검', '굴착면 배수로·집수정 확인', '강우 지속 시 작업 중단 검토']},
        {'name': '중장비 동선 위험', 'actions': [
            '중장비 회전반경 내 접근 통제', '진입로 노면 침하·미끄럼 점검']},
    ],
    ('야외 굴착·토공', '장기가뭄·폭염'): [
        {'name': '지반건조균열·열탈진', 'actions': [
            '굴착면 건조균열 현황 확인', '10~15시 옥외 작업 중단 검토', '시간당 10분 휴식·음용수 확보']},
    ],
    ('야외 굴착·토공', '태풍'): [
        {'name': '가시설·흙막이 변형', 'actions': [
            '흙막이 버팀재·앵커 체결 점검', '임시 자재·가시설 결속 강화', '강풍 시 굴착 구간 출입 통제']},
    ],
    ('야외 굴착·토공', '한파'): [
        {'name': '동결융해 사면 불안정', 'actions': [
            '동결-융해 구간 사면 안정성 확인', '진입로 결빙 제염', '장비 저온 유압·오일 점검']},
    ],
    # ── 고소·구조물 작업 ──────────────────────────
    ('고소·구조물 작업', '집중호우'): [
        {'name': '발판·안전대 결로·미끄럼', 'actions': [
            '비계·작업발판 결로 미끄럼 방지 조치', '안전대 이중걸이·생명줄 재확인', '강우 시 고소작업 중단 기준 공지']},
    ],
    ('고소·구조물 작업', '장기가뭄·폭염'): [
        {'name': '고소 작업자 열탈진', 'actions': [
            '고소 연속 작업 30분 초과 제한', '열탈진 증상 모니터링 및 교대', '냉각용품 지급']},
    ],
    ('고소·구조물 작업', '태풍'): [
        {'name': '추락·비산물 충돌', 'actions': [
            '순간풍속 10m/s 초과 시 고소작업 즉시 중단', '발판 위 자재 고정·낙하물 방지망 점검', '양중 작업 중지 및 장비 고정']},
    ],
    ('고소·구조물 작업', '한파'): [
        {'name': '발판 결빙·강재 취성', 'actions': [
            '발판·계단 결빙 제거', '강재 접합부·용접부 취성 점검', '안전대 버클 결빙 확인']},
    ],
    # ── 지하·밀폐 공간 ────────────────────────────
    ('지하·밀폐 공간', '집중호우'): [
        {'name': '침수·산소결핍', 'actions': [
            '배수펌프 작동 상태 확인', '침수 감지 경보 장치 점검', '비상 대피로 침수 여부 확인']},
    ],
    ('지하·밀폐 공간', '장기가뭄·폭염'): [
        {'name': '열기 집적·환기 부족', 'actions': [
            '환기팬 작동 및 산소농도 측정', '작업자 체온 모니터링·교대', '냉방·공기순환 장비 추가 검토']},
    ],
    ('지하·밀폐 공간', '태풍'): [
        {'name': '진동·균열 구조 위험', 'actions': [
            '지하 구조물 균열·침하 징후 점검', '지상 낙하물 진입로 차단 확인', '비상 대피 신호 체계 확인']},
    ],
    ('지하·밀폐 공간', '한파'): [
        {'name': '설비 동결·배관 파열', 'actions': [
            '동결 우려 배관·밸브 보온 확인', '임시 난방 장치 CO 농도 모니터링', '지하 작업자 방한용품 착용 확인']},
    ],
    # ── 다중 혼재 작업 ────────────────────────────
    ('다중 혼재 작업', '집중호우'): [
        {'name': '혼재 구역 동시 사고 확산', 'actions': [
            '공종별 작업 구역 우선순위 재조정', '비상 집결지·대피 동선 전 공종 공지', '상하부 동시 작업 중단 검토']},
    ],
    ('다중 혼재 작업', '장기가뭄·폭염'): [
        {'name': '집단 열탈진·작업 혼선', 'actions': [
            '공종별 교대 휴식 스케줄 조정', '혼재 구역 안전관리자 상주', '고위험 공종 우선 철수']},
    ],
    ('다중 혼재 작업', '태풍'): [
        {'name': '대피 혼잡·다중 피해', 'actions': [
            '전 공종 동시 대피 훈련 실시', '혼재 구역 자재 결속 긴급 실시', '신호수 배치로 대피 동선 통제']},
    ],
    ('다중 혼재 작업', '한파'): [
        {'name': '방한복 착용 작업 부주의', 'actions': [
            '방한복 착용 상태 작업 범위 재점검', '공종 간 신호 체계 명확화', '고령·외국인 방한용품 지급 확인']},
    ],
    # ── 마감·설비·전기 ────────────────────────────
    ('마감·설비·전기', '집중호우'): [
        {'name': '누전·감전 위험', 'actions': [
            '분전반 누전차단기 동작 시험', '접지선·케이블 피복 손상 확인', '습기 노출 전선 보호 커버 부착']},
    ],
    ('마감·설비·전기', '장기가뭄·폭염'): [
        {'name': '자재 열변형·화재', 'actions': [
            '인화성 자재 직사광선 차단 보관', '용접·절단 작업 화재 감시자 배치', '소화기 위치 재확인']},
    ],
    ('마감·설비·전기', '태풍'): [
        {'name': '마감 자재 비산·낙하', 'actions': [
            '외벽·지붕 마감 자재 결속 긴급 점검', '임시 가설물 고정 확인', '강풍 시 외부 작업 즉시 중단']},
    ],
    ('마감·설비·전기', '한파'): [
        {'name': '배관 동결·작업 안전 저하', 'actions': [
            '동결 우려 배관 보온재·열선 확인', '저온 환경 도장 품질 기준 재확인', '손발 감각 둔화 추락 주의']},
    ],
    # ── 해체·철거 ─────────────────────────────────
    ('해체·철거', '집중호우'): [
        {'name': '구조물 약화·붕괴', 'actions': [
            '강수로 약화된 잔존 구조물 안정성 점검', '해체 순서 재검토 및 임시 지지재 보강', '낙하물 방호 구역 재설정']},
    ],
    ('해체·철거', '장기가뭄·폭염'): [
        {'name': '분진 다량 발생·작업자 부담', 'actions': [
            '해체 전 살수 실시', '방진마스크 착용 의무화', '폭염 시 해체 작업 시간대 조정']},
    ],
    ('해체·철거', '태풍'): [
        {'name': '해체 구조물 붕괴 가속', 'actions': [
            '태풍 전 해체 예정 구조물 임시 보강', '출입 통제선 확대', '강풍 시 해체 작업 즉시 중단']},
    ],
    ('해체·철거', '한파'): [
        {'name': '구조물 응력 변화·취성', 'actions': [
            '동결-융해 반복으로 약화된 접합부 점검', '해체 속도 조절·단계별 안정 확인', '비상 대피로 결빙 제거']},
    ],
}

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
ria_path = f'{ROOT}/RIA_최종통합피쳐셋.csv'
if os.path.exists(ria_path):
    ria = pd.read_csv(ria_path, encoding='utf-8-sig')
    ria['사고일시'] = pd.to_datetime(ria['사고일시'])
    ria['date'] = ria['사고일시'].dt.normalize()
    ria['시간'] = ria['사고일시'].dt.hour
    ria['요일'] = ria['사고일시'].dt.dayofweek
    ria['시간bin'] = pd.cut(ria['시간'], bins=[-1,5,11,17,23], labels=[0,1,2,3]).astype(int)
    ria['공정율_수치'] = ria['공정율_수치'].fillna(50)
    ria['소규모현장'] = ria['소규모현장'].fillna(0).astype(float)
    val_cal = ria[ria['사고일시'] >= pd.Timestamp('2025-01-01')].iloc[::2].reset_index(drop=True)
else:
    print(f"[WARN] {ria_path} 없음: raw model score fallback 사용")
    val_cal = None

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

class IdentityCalibrator:
    def predict(self, values):
        return np.asarray(values, dtype=float)

calibrators, base_rates = {}, {}
if val_cal is not None and len(val_cal):
    cal_preds = predict_rows(val_cal)
    cal_labels = {
        '중대재해':   val_cal['중대재해'].values,
        '다중사상':   (val_cal['총재해자'] >= 2).astype(int).values,
        '외국인피해': (val_cal['외국인재해자'] >= 1).astype(int).values,
        '고령자피해': (val_cal['고령재해자'] >= 1).astype(int).values,
    }
    for h in cal_preds:
        ir = IsotonicRegression(out_of_bounds='clip'); ir.fit(cal_preds[h], cal_labels[h])
        calibrators[h] = ir
        base_rates[h] = cal_labels[h].mean()
else:
    calibrators = {h: IdentityCalibrator() for h in ['중대재해','다중사상','외국인피해','고령자피해']}
    # CSV 없을 때 raw sigmoid 기준점은 0.5 (캘리브레이션 없이 실제 발생률과 비교 불가)
    base_rates = {'중대재해': 0.5, '다중사상': 0.5, '외국인피해': 0.5, '고령자피해': 0.5}

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

def evaluate_env(env, cumf, today_date, scenario):
    # 1) 날씨 시나리오 기반 선언적 이슈 (정상이면 빈 리스트)
    cats = [dict(c, source='rule') for c in ENV_SCENARIO_ACTIONS.get((env['id'], scenario), [])]
    seen_actions = {a for c in cats for a in c['actions']}

    # 2) 모델 추론 (공종_proxy 사용)
    params = ENV_MODEL_PARAMS[env['id']]
    end = pd.Timestamp(today_date)
    row = pd.Series({
        'date': end, '시도구분': SIDO_PLACEHOLDER, '공사대분류': '건축',
        '공종(소분류)': env['공종_proxy'], '시간': params['시간'], '요일': end.dayofweek,
        '시간bin': int(pd.cut([params['시간']], bins=[-1,5,11,17,23], labels=[0,1,2,3])[0]),
        '공정율_수치': params['공정율'], '소규모현장': float(params['소규모']),
        '고위험공종': float(env['고위험']),
    })
    pred_raw = predict_rows(row.to_frame().T)
    pred = {h: float(calibrators[h].predict([pred_raw[h][0]])[0]) for h in pred_raw}

    # 3) 모델 카테고리 (Lift ≥ 임계값인 헤드만 추가)
    for head in ['중대재해', '다중사상', '외국인피해', '고령자피해']:
        v = pred[head]
        lift = v / max(base_rates[head], 1e-9)
        if lift < ALERT_LIFT_THRESHOLD:
            continue
        cat_name = (RISK_CATEGORIES.get((head, env['공종_proxy']))
                    or RISK_CATEGORIES.get((head, '*')))
        if not cat_name:
            continue
        new_actions = [a for a in ACTIONS_BY_CATEGORY.get(cat_name, []) if a not in seen_actions]
        for a in new_actions: seen_actions.add(a)
        if new_actions:
            cats.append({'name': cat_name, 'source': 'model', 'actions': new_actions})

    return {'env': env, 'pred': pred, 'cats': cats}

# ───── 체크리스트 메시지 ─────
def format_checklist(today_date, cumf, scenario, evals):
    today_str = pd.Timestamp(today_date).strftime('%Y-%m-%d (%a)')
    lines = []
    lines.append(f"🚨 *RIA Risk Instinct Alert — {today_str}*")
    lines.append("")
    lines.append(f"🌤️ *오늘의 기상* — _{scenario}_")
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
    lines.append(f"📋 *환경별 이슈 체크리스트*")
    lines.append("")

    triggered = [ev for ev in evals if ev['cats']]
    if not triggered:
        lines.append("   _오늘은 주요 위험 신호 없음 (정상 범위)_")
        return '\n'.join(lines)

    for ev in triggered:
        env = ev['env']
        tag_str = '   🟥 고위험' if env['고위험'] else ''
        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"▣ *{env['id']}*{tag_str}")
        lines.append("")
        for cat in ev['cats']:
            icon = '🔴' if cat['source'] == 'model' else '🟡'
            lines.append(f"   {icon} *{cat['name']}*")
            for a in cat['actions']:
                lines.append(f"      ☐ {a}")
            lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("_🔴 모델 신호  ·  🟡 날씨 룰  ·  체크 항목은 작업 전 안전미팅에서 확인_")
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
    scenario = detect_scenario(cumf)
    print(f"\n오늘 = {today_date.date()},  시나리오: {scenario},  평균 {cumf['평균기온']:.1f}°C,  Δtavg {cumf['Δtavg']:+.1f}°C")

    evals = [evaluate_env(e, cumf, today_date, scenario) for e in ENVIRONMENTS]
    msg = format_checklist(today_date, cumf, scenario, evals)
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
