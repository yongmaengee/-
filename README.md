# RIA — Risk Instinct Alert

기후 이탈량과 시설안전 사고 데이터의 상관관계를 학습해, 실시간 기상 조건에서  
발생 가능한 위험 이슈와 현장 체크리스트를 자동 생성하는 시스템.

## 구조

```
기상청 API (KMA)
    │
    ▼
누적 피처 계산 (강수·기온·풍속 7일 롤링)
    │
    ▼
Multi-task Encoder-Decoder Transformer (Cross-Attention)
시계열 기상 윈도우 → 환경 컨텍스트와 Cross-Attention → 이슈별 위험 확률
    │
    ▼
이슈 활성화 점수 → 환경별 체크리스트 생성
    │
    ▼
리뷰 콘솔 (웹 UI) → Slack 전송
```

- **오프라인**: 사고 이력 40,714건으로 Encoder-Decoder Transformer 학습 → `cache/model/ria_model.pt`
- **온라인**: 실시간 기상 → 누적 피처 → 트랜스포머 추론 → 이슈 점수 → 체크리스트
- **시각화**: W 히트맵은 도메인 지식 기반 정적 행렬 (실제 추론과 별도)

## 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn pyarrow torch

python ria_review_server.py
```

실행 시 KMA API 키와 Slack Webhook URL을 입력하거나 빈 값으로 넘기면  
캐시/dry-run 모드로 동작한다.

```
KMA_KEY 입력 (빈 값이면 캐시/mock 사용):
Slack Webhook URL 입력 (빈 값이면 전송 비활성):

RIA review server: http://127.0.0.1:8765
```

환경변수로 미리 설정하면 입력 프롬프트를 건너뛴다.

```bash
export KMA_KEY=...
export SLACK_WEBHOOK=https://hooks.slack.com/services/...
```

## 웹 UI

| 경로 | 설명 |
|---|---|
| `http://127.0.0.1:8765` | 리뷰 콘솔 — 라이브 데이터, Slack 전송 |
| `http://127.0.0.1:8765/legacy` | 정적 데모 — 시나리오 선택 방식 |

리뷰 콘솔 구성:
- **시나리오 감지**: 집중호우·태풍·한파·폭염 등 자동 분류
- **온라인 추론**: 피처 이탈 벡터 / W 히트맵 / 이슈 활성화 점수
- **OUTPUT**: 환경유형별 체크리스트 (위험·주의 태그)
- **Slack 전송**: 웹훅으로 현장 채널에 직접 발송

## 주요 파일

| 파일 | 설명 |
|---|---|
| `ria_review_server.py` | 로컬 리뷰 서버 + 웹 UI (메인 진입점) |
| `18_run_alert_v3.py` | 추론 엔진 — KMA 연동, 피처 계산, 체크리스트 생성 |
| `RIA_webapp.html` | 정적 데모 페이지 |
| `13_train_model.py` | Transformer 모델 학습 → `cache/model/ria_model.pt` |
| `14_inference_demo.py` | 추론 파이프라인 단독 실행 |
| `15_what_if.py` | What-if 시나리오 시뮬레이션 |

## 데이터

- `RIA_최종통합피쳐셋.csv` — 40,714건, 54컬럼, 2019-07 ~ 2025-12 (gitignore, 별도 공유)
- `national_weather_avg_2019_2026.csv` — 전국 평균 기상 원본 (gitignore)
- `cache/weather_daily.parquet` — KMA 응답 캐시

## 분석 스크립트

| 파일 | 설명 |
|---|---|
| `01_data_overview.py` | 데이터셋 구조·결측·분포 점검 |
| `02_weather_seasonal.py` | 월별·계절별 날씨 패턴과 사고율 |
| `03_cumulative_weather.py` | 누적 강수·풍속 피처 분석 |
| `04_accident_type_sensitivity.py` | 사고유형별 날씨 민감도 (z-score) |
| `05_summary_insights.md` | 데이터 품질 진단 및 모델 설계 권고 |
