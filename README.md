# Risk Instinct Alert (RIA)

기후 변화와 시설안전 사고의 상관관계를 분석하여, 실시간 기후 조건에서 발생할 만한
위험 이슈와 체크리스트를 제공하는 시스템.

## 개요

- **오프라인 학습**: 기후 피처 × 사고원인 → Feature-Issue 가중치 행렬
- **온라인 추론**: 실시간 기후 → 이탈량 누적 → 가중치 행렬과 곱 → 이슈 체크리스트

핵심 아이디어와 모델 설계는 [`아이디어.md`](아이디어.md), 데이터 분석 결과는
[`05_summary_insights.md`](05_summary_insights.md)에 정리되어 있다.

## 데이터

- `RIA_최종통합피쳐셋.csv` — 40,714건, 54컬럼, 2019-07 ~ 2025-12 (gitignore 처리, 별도 공유)
- `national_weather_avg_2019_2026 2.csv` — 전국 평균 기상 데이터 (gitignore)

## 분석 스크립트

| 파일 | 설명 |
|---|---|
| `01_data_overview.py` | 데이터셋 구조 및 결측·분포 점검 |
| `02_weather_seasonal.py` | 월별·계절별 날씨 패턴과 사고율 |
| `03_cumulative_weather.py` | 누적 강수·풍속 등 누적 피처 분석 |
| `04_accident_type_sensitivity.py` | 사고유형별 날씨 민감도 (z-score 기반) |

## 산출 문서

- `05_summary_insights.md` — 데이터 품질 진단, 라벨 누수 위험, 모델 설계 권고 종합 리포트
- `아이디어.md` — RIA 컨셉, Layer 1~3 라벨 설계, Attention 구조

## 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn
```
