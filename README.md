# SCOUT: Surrogate-guided Candidate Screening for TCAD-verified Process Design

> Team-대전야호 (Daejeon-Yaho) · Category: Device / Process

## Introduction

**SCOUT** is a TCAD–Machine Learning **inverse-design framework** that recommends 2-D planar NMOS process recipes from a target device specification, and re-verifies every recommendation in Sentaurus TCAD.

Physically accurate device characteristics require Sentaurus TCAD simulation, but the process design space grows combinatorially with the number of process variables, so exhaustively simulating every recipe is infeasible.
Designers also work in the *reverse* direction: they start from a desired spec (Vth, SS, Ion, Ioff) and need the process conditions that satisfy it, whereas TCAD only runs process → characteristic.

**SCOUT** closes this gap with a **forward surrogate model** and a **large-scale candidate search**.
- Speed: a Random Forest surrogate evaluates **500,000 candidate recipes** in seconds, so the expensive TCAD run is reserved for the final Top-1 candidate.
- Reliability: a **Valid/Invalid classifier** filters out recipes whose TCAD result cannot be trusted *before* regression, and every recommendation is **re-simulated in TCAD** to measure the real error.

## Key Features

### 1. Two-Stage Reliability Gating (Classifier → Regressor)

TCAD results contain both low-performance devices and results that cannot be trusted at all (solver failure, extraction failure, missing output).

- Valid ≠ good device: a recipe with huge SS or high leakage is still **Valid** as long as TCAD computation and parameter extraction completed normally.
- The Random Forest **classifier** is trained on the full Valid + Invalid set; the four Random Forest **regressors** (Vth, SS, Ion, Ioff) are trained only on Valid recipes with all four outputs present.
- At recommendation time, candidates below `P(valid) ≥ 0.5` are dropped before regression, so extraction failures are never learned as a fake "0" performance value.
- Classifier balanced accuracy: **Short 98.35% / Long 99.19%** (ROC-AUC 99.93% / 99.92%).

### 2. Forward-Surrogate Inverse Search (Y → X without direct regression)

Direct Y → X regression is ill-posed: many different recipes produce a similar spec (one-to-many), so an averaged recipe may not be physically valid.

- Instead, SCOUT generates **500,000 candidate recipes** inside the trained min–max range for the selected discrete gate length (Dose log-uniform, Energy / anneal uniform, fixed random seed for reproducibility).
- Each candidate is passed through the classifier, then the four regressors, then scored: relative-error² for Vth / SS and log-scale error² for Ion / Ioff.
- The lowest-score candidate is **Top 1**; the UI shows **Top 1–3** for downstream selection.
- This is a bounded global search inside the learnable design space, not a proof of global optimum — but every candidate is directly comparable, constraint-aware, and comes with predicted characteristics.

### 3. Channel-Aware Model Bundle + SHAP Interpretability

Below 100 nm, short-channel effects (SCE) break the assumptions of the baseline ~250 nm process, so the Short and Long regimes are modeled separately.

- Long (180 / 360 / 720 / 1000 nm): 5 input variables (Gate Length, LDD Dose/Energy, S/D Dose/Energy).
- Short (65 / 90 nm): 8 input variables — the 5 above plus **Halo Dose/Energy** (channel-barrier control) and **Annealing** (thermal-budget sensitivity); gate oxide thinned to 2.0 / 2.5 nm.
- SHAP is computed for every recommendation: Gate Length and S/D implant conditions dominate all four outputs, and their directions match known device physics (Vth roll-off, drive-current channel-length dependence, SCE / DIBL leakage), supporting that the model learned physically meaningful relations.

## System Architecture

The overall system operates via a Host–Surrogate–TCAD loop:

1.  Host PC (Web UI, Flask): the user selects Short / Long and a discrete gate length, then enters the Min–Target–Max window for Vth, SS, Ion, Ioff.
2.  Candidate Generator: builds 500,000 reproducible recipes inside the training min–max range.
3.  SCOUT Surrogate (scikit-learn):
    - Classifier: `P(valid) ≥ 0.5` gate.
    - Regressor bank: 4 × Random Forest → predicted Vth / SS / Ion / Ioff (SS, Ion, Ioff learned in log₁₀ space).
    - Scoring: relative-error (Vth, SS) + log-error (Ion, Ioff) → Top 1–3.
4.  TCAD Re-verification: the Top-1 recipe is re-simulated in Sentaurus TCAD; predicted vs. actual error and target-window satisfaction are measured, and the verified data is fed back into the training set.

| Component | Specification |
|:---:|:---|
| TCAD | Synopsys Sentaurus Process / Sentaurus Device |
| Surrogate | Random Forest (scikit-learn), 300 trees per model |
| Inputs | Long: 5 variables / Short: 8 variables |
| Outputs | Vth, SS, Ion, Ioff (SS · Ion · Ioff learned in log₁₀) |
| Candidate pool | 500,000 per gate length, `P(valid) ≥ 0.5` gate |
| Interface | Flask web application |

## Performance Evaluation

Two evaluations: a **Forward test** (X → Y, 80 / 20 hold-out, `random_state = 42`) for the surrogate itself, and an **Inverse test** (target → recommendation → TCAD) for the end-to-end goal.
Training data: **Short 5,072 recipes** (3,317 Valid) · **Long 2,750 recipes** (2,332 Valid).

### Forward prediction (hold-out)

| Model | Vth | SS | Ion | Ioff |
|:---|:---:|:---:|:---:|:---:|
| Long | R² 0.986 / MAE 6.20 mV | R² 0.994 / MAE 2.66 mV/dec | R² 0.994 | R² 0.794 |
| Short | R² 0.960 / MAE 28.0 mV | R² 0.831 | R² 0.987 | R² 0.996 (large log-RMSE) |

### Inverse recommendation (re-verified in TCAD)

300 target requests → **274 unique Top-1 recipes** → **100% completed TCAD computation and extraction**.

| Model | Vth MAE | SS MAE | Ion (×) | Ioff (×) | Target-window satisfaction (Vth / SS / Ion / Ioff) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Long | 10.2 mV | 1.47 mV/dec | ×1.04 | ×1.92 | 98.5% / 100% / 100% / 76% |
| Short | 85.1 mV (median 43.9) | 23.2 mV/dec (median 7.1) | ×1.33 | ×61 (median ×19.8) | 53% / 84% / 90% / 17% |

- Long: stable across all four outputs — usable as a **recommendation tool** inside the trained range.
- Short: on-current is reproduced well, but at the SCE boundary small Vth / SS errors are exponentially amplified into large Ioff multiplicative error — better used as an **exploration / diagnosis tool** to narrow down candidates that still need TCAD verification.
- The system output is defined as *"a candidate recipe worth re-verifying in TCAD,"* not *"the correct recipe."*

## Demo

*In the demo environment, a Flask web interface visualizes the full flow: target-spec input → Top 1–3 recommended recipes with predicted characteristics and per-variable SHAP contribution.*

| Model region | Input | Result |
|:---|:---|:---|
| Long-channel | Target Vth / SS / Ion / Ioff window at Lg = 0.18–1.0 µm | Top 1–3 recipes; TCAD-verified Vth within ~10 mV, Ion within ~1.04× |
| Short-channel | Target Vth / SS / Ion / Ioff window at Lg = 65 / 90 nm | Top 1–3 recipes flagged for additional TCAD verification (SCE / Ioff sensitivity) |

## Team-대전야호 (Daejeon-Yaho)

- 김두형 (Duhyeong Kim)
  -
- 김상윤 (Sangyun Kim)
  -
- 김정필 (Jeongpil Kim)
  -
- 박상헌 (Sangheon Park)
  -
- 이지연 (Jiyeon Lee)
  -

## Repository

```bash
python3 -m pip install flask numpy pandas matplotlib shap scikit-learn openpyxl
python3 ML_SM.py     # train Short/Long Random Forest models + 80:20 forward evaluation
python3 app.py       # launch the recommendation UI (http://127.0.0.1:5000/)
```

| Path | Description |
|:---|:---|
| `ML_SM.py` | Short / Long Random Forest training and forward hold-out evaluation |
| `app.py` | Flask recommendation UI server |
| `batch_validation.py` / `batch_validation_long.py` | Inverse-test candidate generation and TCAD input files |
| `evaluate_tcad_results.py` | Compare ML predictions against actual TCAD results |
| `data/training/tcad_training_master.xlsx` | Integrated training dataset (Short / Long sheets) |
| `data/inverse_test/` | Inverse-test targets, recommended recipes, TCAD inputs/results, metrics |
| `reports/` | Final report, figures, and the 8-minute presentation deck |

---

# Korean Version:  SCOUT: TCAD–ML 기반 2차원 Planar NMOS 역설계 및 공정 레시피 추천 시스템

> Team-대전야호 · 출품 분야: 소자 / 공정

## Introduction

**SCOUT**는 목표 소자 특성으로부터 2차원 planar NMOS 공정 레시피를 추천하고, 모든 추천 결과를 Sentaurus TCAD로 재검증하는 **TCAD–Machine Learning 역설계 프레임워크**입니다.

물리적으로 정확한 소자 특성을 확인하려면 Sentaurus TCAD 시뮬레이션이 필요하지만,
공정변수가 늘수록 가능한 조합 수가 급격히 증가하여 모든 레시피를 직접 시뮬레이션하는 것은 불가능합니다.
또한 설계자는 보통 *반대 방향*으로 접근합니다. 원하는 스펙(Vth, SS, Ion, Ioff)에서 출발해 이를 만족하는 공정조건을 필요로 하지만,
TCAD는 공정조건 → 특성 방향으로만 동작합니다.

**SCOUT**는 **forward surrogate model**과 **대규모 후보 탐색**으로 이 간극을 메웁니다.
- Speed: Random Forest surrogate가 **후보 레시피 50만 개**를 초 단위로 평가하여, 고비용 TCAD 연산은 최종 Top 1 후보에만 사용합니다.
- Reliability: **Valid/Invalid 분류기**가 신뢰할 수 없는 TCAD 결과를 회귀 학습 *이전에* 걸러내고, 모든 추천 결과는 **TCAD로 재시뮬레이션**하여 실제 오차를 측정합니다.

## Key Features

### 1. 2단계 신뢰성 게이팅 (Classifier → Regressor)

TCAD 결과에는 성능이 낮은 소자와, 결과 자체를 신뢰할 수 없는 경우(solver failure, extraction 실패, 출력 누락)가 함께 존재합니다.

- Valid ≠ 좋은 소자: SS가 크거나 누설이 높아도, TCAD 계산과 parameter extraction이 정상 완료되면 **Valid**로 분류합니다.
- Random Forest **분류기**는 Valid + Invalid 전체로 학습하고, 4개의 Random Forest **회귀기**(Vth, SS, Ion, Ioff)는 4개 출력이 모두 존재하는 Valid 레시피만으로 학습합니다.
- 추천 단계에서는 `P(valid) ≥ 0.5` 미만 후보를 회귀 이전에 제외하여, extraction 실패를 가짜 "0" 성능값으로 학습하지 않습니다.
- 분류기 balanced accuracy: **Short 98.35% / Long 99.19%** (ROC-AUC 99.93% / 99.92%).

### 2. Forward-Surrogate 역탐색 (직접 회귀 없이 Y → X)

직접 Y → X 회귀는 하나의 정답을 갖지 않습니다. 서로 다른 레시피가 비슷한 스펙을 만들 수 있어(one-to-many), 평균 레시피는 물리적으로 유효하지 않을 수 있습니다.

- 대신 SCOUT는 선택한 이산 Gate Length에서 학습 min–max 범위 안에 **후보 레시피 50만 개**를 생성합니다(Dose는 log-uniform, Energy·anneal은 uniform, 재현을 위해 난수 seed 고정).
- 각 후보를 분류기 → 4개 회귀기 순서로 통과시키고, Vth·SS는 상대오차², Ion·Ioff는 log scale 오차²로 score를 계산합니다.
- score가 가장 작은 후보가 **Top 1**이며, UI에는 후속 선택을 위해 **Top 1–3**을 표시합니다.
- 학습 가능한 설계영역 안에서 수행하는 제한된 전역 탐색이며 수학적 전역 최적해를 보장하지는 않지만, 모든 후보를 직접 비교할 수 있고 제약조건을 반영하며 예상 특성을 함께 제시합니다.

### 3. 채널별 모델 bundle + SHAP 해석

100 nm 이하에서는 SCE(Short Channel Effect)가 기존 약 250 nm급 공정의 가정을 깨뜨리므로, Short와 Long을 분리하여 모델링합니다.

- Long (180 / 360 / 720 / 1000 nm): 입력변수 5개 (Gate Length, LDD Dose/Energy, S/D Dose/Energy).
- Short (65 / 90 nm): 입력변수 8개 — 위 5개에 **Halo Dose/Energy**(채널 barrier 제어)와 **Annealing**(thermal budget 민감성)을 추가하고, gate oxide를 2.0 / 2.5 nm로 박막화.
- 모든 추천에 대해 SHAP을 계산합니다. Gate Length와 S/D implant 조건이 네 출력 모두를 지배하고, 그 방향이 알려진 소자물리(Vth roll-off, drive current의 채널길이 의존, SCE·DIBL 누설)와 일치하여 모델이 물리적으로 의미 있는 관계를 학습했음을 뒷받침합니다.

## System Architecture

전체 시스템은 Host–Surrogate–TCAD 루프로 동작합니다.

1.  Host PC (Web UI, Flask): 사용자가 Short / Long과 이산 Gate Length를 선택하고, Vth·SS·Ion·Ioff의 Min–Target–Max 범위를 입력합니다.
2.  Candidate Generator: 학습 min–max 범위 안에 재현 가능한 레시피 50만 개를 생성합니다.
3.  SCOUT Surrogate (scikit-learn):
    - Classifier: `P(valid) ≥ 0.5` 게이트.
    - Regressor bank: 4 × Random Forest → 예상 Vth / SS / Ion / Ioff (SS·Ion·Ioff는 log₁₀ 공간에서 학습).
    - Scoring: 상대오차(Vth, SS) + log 오차(Ion, Ioff) → Top 1–3.
4.  TCAD 재검증: Top 1 레시피를 Sentaurus TCAD로 재시뮬레이션하여 예측 대비 실제 오차와 목표범위 충족 여부를 측정하고, 검증된 데이터는 다시 학습 데이터로 재편입됩니다.

| Component | Specification |
|:---:|:---|
| TCAD | Synopsys Sentaurus Process / Sentaurus Device |
| Surrogate | Random Forest (scikit-learn), 모델당 트리 300개 |
| Inputs | Long: 5개 변수 / Short: 8개 변수 |
| Outputs | Vth, SS, Ion, Ioff (SS · Ion · Ioff는 log₁₀ 학습) |
| Candidate pool | Gate Length별 50만 개, `P(valid) ≥ 0.5` 게이트 |
| Interface | Flask web application |

## Performance Evaluation

두 가지 평가를 수행했습니다. surrogate 자체를 보는 **Forward test**(X → Y, 80 / 20 hold-out, `random_state = 42`)와, 최종 목적을 보는 **Inverse test**(목표 → 추천 → TCAD)입니다.
학습 데이터: **Short 5,072 레시피**(Valid 3,317) · **Long 2,750 레시피**(Valid 2,332).

### Forward prediction (hold-out)

| Model | Vth | SS | Ion | Ioff |
|:---|:---:|:---:|:---:|:---:|
| Long | R² 0.986 / MAE 6.20 mV | R² 0.994 / MAE 2.66 mV/dec | R² 0.994 | R² 0.794 |
| Short | R² 0.960 / MAE 28.0 mV | R² 0.831 | R² 0.987 | R² 0.996 (log-RMSE 큼) |

### Inverse recommendation (TCAD 재검증)

목표 스펙 요청 300건 → **고유 Top 1 레시피 274개** → **모두 TCAD 계산 및 extraction 정상 완료**.

| Model | Vth MAE | SS MAE | Ion (×) | Ioff (×) | 목표범위 만족률 (Vth / SS / Ion / Ioff) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Long | 10.2 mV | 1.47 mV/dec | ×1.04 | ×1.92 | 98.5% / 100% / 100% / 76% |
| Short | 85.1 mV (중앙값 43.9) | 23.2 mV/dec (중앙값 7.1) | ×1.33 | ×61 (중앙값 ×19.8) | 53% / 84% / 90% / 17% |

- Long: 네 출력 모두 안정적 — 학습 범위 안에서 **추천 도구**로 활용 가능.
- Short: on-current는 잘 재현되지만, SCE 경계에서 작은 Vth / SS 오차가 Ioff의 큰 배수오차로 지수적으로 증폭 — 추가 TCAD 검증이 필요한 후보를 좁히는 **탐색 / 진단 도구**로 활용.
- 시스템의 출력은 *"정답 레시피"*가 아니라 *"TCAD로 재검증할 가치가 높은 후보 레시피"*로 정의합니다.

## Demo

*시연 환경에서는 Flask 웹 인터페이스로 전체 흐름을 시각화합니다. 목표 스펙 입력 → 예상 특성과 변수별 SHAP 기여도가 포함된 Top 1–3 추천 레시피.*

| Model region | Input | Result |
|:---|:---|:---|
| Long-channel | Lg = 0.18–1.0 µm 에서 Vth / SS / Ion / Ioff 목표 범위 | Top 1–3 레시피; TCAD 재검증 Vth 오차 ~10 mV, Ion ~1.04× |
| Short-channel | Lg = 65 / 90 nm 에서 Vth / SS / Ion / Ioff 목표 범위 | Top 1–3 레시피 (SCE / Ioff 민감성으로 추가 TCAD 검증 대상) |

## Team-대전야호 (Daejeon-Yaho)

- 김두형 (Duhyeong Kim)
  -
- 김상윤 (Sangyun Kim)
  -
- 김정필 (Jeongpil Kim)
  -
- 박상헌 (Sangheon Park)
  -
- 이지연 (Jiyeon Lee)
  -

## Repository

```bash
python3 -m pip install flask numpy pandas matplotlib shap scikit-learn openpyxl
python3 ML_SM.py     # Short/Long Random Forest 학습 + 80:20 forward 평가
python3 app.py       # 추천 UI 서버 (http://127.0.0.1:5000/)
```

| Path | Description |
|:---|:---|
| `ML_SM.py` | Short / Long Random Forest 학습 및 forward hold-out 평가 |
| `app.py` | Flask 공정 추천 UI 서버 |
| `batch_validation.py` / `batch_validation_long.py` | Inverse-test 후보 및 TCAD 입력 생성 |
| `evaluate_tcad_results.py` | ML 예측값과 실제 TCAD 결과 비교 |
| `data/training/tcad_training_master.xlsx` | 통합 훈련 데이터 (Short / Long 시트) |
| `data/inverse_test/` | inverse-test 목표·추천 레시피·TCAD 입출력·평가 지표 |
| `reports/` | 최종 보고서, 그림, 8분 발표 자료 |

---
*This project was submitted to POLARIS SIF 2026.*
