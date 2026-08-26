# TCAD 기반 NMOS 공정 추천 시스템

## 폴더 구조

- `ML_SM.py`: Short/Long Random Forest 모델 학습 및 80:20 forward 평가
- `app.py`: 공정 추천 UI 서버
- `batch_validation.py`: Short inverse-test 후보 및 TCAD 입력 생성
- `batch_validation_long.py`: Long inverse-test 후보 및 TCAD 입력 생성
- `evaluate_tcad_results.py`: ML 예측값과 실제 TCAD 결과 비교
- `data/training/tcad_training_master.xlsx`: 현재 모델이 사용하는 통합 훈련 데이터
- `data/training/short_sweep_recipes_2000.csv`: 추가 Short TCAD sweep 설계 목록
- `data/inverse_test/inverse_test_results.xlsx`: Short/Long practical inverse-test 통합 결과
- `data/inverse_test/batches/`: inverse-test 목표, 추천 레시피, ML 예측값
- `data/inverse_test/tcad_inputs/`: TCAD에 입력한 고유 추천 레시피
- `data/inverse_test/tcad_results/`: TCAD에서 추출한 원본 CSV
- `data/inverse_test/metrics/`: ML 예측과 TCAD 결과의 상세 평가 JSON
- `reports/TCAD_ML_성능평가_요약.pdf`: 최종 성능 요약 보고서

## 설치 및 실행

```bash
python3 -m pip install flask numpy pandas matplotlib shap scikit-learn openpyxl
python3 ML_SM.py
python3 app.py
```

UI는 기본적으로 `http://127.0.0.1:5000/`에서 열린다.

기존 TCAD 결과를 다시 평가하려면 다음을 실행한다.

```bash
python3 evaluate_tcad_results.py --output-dir data/inverse_test/metrics_regenerated
```
