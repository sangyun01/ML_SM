import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터 불러오기 및 확인
df = pd.read_csv("simulation_data.csv")

print("=== 실제 CSV 데이터 범위 확인 ===")
print(f"Vth 범위: {df['Vth'].min():.4f} ~ {df['Vth'].max():.4f}")
print(f"Id 범위: {df['Id'].min():.4e} ~ {df['Id'].max():.4e}")
print(f"SS 범위: {df['SS'].min():.4f} ~ {df['SS'].max():.4f}")
print(f"gm 범위: {df['gm'].min():.4e} ~ {df['gm'].max():.4e}")
print("===============================\n")

input_features = ["Lg", "LDD_Dose", "LDD_Energy", "SD_Dose", "SD_Energy"]
output_targets = ["Vth", "Id", "SS", "gm"]

X_train = df[input_features].values
Y_train = df[output_targets].values

# 2. 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# =====================================================================
# 3. 추가된 부분: SHAP (설명 가능한 AI) 2x2 대시보드 시각화
# =====================================================================
print("=== 🧠 SHAP 분석: 인공지능 모델의 소자 물리 해석 중 ===")
explainer = shap.TreeExplainer(model)

X_sample = shap.sample(X_train, 500) if len(X_train) > 500 else X_train
shap_values = explainer.shap_values(X_sample)

target_names = ["Vth", "Id", "SS", "gm"]

# 2x2 그리드 창 생성 (가로 16, 세로 12 사이즈 넉넉하게 지정)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()  # 2차원 배열을 1차원으로 펴서 반복문 돌리기 쉽게 만듭니다.

for i, target in enumerate(target_names):
    # 1. 터미널 출력용 % 수치 계산
    mean_abs_shap = np.abs(shap_values[:, :, i]).mean(axis=0)
    weights_pct = (mean_abs_shap / mean_abs_shap.sum()) * 100

    # 2. 기존 변수명 옆에 % 수치를 합성 (예: "Lg (45.8%)")
    feature_names_with_pct = [
        f"{name} ({pct:.1f}%)" for name, pct in zip(input_features, weights_pct)
    ]

    # 3. 2x2 그리드 중 현재 위치(axes[i])에 그래프 그리기
    plt.sca(axes[i])
    shap.summary_plot(
        shap_values[:, :, i],
        X_sample,
        feature_names=feature_names_with_pct,
        show=False,
        plot_size=None,  # 핵심: 2x2 레이아웃이 깨지지 않도록 SHAP의 자동 크기 조절을 차단
    )
    axes[i].set_title(f"SHAP Importance for {target}", fontsize=14, pad=15)

# 그래프 간의 간격을 자동으로 맞춰줌
plt.tight_layout()
print(
    "📊 4개의 타겟에 대한 SHAP 분석 결과를 2x2 창에 모두 띄웁니다. (창을 닫으면 최적 파라미터 탐색이 시작됩니다)"
)
plt.show()
print("=========================================================\n")


# 4. 무작위 탐색(Random Search)을 통한 최적 파라미터 도출
def calcul_parameters(
    model, param_bounds, target_specs, num_candidates=500000, top_k=3
):
    candidates = np.zeros((num_candidates, 5))
    for i in range(5):
        min_val, max_val = param_bounds[i]
        candidates[:, i] = np.random.uniform(min_val, max_val, num_candidates)

    predictions = model.predict(candidates)

    scores = np.zeros(num_candidates)
    valid_mask = np.ones(num_candidates, dtype=bool)

    for i, target_name in enumerate(["Vth", "Id", "SS", "gm"]):
        if target_name in target_specs:
            spec = target_specs[target_name]
            pred_vals = predictions[:, i]

            if "min" in spec:
                valid_mask &= pred_vals >= spec["min"]
            if "max" in spec:
                valid_mask &= pred_vals <= spec["max"]

            if "target" in spec:
                scores += ((pred_vals - spec["target"]) / spec["target"]) ** 2

    valid_candidates = candidates[valid_mask]
    valid_scores = scores[valid_mask]
    valid_predictions = predictions[valid_mask]

    if len(valid_candidates) == 0:
        return None

    best_indices = np.argsort(valid_scores)[:top_k]

    return valid_candidates[best_indices], valid_predictions[best_indices]


# 사용자 파라미터 탐색 범위 설정
parameter_bounds = [
    (0.05, 1),  # Lg (gate Length)
    (1e11, 1e15),  # LDD Dose
    (10, 70),  # LDD Energy
    (1e14, 1e18),  # SD Dose
    (10, 40),  # SD Energy
]

# 타겟 스펙 설정
user_specs = {
    "Vth": {
        "min": 0.6,
        "max": 1.0,
        "target": 0.8,
    },
    # "Id": {
    #     "min": 1e-6,
    #     "max": 1e-4,
    #     "target": 1e-5,
    # },
    "SS": {
        "min": 60,
        "max": 100,
        "target": 70,
    },
    # "gm": {
    #     "min": 1e-5,
    #     "max": 1e-3,
    #     "target": 1e-4,
    # },
}

print("=== 🔍 최적 파라미터 조합 탐색 시작 ===")
result = calcul_parameters(model, parameter_bounds, user_specs)

<<<<<<< HEAD
for rank in range(3):
    print(f"Top {rank+1}")
    print(f"recommend var param : {np.round(best_inputs[rank], 4)}")
    print(f"Predict Result (Vth, Id, SS, gm): {np.round(best_outputs[rank], 4)}\n")

=======
if result is None:
    print("현재 설정된 조건을 만족하는 후보를 찾지 못했습니다.")
    print("이유 1: user_specs 조건이 아직 너무 빡빡함")
    print("이유 2: 학습 데이터(CSV)에 해당 결과값을 낼 수 있는 데이터가 아예 없음")
else:
    best_inputs, best_outputs = result

    num_results = len(best_inputs)
    for rank in range(num_results):
        print(f"Top {rank+1}")
        print(f"recommend var param : {np.round(best_inputs[rank], 4)}")
        print(f"Predict Result (Vth, Id, SS, gm): {np.round(best_outputs[rank], 4)}\n")
>>>>>>> ace06b9deb56cbacec765a2351d9b44e2ac9c73d
