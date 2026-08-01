import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

"""
num_samples = 1000
# 5개 입력 변수 (Lg, LDD Dose & Energy, SD Dose & Energy)
X_train = np.random.rand(num_samples, 5)
# 4개 출력 결과 (Vth, Id, SS, gm)
Y_train = np.random.rand(num_samples, 4)
"""

df = pd.read_csv("simulation_data.csv")

input_features = ["Lg", "LDD_Dose", "LDD_Energy", "SD_Dose", "SD_Energy"]
output_targets = ["Vth", "Id", "SS", "gm"]

X_train = df[input_features].values
Y_train = df[output_targets].values

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)


def calcul_parameters(
    model, param_bounds, target_specs, num_candidates=100000, top_k=3
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
        print("error")
        return None

    best_indices = np.argsort(valid_scores)[:top_k]

    return valid_candidates[best_indices], valid_predictions[best_indices]


# range of the param condition value
parameter_bounds = [
    (10, 50),  # Lg
    (0.5, 2.0),  # LDD Dose
    (1e15, 1e17),  # LDD Energy
    (5, 20),  # SD Dose
    (0.1, 1.0),  # SD Energy
]

# 꼭 min, max, target 3개를 다 넣을 필요는 없습니다. 필요한 조건만 적으면 됩니다.
user_specs = {
    "Vth": {
        "min": 0.3,
        "max": 0.5,
        "target": 0.4,
    },  # Vth는 0.3~0.5 사이이되, 0.4에 가장 가까울 것
    "Id": {"target": 1.0},  # Id는 1.0에 가장 가까울 것
    "SS": {"max": 65, "target": 60},  # SS는 65 이하이되, 60에 가까울수록 좋음
    # gm은 특별한 조건이 없다면 생략 가능
}

best_inputs, best_outputs = calcul_parameters(model, parameter_bounds, user_specs)

for rank in range(3):
    print(f"Top {rank+1}")
    print(f"recommend var param : {np.round(best_inputs[rank], 4)}")
    print(f"Predict Result (Vth, Id, SS, gm): {np.round(best_outputs[rank], 4)}\n")
