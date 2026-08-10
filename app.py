import matplotlib

matplotlib.use("Agg")

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import time
import shap
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# 전역 변수로 두 개의 모델과 SHAP 이미지를 저장할 딕셔너리 생성
models = {}
shap_plots = {}

input_features = ["Lg", "LDD_Dose", "LDD_Energy", "SD_Dose", "SD_Energy"]
output_targets = ["Vth", "Id", "SS", "gm"]
target_names = ["Vth", "Id", "SS", "gm"]


# =====================================================================
# 1. 모델 학습 및 SHAP 그래프 생성 함수
# =====================================================================
def train_model_and_shap(csv_filename, material_name):
    print(f"\n=== 🚀 {material_name} 데이터 로딩 및 모델 학습 시작 ===")
    df = pd.read_csv(csv_filename)
    df = df[df["Vd"] == 0.05]

    X = df[input_features].values
    Y = df[output_targets].values

    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X, Y)
    print(f"✅ {material_name} 모델 학습 완료!")

    print(f"=== {material_name} SHAP 대시보드 생성 중 ===")
    explainer = shap.TreeExplainer(model)
    X_sample = shap.sample(X, 500) if len(X) > 500 else X
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, target in enumerate(target_names):
        mean_abs_shap = np.abs(shap_values[:, :, i]).mean(axis=0)
        if mean_abs_shap.sum() == 0:
            weights_pct = np.zeros_like(mean_abs_shap)
        else:
            weights_pct = (mean_abs_shap / mean_abs_shap.sum()) * 100

        feature_names_with_pct = [
            f"{name} ({pct:.1f}%)" for name, pct in zip(input_features, weights_pct)
        ]

        plt.sca(axes[i])
        shap.summary_plot(
            shap_values[:, :, i],
            X_sample,
            feature_names=feature_names_with_pct,
            show=False,
            plot_size=None,
        )
        axes[i].set_title(
            f"{material_name} SHAP Importance for {target}", fontsize=14, pad=15
        )

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    print(f"✅ {material_name} SHAP 그래프 생성 완료!")

    return model, plot_url


# 서버 기동 시 두 가지 물질에 대한 모델을 모두 학습시킵니다.
models["SiO2"], shap_plots["SiO2"] = train_model_and_shap("simulation_data.csv", "SiO2")
models["HKMG"], shap_plots["HKMG"] = train_model_and_shap(
    "simulation_data_HKMG.csv", "HKMG"
)
print("\n🎉 모든 모델 학습 완료! 웹 서버가 준비되었습니다.")


# =====================================================================
# 2. 최적 파라미터 탐색 함수 (사용할 모델을 인자로 받음)
# =====================================================================
def calcul_parameters(
    target_model, param_bounds, target_specs, num_candidates=500000, top_k=3
):
    candidates = np.zeros((num_candidates, 5))
    for i in range(5):
        min_val, max_val = param_bounds[i]
        candidates[:, i] = np.random.uniform(min_val, max_val, num_candidates)

    predictions = target_model.predict(candidates)
    scores = np.zeros(num_candidates)
    valid_mask = np.ones(num_candidates, dtype=bool)

    for i, target_name in enumerate(target_names):
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


# =====================================================================
# 3. 웹 라우팅
# =====================================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analysis")
def analysis():
    # URL 쿼리스트링으로 넘어온 material 값 확인 (기본값 SiO2)
    material = request.args.get("material", "SiO2")
    # 선택된 물질에 맞는 SHAP 그래프를 전달
    selected_plot = shap_plots.get(material, shap_plots["SiO2"])

    return render_template("analysis.html", shap_plot=selected_plot, material=material)


@app.route("/search", methods=["POST"])
def search():
    user_specs = request.json

    # 프론트엔드에서 보내준 'oxide_material' 값 꺼내기 (기본값 SiO2)
    material = user_specs.get("oxide_material", "SiO2")

    # 선택된 물질에 맞는 모델 불러오기
    selected_model = models.get(material, models["SiO2"])

    parameter_bounds = [(0.05, 1), (1e11, 1e15), (10, 70), (1e14, 1e18), (10, 40)]

    start_time = time.time()
    result = calcul_parameters(selected_model, parameter_bounds, user_specs)
    end_time = time.time()

    if result is None:
        return jsonify(
            {"status": "fail", "message": "조건을 만족하는 후보를 찾지 못했습니다."}
        )

    best_inputs, best_outputs = result

    results_list = []
    for rank in range(len(best_inputs)):
        results_list.append(
            {
                "rank": rank + 1,
                "params": np.round(best_inputs[rank], 4).tolist(),
                "vth": round(best_outputs[rank][0], 4),
                "id": f"{best_outputs[rank][1]:.4e}",
                "ss": round(best_outputs[rank][2], 4),
                "gm": f"{best_outputs[rank][3]:.4e}",
            }
        )

    return jsonify(
        {
            "status": "success",
            "time": round(end_time - start_time, 4),
            "data": results_list,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
