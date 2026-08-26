"""Flask UI for short/long-channel TCAD inverse recipe recommendation."""

from __future__ import annotations

import base64
import io
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tcad-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ML_SM import (
    CANDIDATE_SEED,
    DEFAULT_WORKBOOK,
    DEVICE_CONFIGS,
    MODEL_TARGETS,
    _inverse_target,
    _transform_target,
    build_recipe_classification_table,
    build_regression_table,
    candidate_seed_for_gate_length,
    load_device_sheet,
)


RANDOM_STATE = 42
N_TREES = 300
SHAP_SAMPLE_SIZE = 200
NUM_CANDIDATES = int(os.getenv("TCAD_NUM_CANDIDATES", "500000"))
CANDIDATE_POOL_SEED = int(os.getenv("TCAD_CANDIDATE_SEED", str(CANDIDATE_SEED)))
TOP_K = 3
VALID_PROBABILITY_THRESHOLD = 0.5
WORKBOOK_PATH = Path(os.getenv("TCAD_WORKBOOK", str(DEFAULT_WORKBOOK)))

FEATURE_LABELS = {
    "Lg": "Gate Length",
    "LDD_Dose": "LDD Dose",
    "LDD_Energy": "LDD Energy",
    "SD_Dose": "SD Dose",
    "SD_Energy": "SD Energy",
    "anneal": "Anneal Time",
    "halo_dose": "Halo Dose",
    "halo_energy": "Halo Energy",
}

FEATURE_UNITS = {
    "Lg": "μm",
    "LDD_Dose": "cm⁻²",
    "LDD_Energy": "keV",
    "SD_Dose": "cm⁻²",
    "SD_Energy": "keV",
    "anneal": "s",
    "halo_dose": "cm⁻²",
    "halo_energy": "keV",
}

TARGET_LABELS = {
    "Vth": "Vth (Vd=0.05 V)",
    "SS": "SS (Vd=0.05 V)",
    "Ion": "Ion (Vd=1.0 V)",
    "Ioff": "|Ioff| (Vd=1.0 V)",
    "OnOff": "Ion/|Ioff| (Vd=1.0 V)",
}

TARGET_UNITS = {
    "Vth": "V",
    "SS": "mV/dec",
    "Ion": "A",
    "Ioff": "A",
    "OnOff": "ratio",
}

UI_TARGETS = (*MODEL_TARGETS, "OnOff")


app = Flask(__name__)

models: dict[str, dict[str, object]] = {}
shap_plots: dict[str, dict[str, str]] = {}
shap_data_store: dict[str, dict[str, dict[str, float]]] = {}
client_device_configs: dict[str, dict[str, object]] = {}
target_defaults: dict[str, dict[str, dict[str, float]]] = {}


def _numeric_quantile(values: pd.Series, probability: float, log_scale: bool) -> float:
    numeric = values.to_numpy(dtype=float)
    if log_scale:
        numeric = np.log10(np.clip(np.abs(numeric), 1e-30, None))
        return float(10 ** np.quantile(numeric, probability))
    return float(np.quantile(numeric, probability))


def _build_target_defaults(regression_table: pd.DataFrame) -> dict[str, dict[str, float]]:
    defaults: dict[str, dict[str, float]] = {}
    target_values = {
        target: regression_table[target] for target in MODEL_TARGETS
    }
    target_values["OnOff"] = (
        regression_table["Ion"].abs()
        / regression_table["Ioff"].abs().clip(lower=1e-30)
    )
    for target in UI_TARGETS:
        log_scale = target in {"Ion", "Ioff", "OnOff"}
        defaults[target] = {
            "min": _numeric_quantile(target_values[target], 0.10, log_scale),
            "target": _numeric_quantile(target_values[target], 0.50, log_scale),
            "max": _numeric_quantile(target_values[target], 0.90, log_scale),
        }
    return defaults


def _build_client_config(
    device_name: str,
    regression_table: pd.DataFrame,
    input_features: tuple[str, ...],
) -> dict[str, object]:
    features = []
    for feature in input_features:
        values = regression_table[feature].to_numpy(dtype=float)
        unique_values = sorted(np.unique(values).tolist())
        features.append(
            {
                "name": feature,
                "label": FEATURE_LABELS[feature],
                "unit": FEATURE_UNITS[feature],
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "choices": unique_values if feature == "Lg" else None,
                "log_scale": feature in {"LDD_Dose", "SD_Dose", "halo_dose"},
            }
        )
    return {
        "name": device_name,
        "features": features,
        "gate_lengths": sorted(regression_table["Lg"].unique().tolist()),
    }


def _make_shap_outputs(
    device_name: str,
    regressors: dict[str, RandomForestRegressor],
    X: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    sampled = shap.sample(
        X,
        min(SHAP_SAMPLE_SIZE, len(X)),
        random_state=RANDOM_STATE,
    )
    plots: dict[str, str] = {}
    importance_data: dict[str, dict[str, float]] = {}

    for target, model in regressors.items():
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sampled)
        mean_abs = np.abs(values).mean(axis=0)
        total = mean_abs.sum()
        percentages = np.zeros_like(mean_abs) if total == 0 else mean_abs / total * 100
        importance_data[target] = {
            feature: float(value)
            for feature, value in zip(X.columns, percentages)
        }

        display_names = [
            f"{FEATURE_LABELS[feature]} ({percentage:.1f}%)"
            for feature, percentage in zip(X.columns, percentages)
        ]
        plt.figure(figsize=(8, 5.5))
        shap.summary_plot(
            values,
            sampled,
            feature_names=display_names,
            show=False,
            plot_size=None,
        )
        model_suffix = " (log model)" if target in {"SS", "Ion", "Ioff"} else ""
        plt.title(
            f"{device_name} SHAP Importance for {TARGET_LABELS[target]}{model_suffix}",
            fontsize=13,
            pad=14,
        )
        plt.tight_layout()
        image = io.BytesIO()
        plt.savefig(image, format="png", bbox_inches="tight", dpi=120)
        image.seek(0)
        plots[target] = base64.b64encode(image.getvalue()).decode("ascii")
        plt.close()

    return plots, importance_data


def train_device_model(device_name: str) -> None:
    config = DEVICE_CONFIGS[device_name]
    print(f"\n=== {device_name} model training ===")
    source = load_device_sheet(WORKBOOK_PATH, config)
    recipe_table = build_recipe_classification_table(source, config)
    regression_table = build_regression_table(source, config)
    features = list(config.input_features)

    classifier = RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    classifier.fit(recipe_table[features], recipe_table["Label"])

    regressors: dict[str, RandomForestRegressor] = {}
    X = regression_table[features].reset_index(drop=True)
    for target in MODEL_TARGETS:
        regressor = RandomForestRegressor(
            n_estimators=N_TREES,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        regressor.fit(
            X,
            _transform_target(target, regression_table[target].to_numpy(dtype=float)),
        )
        regressors[target] = regressor

    models[device_name] = {
        "config": config,
        "classifier": classifier,
        "regressors": regressors,
        "training_table": regression_table,
    }
    client_device_configs[device_name] = _build_client_config(
        device_name,
        regression_table,
        config.input_features,
    )
    target_defaults[device_name] = _build_target_defaults(regression_table)
    plots, importance = _make_shap_outputs(device_name, regressors, X)
    shap_plots[device_name] = plots
    shap_data_store[device_name] = importance
    print(
        f"{device_name}: recipes={len(recipe_table)}, "
        f"valid regression rows={len(regression_table)}, "
        f"Lg={client_device_configs[device_name]['gate_lengths']}"
    )


def _generate_candidates(bundle: dict[str, object], gate_length: float) -> pd.DataFrame:
    config = bundle["config"]
    training_table = bundle["training_table"]
    rng = np.random.default_rng(
        candidate_seed_for_gate_length(gate_length, CANDIDATE_POOL_SEED)
    )
    candidate_data: dict[str, np.ndarray] = {}

    for feature in config.input_features:
        values = training_table[feature].to_numpy(dtype=float)
        if feature == "Lg":
            candidate_data[feature] = np.full(NUM_CANDIDATES, gate_length)
        elif feature in {"LDD_Dose", "SD_Dose", "halo_dose"}:
            candidate_data[feature] = np.power(
                10.0,
                rng.uniform(np.log10(values.min()), np.log10(values.max()), NUM_CANDIDATES),
            )
        else:
            candidate_data[feature] = rng.uniform(
                values.min(), values.max(), NUM_CANDIDATES
            )

    return pd.DataFrame(candidate_data, columns=list(config.input_features))


def _allowed_gate_length(device_name: str, gate_length: float) -> bool:
    allowed = client_device_configs[device_name]["gate_lengths"]
    return any(np.isclose(gate_length, choice) for choice in allowed)


def _predict_valid_candidates(
    bundle: dict[str, object], candidates: pd.DataFrame
) -> pd.DataFrame:
    classifier = bundle["classifier"]
    probabilities = classifier.predict_proba(candidates)[:, 1]
    valid_mask = probabilities >= VALID_PROBABILITY_THRESHOLD
    valid = candidates.loc[valid_mask].copy()
    valid["valid_probability"] = probabilities[valid_mask]

    feature_columns = list(bundle["config"].input_features)
    for target, regressor in bundle["regressors"].items():
        transformed = regressor.predict(valid[feature_columns])
        valid[target] = _inverse_target(target, transformed)
    valid["OnOff"] = valid["Ion"] / np.clip(valid["Ioff"], 1e-30, None)
    return valid


def _target_error(target: str, predictions: np.ndarray, target_value: float) -> np.ndarray:
    if target in {"Ion", "Ioff", "OnOff"}:
        predicted_log = np.log10(np.clip(predictions, 1e-30, None))
        target_log = np.log10(max(target_value, 1e-30))
        return (predicted_log - target_log) ** 2
    denominator = max(abs(target_value), 1e-12)
    return ((predictions - target_value) / denominator) ** 2


def recommend(
    device_name: str,
    gate_length: float,
    target_specs: dict[str, dict[str, float]],
) -> pd.DataFrame | None:
    bundle = models[device_name]
    candidates = _generate_candidates(bundle, gate_length)
    valid = _predict_valid_candidates(bundle, candidates)
    if valid.empty:
        return None

    mask = np.ones(len(valid), dtype=bool)
    score = np.zeros(len(valid), dtype=float)
    for target in UI_TARGETS:
        if target not in target_specs:
            continue
        spec = target_specs[target]
        predictions = valid[target].to_numpy(dtype=float)
        mask &= predictions >= spec["min"]
        mask &= predictions <= spec["max"]
        score += _target_error(target, predictions, spec["target"])

    valid["score"] = score
    filtered = valid.loc[mask]
    if filtered.empty:
        return None
    return filtered.nsmallest(TOP_K, "score")


def _parse_target_specs(payload: dict[str, object]) -> dict[str, dict[str, float]]:
    specs: dict[str, dict[str, float]] = {}
    for target in UI_TARGETS:
        if target not in payload:
            continue
        raw = payload[target]
        spec = {key: float(raw[key]) for key in ("min", "target", "max")}
        if not spec["min"] <= spec["target"] <= spec["max"]:
            raise ValueError(f"{target}: min ≤ target ≤ max 조건이 필요합니다.")
        if target in {"Ion", "Ioff", "OnOff"} and spec["min"] <= 0:
            raise ValueError(f"{target}: 로그 점수 계산을 위해 양수여야 합니다.")
        specs[target] = spec
    if not specs:
        raise ValueError("최소 한 개의 목표 스펙을 선택하세요.")
    return specs


def _serialize_row(row: pd.Series, bundle: dict[str, object], rank: int) -> dict[str, object]:
    parameters = {
        feature: float(row[feature]) for feature in bundle["config"].input_features
    }
    predictions = {target: float(row[target]) for target in UI_TARGETS}
    return {
        "rank": rank,
        "params": parameters,
        "predictions": predictions,
        "valid_probability": float(row["valid_probability"]),
        "score": float(row["score"]),
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        device_configs=client_device_configs,
        target_defaults=target_defaults,
        target_labels=TARGET_LABELS,
        target_units=TARGET_UNITS,
    )


@app.route("/analysis")
def analysis():
    device_name = request.args.get("device", "Short")
    if device_name not in models:
        device_name = "Short"
    return render_template(
        "analysis.html",
        shap_plots=shap_plots[device_name],
        shap_data=shap_data_store[device_name],
        device=device_name,
        feature_labels=FEATURE_LABELS,
        target_labels=TARGET_LABELS,
    )


@app.route("/search", methods=["POST"])
def search():
    try:
        payload = request.get_json(force=True)
        device_name = payload.get("device_type", "Short")
        if device_name not in models:
            raise ValueError("지원하지 않는 device type입니다.")
        gate_length = float(payload["gate_length"])
        if not _allowed_gate_length(device_name, gate_length):
            raise ValueError("Gate Length는 학습 데이터의 선택값만 사용할 수 있습니다.")
        specs = _parse_target_specs(payload)

        started = time.perf_counter()
        result = recommend(device_name, gate_length, specs)
        elapsed = time.perf_counter() - started
        if result is None:
            return jsonify(
                {
                    "status": "fail",
                    "message": "조건을 만족하는 valid 후보를 찾지 못했습니다.",
                }
            )

        bundle = models[device_name]
        records = [
            _serialize_row(row, bundle, rank)
            for rank, (_, row) in enumerate(result.iterrows(), start=1)
        ]
        return jsonify(
            {
                "status": "success",
                "time": round(elapsed, 4),
                "device_type": device_name,
                "gate_length": gate_length,
                "candidate_count": NUM_CANDIDATES,
                "candidate_seed": candidate_seed_for_gate_length(
                    gate_length, CANDIDATE_POOL_SEED
                ),
                "data": records,
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"status": "fail", "message": str(error)}), 400


@app.route("/predict_single", methods=["POST"])
def predict_single():
    try:
        payload = request.get_json(force=True)
        device_name = payload.get("device_type", "Short")
        if device_name not in models:
            raise ValueError("지원하지 않는 device type입니다.")
        bundle = models[device_name]
        features = list(bundle["config"].input_features)
        parameters = payload["params"]
        row = pd.DataFrame(
            [{feature: float(parameters[feature]) for feature in features}],
            columns=features,
        )
        gate_length = float(row.iloc[0]["Lg"])
        if not _allowed_gate_length(device_name, gate_length):
            raise ValueError("Gate Length는 학습 데이터의 선택값만 사용할 수 있습니다.")

        probability = float(bundle["classifier"].predict_proba(row)[0, 1])
        if probability < VALID_PROBABILITY_THRESHOLD:
            return jsonify(
                {
                    "status": "fail",
                    "message": "이 공정조건은 TCAD 결과가 invalid일 가능성이 높습니다.",
                    "valid_probability": probability,
                }
            )

        predictions = {}
        for target, regressor in bundle["regressors"].items():
            transformed = regressor.predict(row)[0]
            predictions[target] = float(_inverse_target(target, transformed))
        predictions["OnOff"] = predictions["Ion"] / max(predictions["Ioff"], 1e-30)
        return jsonify(
            {
                "status": "success",
                "predictions": predictions,
                "valid_probability": probability,
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"status": "fail", "message": str(error)}), 400


if not WORKBOOK_PATH.exists():
    raise FileNotFoundError(f"Integrated workbook not found: {WORKBOOK_PATH}")

for device in DEVICE_CONFIGS:
    train_device_model(device)

print("\nAll Short/Long models are ready.")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
