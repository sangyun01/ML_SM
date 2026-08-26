"""Short/long-channel TCAD surrogate-model baseline and evaluation.

This module intentionally does not touch the Flask UI.  It reads the two sheets
from the integrated workbook, trains independent model bundles for Short and
Long devices, and reports validation metrics before the models are wired into
the application.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
CANDIDATE_SEED = 20260823
TEST_SIZE = 0.20
N_TREES = 300
LOW_DRAIN_BIAS = 0.05
HIGH_DRAIN_BIAS = 1.0
DEFAULT_WORKBOOK = (
    Path(__file__).resolve().parent
    / "data"
    / "training"
    / "tcad_training_master.xlsx"
)


@dataclass(frozen=True)
class DeviceConfig:
    sheet_name: str
    input_features: tuple[str, ...]
    threshold_column: str


DEVICE_CONFIGS = {
    "Short": DeviceConfig(
        sheet_name="Short",
        input_features=(
            "Lg",
            "LDD_Dose",
            "LDD_Energy",
            "SD_Dose",
            "SD_Energy",
            "anneal",
            "halo_dose",
            "halo_energy",
        ),
        threshold_column="Vth",
    ),
    "Long": DeviceConfig(
        sheet_name="Long",
        input_features=(
            "Lg",
            "LDD_Dose",
            "LDD_Energy",
            "SD_Dose",
            "SD_Energy",
        ),
        threshold_column="Vtgm",
    ),
}

MODEL_TARGETS = ("Vth", "SS", "Ion", "Ioff")
REQUIRED_BIASES = (LOW_DRAIN_BIAS, HIGH_DRAIN_BIAS)
TARGET_BIASES = {
    "Vth": LOW_DRAIN_BIAS,
    "SS": LOW_DRAIN_BIAS,
    "Ion": HIGH_DRAIN_BIAS,
    "Ioff": HIGH_DRAIN_BIAS,
}


def candidate_seed_for_gate_length(
    gate_length: float, base_seed: int = CANDIDATE_SEED
) -> int:
    """Return the reproducible candidate-pool seed used by UI and batch runs."""

    return int(base_seed) + int(round(float(gate_length) * 1_000_000))


def load_device_sheet(workbook_path: Path, config: DeviceConfig) -> pd.DataFrame:
    """Load and normalize one Excel sheet.

    The workbook has two blank leading rows and one blank leading column, so the
    real header is Excel row 3.  Numeric coercion also converts '-' cells used
    for failed extractions into NaN.
    """

    df = pd.read_excel(workbook_path, sheet_name=config.sheet_name, header=2)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if config.threshold_column != "Vth":
        df = df.rename(columns={config.threshold_column: "Vth"})

    required_columns = [
        *config.input_features,
        "Vd",
        *MODEL_TARGETS,
    ]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"{config.sheet_name} sheet is missing columns: {missing_columns}"
        )

    numeric_columns = list(required_columns)
    if "Valid" in df.columns:
        numeric_columns.append("Valid")

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[list(config.input_features) + ["Vd"]].isna().any().any():
        raise ValueError(
            f"{config.sheet_name} contains missing/non-numeric process inputs."
        )

    return df


def build_recipe_classification_table(
    df: pd.DataFrame, config: DeviceConfig
) -> pd.DataFrame:
    """Collapse two Vd rows into one recipe and derive a conservative Label.

    If the sheet contains an explicit Valid flag, it is authoritative.  This is
    used by the six-CSV Short dataset.  Older sheets without Valid retain the
    legacy rule requiring all four outputs at both drain biases.
    """

    inputs = list(config.input_features)
    work = df[df["Vd"].isin(REQUIRED_BIASES)].copy()

    if "Valid" in work.columns:
        recipe_table = (
            work.groupby(inputs, as_index=False, dropna=False)
            .agg(
                bias_count=("Vd", "nunique"),
                valid_value_count=("Valid", "count"),
                valid_min=("Valid", "min"),
            )
            .copy()
        )
        recipe_table["Label"] = (
            recipe_table["bias_count"].eq(len(REQUIRED_BIASES))
            & recipe_table["valid_value_count"].eq(len(REQUIRED_BIASES))
            & recipe_table["valid_min"].eq(1)
        ).astype(int)
        return recipe_table

    work["row_valid"] = work[list(MODEL_TARGETS)].notna().all(axis=1)

    recipe_table = (
        work.groupby(inputs, as_index=False, dropna=False)
        .agg(
            bias_count=("Vd", "nunique"),
            valid_bias_count=("row_valid", "sum"),
        )
        .copy()
    )
    recipe_table["Label"] = (
        recipe_table["bias_count"].eq(len(REQUIRED_BIASES))
        & recipe_table["valid_bias_count"].eq(len(REQUIRED_BIASES))
    ).astype(int)
    return recipe_table


def build_regression_table(
    df: pd.DataFrame, config: DeviceConfig
) -> pd.DataFrame:
    """Build one physics-oriented output row per process recipe.

    Vth and SS use the low-drain-voltage transfer curve, while Ion and Ioff use
    the high-drain-voltage curve.  This keeps the UI outputs explicit and avoids
    comparing low-Vd surrogate currents with high-Vd TCAD validation results.
    """

    inputs = list(config.input_features)
    low_columns = [*inputs, "Vth", "SS"]
    high_columns = [*inputs, "Ion", "Ioff"]
    if "Valid" in df.columns:
        low_columns.append("Valid")
        high_columns.append("Valid")

    low_bias = df[np.isclose(df["Vd"], LOW_DRAIN_BIAS)][low_columns].copy()
    high_bias = df[np.isclose(df["Vd"], HIGH_DRAIN_BIAS)][high_columns].copy()

    table = low_bias.merge(
        high_bias,
        on=inputs,
        how="inner",
        validate="one_to_one",
        suffixes=("_low", "_high"),
    )
    if "Valid_low" in table.columns:
        table = table[
            table["Valid_low"].eq(1) & table["Valid_high"].eq(1)
        ].copy()
    table = table.dropna(subset=list(MODEL_TARGETS))
    table = table[
        (table["Vth"] > 0)
        & (table["SS"] > 0)
        & (table["Ion"] > 0)
        & table["Ioff"].notna()
    ].copy()
    return table


def train_and_evaluate_classifier(
    recipe_table: pd.DataFrame, config: DeviceConfig
) -> RandomForestClassifier:
    features = list(config.input_features)
    X = recipe_table[features]
    y = recipe_table["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    model = RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    valid_probability = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])

    print("\n[Classifier: recipe valid/invalid]")
    print(f"recipes={len(recipe_table):,}, valid={int(y.sum()):,}, invalid={int((1-y).sum()):,}")
    print(f"accuracy={accuracy_score(y_test, prediction):.4f}")
    print(f"balanced_accuracy={balanced_accuracy_score(y_test, prediction):.4f}")
    print(f"roc_auc={roc_auc_score(y_test, valid_probability):.4f}")
    print("confusion_matrix rows=actual[invalid, valid], cols=predicted[invalid, valid]")
    print(matrix)
    print(classification_report(y_test, prediction, target_names=["invalid", "valid"], digits=4))

    # Refit the deployable model on all available recipes after evaluation.
    model.fit(X, y)
    return model


def _transform_target(target: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if target in {"Ion", "Ioff"}:
        # Ioff contains a few tiny negative numerical values in the Long sheet.
        # For this baseline, current magnitude is modeled in log10 space.
        return np.log10(np.clip(np.abs(values), 1e-30, None))
    if target == "SS":
        return np.log10(np.clip(values, 1e-30, None))
    return values


def _inverse_target(target: str, values: np.ndarray) -> np.ndarray:
    if target in {"SS", "Ion", "Ioff"}:
        return np.power(10.0, values)
    return values


def _regression_metrics(
    target: str, actual: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    metrics = {
        "r2": r2_score(actual, prediction),
        "mae": mean_absolute_error(actual, prediction),
        "rmse": np.sqrt(mean_squared_error(actual, prediction)),
    }
    if target in {"SS", "Ion", "Ioff"}:
        actual_log = _transform_target(target, actual)
        prediction_log = _transform_target(target, prediction)
        metrics["log_rmse_decades"] = np.sqrt(
            mean_squared_error(actual_log, prediction_log)
        )
    return metrics


def train_and_evaluate_regressors(
    regression_table: pd.DataFrame, config: DeviceConfig
) -> dict[str, RandomForestRegressor]:
    features = list(config.input_features)
    train_index, test_index = train_test_split(
        np.arange(len(regression_table)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X = regression_table[features].reset_index(drop=True)
    print(
        "\n[Regressors: Vth/SS @ Vd="
        f"{LOW_DRAIN_BIAS:g} V; Ion/Ioff @ Vd={HIGH_DRAIN_BIAS:g} V; "
        f"complete recipes={len(X):,}]"
    )
    models: dict[str, RandomForestRegressor] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for target in MODEL_TARGETS:
        y_raw = regression_table[target].to_numpy(dtype=float)
        y_model = _transform_target(target, y_raw)

        model = RandomForestRegressor(
            n_estimators=N_TREES,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        model.fit(X.iloc[train_index], y_model[train_index])
        prediction_model = model.predict(X.iloc[test_index])
        prediction_raw = _inverse_target(target, prediction_model)
        actual_raw = np.abs(y_raw[test_index]) if target == "Ioff" else y_raw[test_index]
        metrics = _regression_metrics(target, actual_raw, prediction_raw)
        test_predictions[target] = prediction_raw

        metric_text = " ".join(f"{name}={value:.5g}" for name, value in metrics.items())
        print(f"{target:>4}: {metric_text}")

        # Refit the deployable model on all valid rows after evaluation.
        model.fit(X, y_model)
        models[target] = model

    actual_onoff = (
        np.abs(regression_table["Ion"].to_numpy(dtype=float)[test_index])
        / np.clip(
            np.abs(regression_table["Ioff"].to_numpy(dtype=float)[test_index]),
            1e-30,
            None,
        )
    )
    predicted_onoff = test_predictions["Ion"] / np.clip(
        test_predictions["Ioff"], 1e-30, None
    )
    actual_log = np.log10(np.clip(actual_onoff, 1e-30, None))
    predicted_log = np.log10(np.clip(predicted_onoff, 1e-30, None))
    print(
        "OnOff (derived): "
        f"log_r2={r2_score(actual_log, predicted_log):.5g} "
        f"log_mae_decades={mean_absolute_error(actual_log, predicted_log):.5g} "
        "log_rmse_decades="
        f"{np.sqrt(mean_squared_error(actual_log, predicted_log)):.5g}"
    )

    return models


def evaluate_unseen_gate_lengths(
    regression_table: pd.DataFrame, config: DeviceConfig
) -> None:
    """Measure extrapolation when one complete Lg choice is absent in training."""

    features = list(config.input_features)
    print("\n[Leave-one-Lg-out: Vth and SS]")
    for held_lg in sorted(regression_table["Lg"].unique()):
        train = regression_table[~np.isclose(regression_table["Lg"], held_lg)]
        test = regression_table[np.isclose(regression_table["Lg"], held_lg)]
        result_parts = [f"held_Lg={held_lg:g}", f"test_rows={len(test)}"]

        for target in ("Vth", "SS"):
            model = RandomForestRegressor(
                n_estimators=N_TREES,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            y_train = _transform_target(target, train[target].to_numpy(dtype=float))
            model.fit(train[features], y_train)
            prediction = _inverse_target(target, model.predict(test[features]))
            actual = test[target].to_numpy(dtype=float)
            result_parts.append(f"{target}_r2={r2_score(actual, prediction):.4f}")
            result_parts.append(
                f"{target}_rmse={np.sqrt(mean_squared_error(actual, prediction)):.4g}"
            )

        print(" | ".join(result_parts))


def train_device_bundle(
    workbook_path: Path, config: DeviceConfig
) -> dict[str, object]:
    print("\n" + "=" * 72)
    print(f"{config.sheet_name.upper()}-CHANNEL MODEL")
    print(f"features={list(config.input_features)}")

    df = load_device_sheet(workbook_path, config)
    recipe_table = build_recipe_classification_table(df, config)
    regression_table = build_regression_table(df, config)

    classifier = train_and_evaluate_classifier(recipe_table, config)
    regressors = train_and_evaluate_regressors(regression_table, config)
    evaluate_unseen_gate_lengths(regression_table, config)

    return {
        "config": config,
        "classifier": classifier,
        "regressors": regressors,
        "gate_length_choices": sorted(regression_table["Lg"].unique().tolist()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=DEFAULT_WORKBOOK,
        help=f"Integrated Excel dataset (default: {DEFAULT_WORKBOOK})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.workbook.exists():
        raise FileNotFoundError(f"Workbook not found: {args.workbook}")

    bundles = {
        name: train_device_bundle(args.workbook, config)
        for name, config in DEVICE_CONFIGS.items()
    }

    print("\n" + "=" * 72)
    print("TRAINED GATE-LENGTH CHOICES")
    for name, bundle in bundles.items():
        print(f"{name}: {bundle['gate_length_choices']}")


if __name__ == "__main__":
    main()
