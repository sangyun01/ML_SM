"""Match TCAD exports to batch recommendations and calculate validation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from CODE.batch_validation import classify_existing_case


TARGETS = ("Vth", "SS", "Ion", "Ioff")
PRACTICAL_LIMITS = {
    "Vth": (0.3, 1.2),
    "SS": (None, 200.0),
    "Ion": (1e-4, None),
    "Ioff": (None, 1e-6),
}


def load_eval(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=2)
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    for column in frame.columns:
        if column not in {"Vtgm", "SS"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def feature_key(row: pd.Series | dict[str, object], features: list[str]) -> tuple[object, ...]:
    key: list[object] = []
    for feature in features:
        value = float(row[feature])
        if feature in {"LDD_Dose", "SD_Dose", "halo_dose"}:
            key.append(format(value, ".6e"))
        elif feature == "Lg":
            key.append(round(value, 8))
        else:
            key.append(round(value, 10))
    return tuple(key)


def collapse_recipes(frame: pd.DataFrame, features: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[int]] = {}
    for index, row in frame.iterrows():
        grouped.setdefault(feature_key(row, features), []).append(index)

    recipes: list[dict[str, object]] = []
    for key, indices in grouped.items():
        rows = frame.loc[indices]
        low = rows[np.isclose(rows["Vd"], 0.05)]
        high = rows[np.isclose(rows["Vd"], 1.0)]
        record: dict[str, object] = {
            "key": key,
            "source_rows": [int(i + 4) for i in indices],
            **{feature: float(rows.iloc[0][feature]) for feature in features},
            "bias_count": int(rows["Vd"].nunique()),
        }
        if len(low) != 1 or len(high) != 1:
            record.update({"status": "MissingBias", "Vth": None, "SS": None, "Ion": None, "Ioff": None})
            recipes.append(record)
            continue

        low_row = low.iloc[0]
        high_row = high.iloc[0]
        curve_valid = True
        if "CurveValid" in rows.columns:
            curve_valid = bool(
                pd.to_numeric(rows["CurveValid"], errors="coerce").eq(1).all()
            )
        vth = pd.to_numeric(pd.Series([low_row["Vtgm"]]), errors="coerce").iloc[0]
        ss = pd.to_numeric(pd.Series([low_row["SS"]]), errors="coerce").iloc[0]
        ion = pd.to_numeric(pd.Series([high_row["Ion"]]), errors="coerce").iloc[0]
        ioff = pd.to_numeric(pd.Series([high_row["Ioff"]]), errors="coerce").iloc[0]
        numeric_valid = all(pd.notna(value) and float(value) > 0 for value in (vth, ss, ion, ioff))
        status = "Valid" if curve_valid and numeric_valid else "InvalidCurve"
        record.update(
            {
                "status": status,
                "CurveValidLow": int(low_row["CurveValid"]) if "CurveValid" in rows.columns else None,
                "CurveValidHigh": int(high_row["CurveValid"]) if "CurveValid" in rows.columns else None,
                "Vth": float(vth) if pd.notna(vth) else None,
                "SS": float(ss) if pd.notna(ss) else None,
                "Ion": float(ion) if pd.notna(ion) else None,
                "Ioff": float(abs(ioff)) if pd.notna(ioff) else None,
            }
        )
        recipes.append(record)
    return recipes


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def summarize_cases(cases: list[dict[str, object]]) -> dict[str, object]:
    valid = [case for case in cases if case["TCAD_Status"] == "Valid"]
    metrics: dict[str, object] = {}
    for target in TARGETS:
        actual = np.asarray([case[f"Actual_{target}"] for case in valid], dtype=float)
        predicted = np.asarray([case[f"Pred_{target}"] for case in valid], dtype=float)
        if target in {"Ion", "Ioff"}:
            actual_log = np.log10(np.clip(actual, 1e-30, None))
            predicted_log = np.log10(np.clip(predicted, 1e-30, None))
            item = regression_metrics(actual_log, predicted_log)
            item = {f"log_{key}": value for key, value in item.items()}
            item["mae_A"] = float(mean_absolute_error(actual, predicted))
        else:
            item = regression_metrics(actual, predicted)
        metrics[target] = item

    actual_onoff = np.asarray([case["Actual_OnOff"] for case in valid], dtype=float)
    predicted_onoff = np.asarray([case["Pred_OnOff"] for case in valid], dtype=float)
    onoff_metrics = regression_metrics(
        np.log10(np.clip(actual_onoff, 1e-30, None)),
        np.log10(np.clip(predicted_onoff, 1e-30, None)),
    )
    metrics["OnOff"] = {f"log_{key}": value for key, value in onoff_metrics.items()}

    per_metric_pass = {
        target: sum(bool(case[f"{target}_TargetPass"]) for case in cases) / len(cases)
        for target in TARGETS
    }
    practical_metric_pass = {
        target: sum(bool(case[f"{target}_PracticalPass"]) for case in cases) / len(cases)
        for target in PRACTICAL_LIMITS
    }
    return {
        "case_count": len(cases),
        "valid_case_count": len(valid),
        "invalid_case_count": len(cases) - len(valid),
        "actual_valid_rate": len(valid) / len(cases),
        "all_target_pass_count": sum(bool(case["All_TargetPass"]) for case in cases),
        "all_target_pass_rate": sum(bool(case["All_TargetPass"]) for case in cases) / len(cases),
        "metric_pass_rate": per_metric_pass,
        "actual_practical_pass_count": sum(bool(case["All_PracticalPass"]) for case in cases),
        "actual_practical_pass_rate": sum(bool(case["All_PracticalPass"]) for case in cases) / len(cases),
        "practical_metric_pass_rate": practical_metric_pass,
        "prediction_metrics": metrics,
    }


def evaluate_device(eval_path: Path, batch_path: Path) -> dict[str, object]:
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    features = list(batch["metadata"].get("features") or [
        key for key in ("Lg", "LDD_Dose", "LDD_Energy", "SD_Dose", "SD_Energy", "anneal", "halo_dose", "halo_energy")
        if key in batch["unique_recipes"][0]
    ])
    source = load_eval(eval_path)
    actual_recipes = collapse_recipes(source, features)
    actual_by_key = {tuple(recipe["key"]): recipe for recipe in actual_recipes}

    matched: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    expected_keys: set[tuple[object, ...]] = set()
    for recipe in batch["unique_recipes"]:
        key = feature_key(recipe, features)
        expected_keys.add(key)
        actual = actual_by_key.get(key)
        if actual is None:
            missing.append(recipe)
            continue
        matched.append(
            {
                "RecipeID": recipe["RecipeID"],
                "CaseIDs": recipe["CaseIDs"],
                **{feature: recipe[feature] for feature in features},
                "TCAD_Status": actual["status"],
                "TCAD_Vth": actual["Vth"],
                "TCAD_SS": actual["SS"],
                "TCAD_Ion": actual["Ion"],
                "TCAD_Ioff": actual["Ioff"],
                "SourceRows": actual["source_rows"],
            }
        )

    unmatched = [
        {key: value for key, value in recipe.items() if key != "key"}
        for recipe in actual_recipes
        if tuple(recipe["key"]) not in expected_keys
    ]
    matched_by_id = {recipe["RecipeID"]: recipe for recipe in matched}

    case_results: list[dict[str, object]] = []
    for case in batch["cases"]:
        actual = matched_by_id.get(case["RecipeID"])
        status = actual["TCAD_Status"] if actual else "MissingRecipe"
        result = dict(case)
        result["TestGroup"], result["TestGroupReason"] = classify_existing_case(case)
        result["TCAD_Status"] = status
        for target in TARGETS:
            value = actual[f"TCAD_{target}"] if actual and status == "Valid" else None
            result[f"Actual_{target}"] = value
        if status == "Valid":
            result["Actual_OnOff"] = result["Actual_Ion"] / max(result["Actual_Ioff"], 1e-30)
            result["ErrPred_Vth_mV"] = abs(result["Actual_Vth"] - result["Pred_Vth"]) * 1000
            result["ErrPred_SS_mVdec"] = abs(result["Actual_SS"] - result["Pred_SS"])
            result["ErrPred_Ion_dec"] = abs(np.log10(result["Actual_Ion"]) - np.log10(result["Pred_Ion"]))
            result["ErrPred_Ioff_dec"] = abs(np.log10(result["Actual_Ioff"]) - np.log10(result["Pred_Ioff"]))
            result["ErrPred_OnOff_dec"] = abs(np.log10(result["Actual_OnOff"]) - np.log10(result["Pred_OnOff"]))
            for target in TARGETS:
                result[f"{target}_TargetPass"] = bool(
                    result[f"{target}_Min"] <= result[f"Actual_{target}"] <= result[f"{target}_Max"]
                )
            result["All_TargetPass"] = all(result[f"{target}_TargetPass"] for target in TARGETS)
            actual_values = {
                target: result[f"Actual_{target}"] for target in PRACTICAL_LIMITS
            }
            for target, (lower, upper) in PRACTICAL_LIMITS.items():
                value = actual_values[target]
                result[f"{target}_PracticalPass"] = bool(
                    (lower is None or value >= lower)
                    and (upper is None or value <= upper)
                )
            result["All_PracticalPass"] = all(
                result[f"{target}_PracticalPass"] for target in PRACTICAL_LIMITS
            )
            result["PredictionErrorScore"] = (
                ((result["Actual_Vth"] - result["Pred_Vth"]) / max(abs(result["Pred_Vth"]), 1e-12)) ** 2
                + ((result["Actual_SS"] - result["Pred_SS"]) / max(abs(result["Pred_SS"]), 1e-12)) ** 2
                + result["ErrPred_Ion_dec"] ** 2
                + result["ErrPred_Ioff_dec"] ** 2
            )
        else:
            result["Actual_OnOff"] = None
            for key in (
                "ErrPred_Vth_mV", "ErrPred_SS_mVdec", "ErrPred_Ion_dec",
                "ErrPred_Ioff_dec", "ErrPred_OnOff_dec", "PredictionErrorScore",
            ):
                result[key] = None
            for target in TARGETS:
                result[f"{target}_TargetPass"] = False
            result["All_TargetPass"] = False
            for target in PRACTICAL_LIMITS:
                result[f"{target}_PracticalPass"] = False
            result["All_PracticalPass"] = False
        case_results.append(result)

    summary = summarize_cases(case_results)
    by_lg = {
        str(lg): summarize_cases([case for case in case_results if np.isclose(case["Lg"], lg)])
        for lg in sorted({float(case["Lg"]) for case in case_results})
    }
    by_test_group = {
        group: summarize_cases(
            [case for case in case_results if case["TestGroup"] == group]
        )
        for group in ("Practical", "Boundary")
        if any(case["TestGroup"] == group for case in case_results)
    }
    worst = sorted(
        [case for case in case_results if case["PredictionErrorScore"] is not None],
        key=lambda case: case["PredictionErrorScore"],
        reverse=True,
    )[:10]
    return {
        "metadata": {
            "eval_source": str(eval_path.resolve()),
            "batch_source": str(batch_path.resolve()),
            "features": features,
            "source_row_count": len(source),
            "source_recipe_count": len(actual_recipes),
            "matched_recipe_count": len(matched),
            "missing_recipe_count": len(missing),
            "unmatched_source_recipe_count": len(unmatched),
        },
        "summary": summary,
        "by_lg": by_lg,
        "by_test_group": by_test_group,
        "matched_recipes": matched,
        "missing_recipes": missing,
        "unmatched_source_recipes": unmatched,
        "cases": case_results,
        "worst_cases": worst,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--short-eval",
        type=Path,
        default=Path("data/inverse_test/tcad_results/short.csv"),
    )
    parser.add_argument(
        "--long-eval",
        type=Path,
        default=Path("data/inverse_test/tcad_results/long.csv"),
    )
    parser.add_argument(
        "--short-batch",
        type=Path,
        default=Path("data/inverse_test/batches/short.json"),
    )
    parser.add_argument(
        "--long-batch",
        type=Path,
        default=Path("data/inverse_test/batches/long.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    short = evaluate_device(args.short_eval, args.short_batch)
    long = evaluate_device(args.long_eval, args.long_batch)
    for name, result in (("short", short), ("long", long)):
        (args.output_dir / f"evaluation_{name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({"short": short["metadata"], "long": long["metadata"]}, indent=2))
    print(json.dumps({"short": short["summary"], "long": long["summary"]}, indent=2))


if __name__ == "__main__":
    main()
