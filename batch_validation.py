"""Create reproducible Short-channel inverse-design validation cases.

The script mirrors the deployed Flask app's pooled Short model, candidate
distribution, validity filter, constraints, and target score.  It generates one
candidate pool for each selected discrete gate length, then reuses that pool for
many validation targets so that 100 recommendations do not require 100 browser
requests or 100 independent 500k-point predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ML_SM import (
    CANDIDATE_SEED,
    DEFAULT_WORKBOOK,
    DEVICE_CONFIGS,
    MODEL_TARGETS,
    N_TREES,
    RANDOM_STATE,
    REQUIRED_BIASES,
    _inverse_target,
    _transform_target,
    build_recipe_classification_table,
    build_regression_table,
    candidate_seed_for_gate_length,
    load_device_sheet,
)


FEATURES = list(DEVICE_CONFIGS["Short"].input_features)
LOG_FEATURES = {"LDD_Dose", "SD_Dose", "halo_dose"}
LOG_SCORE_TARGETS = {"Ion", "Ioff", "OnOff"}
VALID_PROBABILITY_THRESHOLD = 0.5
DEFAULT_GATE_LENGTHS = (0.065, 0.09)

# Engineering-oriented target window used only to separate validation suites.
# These limits do not delete training rows or redefine TCAD numerical validity.
PRACTICAL_TARGET_LIMITS = {
    "Vth_min": 0.3,
    "Vth_max": 1.2,
    "SS_max": 200.0,
    "Ion_min": 1e-4,
    "Ioff_max": 1e-6,
}


def practical_output_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return targets suitable for the practical-use validation suite."""

    limits = PRACTICAL_TARGET_LIMITS
    return (
        frame["Vth"].between(limits["Vth_min"], limits["Vth_max"]).to_numpy()
        & (frame["SS"].to_numpy(float) <= limits["SS_max"])
        & (frame["Ion"].to_numpy(float) >= limits["Ion_min"])
        & (frame["Ioff"].to_numpy(float) <= limits["Ioff_max"])
    )


def classify_existing_case(case: dict[str, object]) -> tuple[str, str]:
    """Classify a completed mixed-suite case without rerunning TCAD."""

    reasons: list[str] = []
    if str(case.get("CaseType")) == "Boundary":
        reasons.append("multivariate output boundary")

    limits = PRACTICAL_TARGET_LIMITS
    checks = (
        (limits["Vth_min"] <= float(case["Vth_Target"]) <= limits["Vth_max"], "Vth target"),
        (float(case["SS_Target"]) <= limits["SS_max"], "SS target"),
        (float(case["Ion_Target"]) >= limits["Ion_min"], "Ion target"),
        (float(case["Ioff_Target"]) <= limits["Ioff_max"], "Ioff target"),
    )
    reasons.extend(label for passed, label in checks if not passed)
    if reasons:
        return "Boundary", "; ".join(reasons)
    return "Practical", "within practical target window"


def train_short_bundle(workbook_path: Path) -> dict[str, object]:
    """Train the same pooled Short classifier/regressors used by app.py."""

    config = DEVICE_CONFIGS["Short"]
    source = load_device_sheet(workbook_path, config)
    recipe_table = build_recipe_classification_table(source, config)
    regression_table = build_regression_table(source, config).reset_index(drop=True)

    classifier = RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    classifier.fit(recipe_table[FEATURES], recipe_table["Label"])

    regressors: dict[str, RandomForestRegressor] = {}
    for target in MODEL_TARGETS:
        model = RandomForestRegressor(
            n_estimators=N_TREES,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        model.fit(
            regression_table[FEATURES],
            _transform_target(target, regression_table[target].to_numpy(float)),
        )
        regressors[target] = model

    return {
        "classifier": classifier,
        "regressors": regressors,
        "training_table": regression_table,
    }


def generate_candidates(
    training_table: pd.DataFrame,
    gate_length: float,
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Use the same min/max, log-uniform, and uniform rules as app.py."""

    data: dict[str, np.ndarray] = {}
    for feature in FEATURES:
        values = training_table[feature].to_numpy(float)
        if feature == "Lg":
            data[feature] = np.full(count, gate_length)
        elif feature in LOG_FEATURES:
            data[feature] = np.power(
                10.0,
                rng.uniform(np.log10(values.min()), np.log10(values.max()), count),
            )
        else:
            data[feature] = rng.uniform(values.min(), values.max(), count)
    return pd.DataFrame(data, columns=FEATURES)


def predict_valid_pool(
    bundle: dict[str, object], candidates: pd.DataFrame
) -> pd.DataFrame:
    classifier = bundle["classifier"]
    probabilities = classifier.predict_proba(candidates)[:, 1]
    valid_mask = probabilities >= VALID_PROBABILITY_THRESHOLD
    valid = candidates.loc[valid_mask].copy()
    valid["valid_probability"] = probabilities[valid_mask]

    for target, model in bundle["regressors"].items():
        prediction = model.predict(valid[FEATURES])
        valid[target] = _inverse_target(target, prediction)
    valid["OnOff"] = valid["Ion"] / np.clip(valid["Ioff"], 1e-30, None)
    return valid.reset_index(drop=True)


def target_error(target: str, prediction: np.ndarray, target_value: float) -> np.ndarray:
    """Copy app.py's score: log error for currents/ratio, relative otherwise."""

    if target in LOG_SCORE_TARGETS:
        return (
            np.log10(np.clip(prediction, 1e-30, None))
            - np.log10(max(target_value, 1e-30))
        ) ** 2
    denominator = max(abs(target_value), 1e-12)
    return ((prediction - target_value) / denominator) ** 2


def transformed_outputs(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame["Vth"].to_numpy(float),
            np.log10(frame["SS"].to_numpy(float)),
            np.log10(frame["Ion"].to_numpy(float)),
            np.log10(frame["Ioff"].to_numpy(float)),
        ]
    )


def standardized_outputs(frame: pd.DataFrame) -> np.ndarray:
    values = transformed_outputs(frame)
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = np.where((q75 - q25) > 1e-12, q75 - q25, 1.0)
    return (values - median) / scale


def maximin_indices(
    standardized: np.ndarray,
    candidates: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    """Select a diverse subset without fabricating independent target tuples."""

    pool = np.asarray(candidates, dtype=int)
    if count <= 0:
        return []
    if len(pool) <= count:
        return pool.tolist()

    first = int(rng.choice(pool))
    selected = [first]
    nearest = np.sum((standardized[pool] - standardized[first]) ** 2, axis=1)
    available = np.ones(len(pool), dtype=bool)
    available[np.where(pool == first)[0][0]] = False

    for _ in range(1, count):
        masked = np.where(available, nearest, -np.inf)
        position = int(np.argmax(masked))
        chosen = int(pool[position])
        selected.append(chosen)
        available[position] = False
        distance = np.sum(
            (standardized[pool] - standardized[chosen]) ** 2, axis=1
        )
        nearest = np.minimum(nearest, distance)
    return selected


def choose_target_rows(
    table_for_lg: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    profile: str = "mixed",
) -> list[tuple[int, str]]:
    """Select mixed, practical-use, or boundary-only target rows."""

    z = standardized_outputs(table_for_lg)
    distance = np.sqrt(np.sum(z**2, axis=1))
    typical_pool = np.flatnonzero(distance <= np.quantile(distance, 0.60))
    edge_pool = np.flatnonzero(distance >= np.quantile(distance, 0.75))

    if profile == "practical":
        candidates = np.intersect1d(
            typical_pool,
            np.flatnonzero(practical_output_mask(table_for_lg)),
        )
        if len(candidates) < count:
            raise ValueError(
                f"Practical target pool has {len(candidates)} rows; {count} required."
            )
        return [(idx, "Practical") for idx in maximin_indices(z, candidates, count, rng)]

    if profile == "boundary":
        non_practical = np.flatnonzero(~practical_output_mask(table_for_lg))
        candidates = np.union1d(edge_pool, non_practical)
        if len(candidates) < count:
            raise ValueError(
                f"Boundary target pool has {len(candidates)} rows; {count} required."
            )
        return [(idx, "Boundary") for idx in maximin_indices(z, candidates, count, rng)]

    if profile != "mixed":
        raise ValueError(f"Unknown validation profile: {profile}")

    typical_n = int(round(count * 0.40))
    edge_n = int(round(count * 0.30))
    coverage_n = count - typical_n - edge_n

    selected: list[tuple[int, str]] = []
    used: set[int] = set()
    for idx in maximin_indices(z, typical_pool, typical_n, rng):
        selected.append((idx, "Typical"))
        used.add(idx)
    for idx in maximin_indices(z, edge_pool, edge_n, rng):
        if idx not in used:
            selected.append((idx, "Boundary"))
            used.add(idx)

    remainder = np.asarray([i for i in range(len(table_for_lg)) if i not in used])
    for idx in maximin_indices(z, remainder, coverage_n, rng):
        selected.append((idx, "Coverage"))
        used.add(idx)

    # Defensive fill for unusually small datasets or category overlap.
    if len(selected) < count:
        remainder = [i for i in range(len(table_for_lg)) if i not in used]
        for idx in rng.choice(remainder, size=count - len(selected), replace=False):
            selected.append((int(idx), "Coverage"))
    rng.shuffle(selected)
    return selected[:count]


def perturb_target(
    row: pd.Series,
    table_for_lg: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Make a nearby non-tabulated target while retaining physical correlation."""

    values: dict[str, float] = {}
    for target in MODEL_TARGETS:
        source = float(abs(row[target]))
        column = np.abs(table_for_lg[target].to_numpy(float))
        if target == "Vth":
            span = float(np.max(column) - np.min(column))
            value = source + rng.normal(0.0, 0.008 * max(span, 1e-12))
        else:
            sigma = 0.025 if target == "SS" else 0.05
            value = source * (10.0 ** rng.normal(0.0, sigma))
        values[target] = float(np.clip(value, np.min(column), np.max(column)))
    values["OnOff"] = values["Ion"] / max(values["Ioff"], 1e-30)
    return values


def make_specs(targets: dict[str, float], scale: float) -> dict[str, dict[str, float]]:
    """Create explicit UI min/target/max values for the four trained outputs."""

    specs: dict[str, dict[str, float]] = {}
    vth_tolerance = max(0.05, 0.10 * abs(targets["Vth"])) * scale
    specs["Vth"] = {
        "min": max(0.0, targets["Vth"] - vth_tolerance),
        "target": targets["Vth"],
        "max": targets["Vth"] + vth_tolerance,
    }
    for target, base_decades in (("SS", 0.12), ("Ion", 0.30), ("Ioff", 0.30)):
        factor = 10.0 ** (base_decades * scale)
        specs[target] = {
            "min": targets[target] / factor,
            "target": targets[target],
            "max": targets[target] * factor,
        }
    return specs


def recommend_one(
    valid_pool: pd.DataFrame,
    targets: dict[str, float],
) -> tuple[pd.Series, dict[str, dict[str, float]], float, int]:
    """Return TOP1 and widen only when the initial feasible window is empty."""

    score = np.zeros(len(valid_pool), dtype=float)
    for target in MODEL_TARGETS:
        score += target_error(
            target,
            valid_pool[target].to_numpy(float),
            targets[target],
        )

    for scale in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        specs = make_specs(targets, scale)
        mask = np.ones(len(valid_pool), dtype=bool)
        for target, spec in specs.items():
            values = valid_pool[target].to_numpy(float)
            mask &= values >= spec["min"]
            mask &= values <= spec["max"]
        matches = np.flatnonzero(mask)
        if len(matches):
            best_position = int(matches[np.argmin(score[matches])])
            best = valid_pool.iloc[best_position].copy()
            best["score"] = float(score[best_position])
            return best, specs, scale, int(len(matches))

    # A result is still useful for diagnosing an unreachable target.  It is
    # explicitly flagged with zero feasible matches rather than silently passed.
    best_position = int(np.argmin(score))
    best = valid_pool.iloc[best_position].copy()
    best["score"] = float(score[best_position])
    return best, make_specs(targets, 8.0), 8.0, 0


def rounded_recipe_key(row: pd.Series) -> tuple[str, ...]:
    return tuple(format(float(row[feature]), ".12g") for feature in FEATURES)


def create_batch(
    bundle: dict[str, object],
    workbook_path: Path,
    gate_lengths: tuple[float, ...],
    cases_per_lg: int,
    candidate_count: int,
    seed: int,
    test_profile: str = "mixed",
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    training_table = bundle["training_table"]
    practical_bounds: dict[str, tuple[float, float]] = {}
    for target in MODEL_TARGETS:
        values = np.abs(training_table[target].to_numpy(float))
        if target in {"Ion", "Ioff"}:
            low, high = np.power(
                10.0, np.quantile(np.log10(np.clip(values, 1e-30, None)), [0.10, 0.90])
            )
        else:
            low, high = np.quantile(values, [0.10, 0.90])
        practical_bounds[target] = (float(low), float(high))
    rows: list[dict[str, object]] = []
    recipe_ids: dict[tuple[str, ...], str] = {}
    unique_recipes: list[dict[str, object]] = []

    for gate_length in gate_lengths:
        print(f"Generating {candidate_count:,} candidates for Lg={gate_length:g} um...")
        candidate_rng = np.random.default_rng(
            candidate_seed_for_gate_length(gate_length, seed)
        )
        candidates = generate_candidates(
            training_table, gate_length, candidate_count, candidate_rng
        )
        valid_pool = predict_valid_pool(bundle, candidates)
        print(f"  classifier-valid pool: {len(valid_pool):,}")
        if valid_pool.empty:
            raise RuntimeError(f"No valid candidates for Lg={gate_length:g}")

        practical_mask = np.isclose(training_table["Lg"], gate_length)
        for target, (low, high) in practical_bounds.items():
            values = np.abs(training_table[target].to_numpy(float))
            practical_mask &= (values >= low) & (values <= high)
        table_for_lg = training_table[practical_mask]
        table_for_lg = table_for_lg.reset_index(drop=True)
        if len(table_for_lg) < cases_per_lg:
            raise ValueError(
                f"Lg={gate_length:g} has only {len(table_for_lg)} valid rows."
            )

        selections = choose_target_rows(
            table_for_lg, cases_per_lg, rng, profile=test_profile
        )
        for local_number, (row_index, case_type) in enumerate(selections, start=1):
            source_row = table_for_lg.iloc[row_index]
            targets = perturb_target(source_row, table_for_lg, rng)
            best, specs, tolerance_scale, feasible_count = recommend_one(
                valid_pool, targets
            )

            key = rounded_recipe_key(best)
            if key not in recipe_ids:
                recipe_id = f"R{len(recipe_ids) + 1:03d}"
                recipe_ids[key] = recipe_id
                unique_recipes.append(
                    {
                        "RecipeID": recipe_id,
                        **{feature: float(best[feature]) for feature in FEATURES},
                    }
                )
            recipe_id = recipe_ids[key]
            case_id = f"S{int(round(gate_length * 1000)):03d}-{local_number:02d}"
            row: dict[str, object] = {
                "CaseID": case_id,
                "RecipeID": recipe_id,
                "CaseType": case_type,
                "Lg": gate_length,
                "ToleranceScale": tolerance_scale,
                "FeasibleCandidateCount": feasible_count,
            }
            for target in MODEL_TARGETS:
                row[f"{target}_Min"] = specs[target]["min"]
                row[f"{target}_Target"] = specs[target]["target"]
                row[f"{target}_Max"] = specs[target]["max"]
            row["OnOff_Target"] = targets["OnOff"]
            row.update({feature: float(best[feature]) for feature in FEATURES})
            row.update(
                {
                    "Pred_Vth": float(best["Vth"]),
                    "Pred_SS": float(best["SS"]),
                    "Pred_Ion": float(best["Ion"]),
                    "Pred_Ioff": float(best["Ioff"]),
                    "Pred_OnOff": float(best["OnOff"]),
                    "ValidProbability": float(best["valid_probability"]),
                    "Score": float(best["score"]),
                }
            )
            rows.append(row)

    recipe_to_cases: dict[str, list[str]] = {}
    for row in rows:
        recipe_to_cases.setdefault(str(row["RecipeID"]), []).append(str(row["CaseID"]))
    for recipe in unique_recipes:
        recipe["CaseIDs"] = ", ".join(recipe_to_cases[str(recipe["RecipeID"])])

    return {
        "metadata": {
            "workbook": str(workbook_path.resolve()),
            "model": "Pooled Short Random Forest",
            "trees": N_TREES,
            "candidate_count_per_lg": candidate_count,
            "valid_probability_threshold": VALID_PROBABILITY_THRESHOLD,
            "seed": seed,
            "gate_lengths": list(gate_lengths),
            "cases_per_lg": cases_per_lg,
            "case_count": len(rows),
            "unique_recipe_count": len(unique_recipes),
            "target_sampling_range": "Pooled Short UI default 10th-90th percentiles",
            "practical_target_bounds": {
                target: {"min": bounds[0], "max": bounds[1]}
                for target, bounds in practical_bounds.items()
            },
            "primary_comparison": "Predicted specs vs TCAD specs",
            "secondary_comparison": "Target window vs TCAD specs",
            "test_profile": test_profile,
            "practical_suite_limits": PRACTICAL_TARGET_LIMITS,
        },
        "cases": rows,
        "unique_recipes": unique_recipes,
    }


def write_tcad_csv(path: Path, recipes: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "RecipeID",
        "CaseIDs",
        *FEATURES,
        "TCAD_Vth_V",
        "TCAD_SS_mVdec",
        "TCAD_Ion_A",
        "TCAD_Ioff_A",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for recipe in recipes:
            formatted = dict(recipe)
            for feature in LOG_FEATURES:
                formatted[feature] = format(float(recipe[feature]), ".6e")
            writer.writerow(formatted)


def write_swb_import_files(output_dir: Path, recipes: list[dict[str, object]]) -> None:
    """Write process-only and duplicated-Vd files for Sentaurus Workbench."""

    output_dir.mkdir(parents=True, exist_ok=True)
    process_path = output_dir / "swb_import_short_process.csv"
    with process_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURES)
        writer.writeheader()
        for recipe in recipes:
            row = {feature: recipe[feature] for feature in FEATURES}
            for feature in LOG_FEATURES:
                row[feature] = format(float(recipe[feature]), ".6e")
            writer.writerow(row)

    for filename in ("swb_import_short_full.csv", "swb_import_short_full.txt"):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*FEATURES, "Vd"])
            writer.writeheader()
            for recipe in recipes:
                base = {feature: recipe[feature] for feature in FEATURES}
                for feature in LOG_FEATURES:
                    base[feature] = format(float(recipe[feature]), ".6e")
                for drain_bias in REQUIRED_BIASES:
                    writer.writerow({**base, "Vd": drain_bias})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--cases-per-lg", type=int, default=50)
    parser.add_argument("--candidate-count", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=CANDIDATE_SEED)
    parser.add_argument(
        "--gate-lengths", type=float, nargs="+", default=list(DEFAULT_GATE_LENGTHS)
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--test-profile",
        choices=("mixed", "practical", "boundary"),
        default="mixed",
        help="Target suite: mixed legacy set, practical-use only, or boundary stress only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.workbook.exists():
        raise FileNotFoundError(args.workbook)
    print("Training pooled Short model...")
    bundle = train_short_bundle(args.workbook)
    batch = create_batch(
        bundle,
        args.workbook,
        tuple(args.gate_lengths),
        args.cases_per_lg,
        args.candidate_count,
        args.seed,
        args.test_profile,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_tcad_csv(args.output_csv, batch["unique_recipes"])
    write_swb_import_files(args.output_csv.parent, batch["unique_recipes"])
    print(
        f"Wrote {batch['metadata']['case_count']} cases and "
        f"{batch['metadata']['unique_recipe_count']} unique recipes."
    )
    print(args.output_json)
    print(args.output_csv)


if __name__ == "__main__":
    main()
