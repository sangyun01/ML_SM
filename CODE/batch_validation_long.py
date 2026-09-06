"""Generate reproducible Long-channel inverse-design validation cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from CODE.batch_validation import (
    PRACTICAL_TARGET_LIMITS,
    choose_target_rows,
    perturb_target,
    recommend_one,
)
from CODE.ML_SM import (
    CANDIDATE_SEED,
    DEFAULT_WORKBOOK,
    DEVICE_CONFIGS,
    MODEL_TARGETS,
    N_TREES,
    RANDOM_STATE,
    _inverse_target,
    _transform_target,
    build_recipe_classification_table,
    build_regression_table,
    candidate_seed_for_gate_length,
    load_device_sheet,
)


DEVICE_NAME = "Long"
FEATURES = list(DEVICE_CONFIGS[DEVICE_NAME].input_features)
LOG_FEATURES = {"LDD_Dose", "SD_Dose"}
GATE_LENGTHS = (0.18, 0.36, 0.72, 1.0)
VALID_PROBABILITY_THRESHOLD = 0.5


def train_bundle(workbook_path: Path) -> dict[str, object]:
    config = DEVICE_CONFIGS[DEVICE_NAME]
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
    probabilities = bundle["classifier"].predict_proba(candidates)[:, 1]
    mask = probabilities >= VALID_PROBABILITY_THRESHOLD
    valid = candidates.loc[mask].copy()
    valid["valid_probability"] = probabilities[mask]
    for target, model in bundle["regressors"].items():
        valid[target] = _inverse_target(target, model.predict(valid[FEATURES]))
    valid["OnOff"] = valid["Ion"] / np.clip(valid["Ioff"], 1e-30, None)
    return valid.reset_index(drop=True)


def recipe_key(row: pd.Series) -> tuple[str, ...]:
    return tuple(format(float(row[feature]), ".12g") for feature in FEATURES)


def practical_bounds(training_table: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for target in MODEL_TARGETS:
        values = np.abs(training_table[target].to_numpy(float))
        if target in {"Ion", "Ioff"}:
            low, high = np.power(
                10.0,
                np.quantile(np.log10(np.clip(values, 1e-30, None)), [0.10, 0.90]),
            )
        else:
            low, high = np.quantile(values, [0.10, 0.90])
        bounds[target] = (float(low), float(high))
    return bounds


def create_batch(
    bundle: dict[str, object],
    workbook_path: Path,
    cases_per_lg: int,
    candidate_count: int,
    seed: int,
    test_profile: str = "mixed",
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    training = bundle["training_table"]
    bounds = practical_bounds(training)
    cases: list[dict[str, object]] = []
    recipe_ids: dict[tuple[str, ...], str] = {}
    recipes: list[dict[str, object]] = []

    for gate_length in GATE_LENGTHS:
        print(f"Generating {candidate_count:,} candidates for Lg={gate_length:g} um...")
        pool_rng = np.random.default_rng(
            candidate_seed_for_gate_length(gate_length, seed)
        )
        pool = predict_valid_pool(
            bundle,
            generate_candidates(training, gate_length, candidate_count, pool_rng),
        )
        print(f"  classifier-valid pool: {len(pool):,}")

        mask = np.isclose(training["Lg"], gate_length)
        for target, (low, high) in bounds.items():
            values = np.abs(training[target].to_numpy(float))
            mask &= (values >= low) & (values <= high)
        table = training[mask].reset_index(drop=True)
        if len(table) < cases_per_lg:
            raise ValueError(f"Lg={gate_length:g}: only {len(table)} practical rows")

        selections = choose_target_rows(
            table, cases_per_lg, rng, profile=test_profile
        )
        for number, (row_index, case_type) in enumerate(selections, start=1):
            targets = perturb_target(table.iloc[row_index], table, rng)
            best, specs, tolerance_scale, feasible_count = recommend_one(pool, targets)
            key = recipe_key(best)
            if key not in recipe_ids:
                recipe_id = f"LR{len(recipe_ids) + 1:03d}"
                recipe_ids[key] = recipe_id
                recipes.append(
                    {
                        "RecipeID": recipe_id,
                        **{feature: float(best[feature]) for feature in FEATURES},
                    }
                )
            recipe_id = recipe_ids[key]
            case_id = f"L{int(round(gate_length * 1000)):04d}-{number:02d}"
            record: dict[str, object] = {
                "CaseID": case_id,
                "RecipeID": recipe_id,
                "CaseType": case_type,
                "Lg": gate_length,
                "ToleranceScale": tolerance_scale,
                "FeasibleCandidateCount": feasible_count,
            }
            for target in MODEL_TARGETS:
                record[f"{target}_Min"] = specs[target]["min"]
                record[f"{target}_Target"] = specs[target]["target"]
                record[f"{target}_Max"] = specs[target]["max"]
            record["OnOff_Target"] = targets["OnOff"]
            record.update({feature: float(best[feature]) for feature in FEATURES})
            record.update(
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
            cases.append(record)

    recipe_cases: dict[str, list[str]] = {}
    for case in cases:
        recipe_cases.setdefault(str(case["RecipeID"]), []).append(str(case["CaseID"]))
    for recipe in recipes:
        recipe["CaseIDs"] = ", ".join(recipe_cases[str(recipe["RecipeID"])])

    return {
        "metadata": {
            "workbook": str(workbook_path.resolve()),
            "device": DEVICE_NAME,
            "model": "Pooled Long Random Forest",
            "features": FEATURES,
            "trees": N_TREES,
            "candidate_count_per_lg": candidate_count,
            "valid_probability_threshold": VALID_PROBABILITY_THRESHOLD,
            "seed": seed,
            "gate_lengths": list(GATE_LENGTHS),
            "cases_per_lg": cases_per_lg,
            "case_count": len(cases),
            "unique_recipe_count": len(recipes),
            "target_sampling_range": "Pooled Long UI default 10th-90th percentiles",
            "practical_target_bounds": {
                target: {"min": pair[0], "max": pair[1]}
                for target, pair in bounds.items()
            },
            "primary_comparison": "Predicted specs vs TCAD specs",
            "secondary_comparison": "Target window vs TCAD specs",
            "test_profile": test_profile,
            "practical_suite_limits": PRACTICAL_TARGET_LIMITS,
        },
        "cases": cases,
        "unique_recipes": recipes,
    }


def formatted(feature: str, value: object) -> object:
    if feature in LOG_FEATURES:
        return format(float(value), ".6e")
    return value


def write_outputs(output_dir: Path, batch: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "batch_validation_long_200.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    recipes = batch["unique_recipes"]

    with (output_dir / "tcad_input_unique_recipes_long.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        columns = [
            "RecipeID", "CaseIDs", *FEATURES,
            "TCAD_Vth_V", "TCAD_SS_mVdec", "TCAD_Ion_A", "TCAD_Ioff_A",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for recipe in recipes:
            writer.writerow(
                {**recipe, **{f: formatted(f, recipe[f]) for f in FEATURES}}
            )

    process_path = output_dir / "swb_import_long_process.csv"
    with process_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURES)
        writer.writeheader()
        for recipe in recipes:
            writer.writerow({f: formatted(f, recipe[f]) for f in FEATURES})

    for filename in ("swb_import_long_full.csv", "swb_import_long_full.txt"):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*FEATURES, "Vd"])
            writer.writeheader()
            for recipe in recipes:
                base = {f: formatted(f, recipe[f]) for f in FEATURES}
                for vd in (0.05, 1.0):
                    writer.writerow({**base, "Vd": vd})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--cases-per-lg", type=int, default=50)
    parser.add_argument("--candidate-count", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=CANDIDATE_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--test-profile",
        choices=("mixed", "practical", "boundary"),
        default="mixed",
        help="Target suite: mixed legacy set, practical-use only, or boundary stress only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = train_bundle(args.workbook)
    batch = create_batch(
        bundle,
        args.workbook,
        args.cases_per_lg,
        args.candidate_count,
        args.seed,
        args.test_profile,
    )
    write_outputs(args.output_dir, batch)
    print(
        f"Wrote {batch['metadata']['case_count']} cases and "
        f"{batch['metadata']['unique_recipe_count']} unique recipes to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
