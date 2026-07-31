from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    import lightgbm as lgb
except Exception as exc:
    raise RuntimeError("lightgbm is required") from exc

try:
    import xgboost as xgb
except Exception as exc:
    raise RuntimeError("xgboost is required") from exc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_table(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    elif suffix == ".xlsx":
        frame.to_excel(path, index=False)
    elif suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def stable_hash(values):
    text = "\n".join(sorted(map(str, values)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_columns(frame, columns, name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def numeric_frame(frame, columns):
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def validate_sm(values, lower, upper, name):
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array)
    if valid.sum() == 0:
        raise ValueError(f"{name} has no finite values")
    invalid = valid & ((array < lower) | (array > upper))
    if invalid.any():
        raise ValueError(
            f"{name} contains {int(invalid.sum())} values outside [{lower}, {upper}]"
        )


def add_temporal_features(frame, year_column, month_column, start_year, end_year):
    out = frame.copy()
    month = out[month_column].to_numpy(dtype=np.float64)
    year = out[year_column].to_numpy(dtype=np.float64)
    out["Month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["Month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    denominator = max(float(end_year - start_year), 1.0)
    out["Year_norm"] = (year - float(start_year)) / denominator
    return out


def calibration_matrix(pm_raw, year, month, start_year, end_year):
    pm = np.asarray(pm_raw, dtype=np.float64)
    year = np.asarray(year, dtype=np.float64)
    month = np.asarray(month, dtype=np.float64)
    month_sin = np.sin(2.0 * np.pi * month / 12.0)
    month_cos = np.cos(2.0 * np.pi * month / 12.0)
    denominator = max(float(end_year - start_year), 1.0)
    year_norm = (year - float(start_year)) / denominator
    return np.column_stack(
        [
            pm,
            pm ** 2,
            month_sin,
            month_cos,
            year_norm,
            pm * month_sin,
            pm * month_cos,
        ]
    )


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) < 3:
        return {
            "N": int(len(y_true)),
            "R": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "ubRMSE": np.nan,
        }
    correlation = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if np.std(y_true) > 0 and np.std(y_pred) > 0
        else np.nan
    )
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(np.mean(y_pred - y_true))
    return {
        "N": int(len(y_true)),
        "R": correlation,
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": bias,
        "ubRMSE": float(np.sqrt(max(rmse ** 2 - bias ** 2, 0.0))),
    }


def weighted_rmse(y_true, y_pred, weights):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(weights) & (weights > 0)
    if valid.sum() == 0:
        return np.inf
    error = y_pred[valid] - y_true[valid]
    return float(np.sqrt(np.average(error ** 2, weights=weights[valid])))


def cluster_bootstrap(predictions, station_column, observation_column, model_columns, replicates, seed):
    stations = np.array(sorted(predictions[station_column].astype(str).unique()))
    station_indices = {
        station: np.flatnonzero(predictions[station_column].astype(str).to_numpy() == station)
        for station in stations
    }
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        indices = np.concatenate([station_indices[station] for station in sampled])
        subset = predictions.iloc[indices]
        for model in model_columns:
            result = metrics(subset[observation_column], subset[model])
            result.update({"Replicate": replicate, "Model": model})
            rows.append(result)
    bootstrap = pd.DataFrame(rows)
    summary_rows = []
    for model, group in bootstrap.groupby("Model"):
        row = {"Model": model}
        for metric_name in ["R", "R2", "RMSE", "MAE", "Bias", "ubRMSE"]:
            values = group[metric_name].dropna().to_numpy()
            row[f"{metric_name}_CI_Lower"] = (
                float(np.percentile(values, 2.5)) if len(values) else np.nan
            )
            row[f"{metric_name}_CI_Upper"] = (
                float(np.percentile(values, 97.5)) if len(values) else np.nan
            )
        summary_rows.append(row)
    return bootstrap, pd.DataFrame(summary_rows)


def split_stations(station_frame, station_column, test_fraction, seed):
    stations = np.array(sorted(station_frame[station_column].astype(str).unique()))
    if len(stations) < 3:
        raise ValueError("At least three stations are required")
    train_stations, test_stations = train_test_split(
        stations,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
    )
    train_set = set(map(str, train_stations))
    test_set = set(map(str, test_stations))
    overlap = train_set & test_set
    if overlap:
        raise RuntimeError(f"Station overlap detected: {sorted(overlap)}")
    train = station_frame[station_frame[station_column].astype(str).isin(train_set)].copy()
    test = station_frame[station_frame[station_column].astype(str).isin(test_set)].copy()
    return train, test, train_set, test_set


def internal_split_indices(source, group, test_fraction, seed):
    source = np.asarray(source)
    group = np.asarray(group).astype(str)
    train_indices = []
    validation_indices = []
    for offset, source_name in enumerate(sorted(np.unique(source))):
        indices = np.flatnonzero(source == source_name)
        groups = group[indices]
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_fraction,
                random_state=seed + offset,
            )
            local_train, local_validation = next(
                splitter.split(np.zeros(len(indices)), groups=groups)
            )
        else:
            local_train, local_validation = train_test_split(
                np.arange(len(indices)),
                test_size=test_fraction,
                random_state=seed + offset,
                shuffle=True,
            )
        train_indices.append(indices[local_train])
        validation_indices.append(indices[local_validation])
    return np.concatenate(train_indices), np.concatenate(validation_indices)


def build_learners(config):
    seed = int(config["random_seed"])
    threads = int(config["threads"])
    return {
        "XGBoost": xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=threads,
            **config["xgboost"],
        ),
        "RF": RandomForestRegressor(
            random_state=seed,
            n_jobs=threads,
            **config["random_forest"],
        ),
        "LightGBM": lgb.LGBMRegressor(
            objective="regression",
            random_state=seed,
            n_jobs=threads,
            verbosity=-1,
            **config["lightgbm"],
        ),
        "GBDT": GradientBoostingRegressor(
            random_state=seed,
            **config["gradient_boosting"],
        ),
    }


def fit_models(models, x, y, weights):
    fitted = {}
    for name, model in models.items():
        model.fit(x, y, sample_weight=weights)
        fitted[name] = model
    return fitted


def predict_models(models, x):
    return {
        name: np.asarray(model.predict(x), dtype=np.float64)
        for name, model in models.items()
    }


def select_pwe_weights(y, predictions, sample_weights, powers):
    rows = []
    best = None
    for power in powers:
        errors = {
            name: weighted_rmse(y, prediction, sample_weights)
            for name, prediction in predictions.items()
        }
        if any(not np.isfinite(value) or value <= 0 for value in errors.values()):
            raise RuntimeError(f"Invalid base-learner errors: {errors}")
        raw = {name: 1.0 / (value ** float(power)) for name, value in errors.items()}
        total = sum(raw.values())
        weights = {name: value / total for name, value in raw.items()}
        ensemble = sum(weights[name] * predictions[name] for name in predictions)
        ensemble_rmse = weighted_rmse(y, ensemble, sample_weights)
        row = {
            "Power": float(power),
            "Ensemble_RMSE": ensemble_rmse,
            **{f"RMSE_{name}": errors[name] for name in errors},
            **{f"Weight_{name}": weights[name] for name in weights},
        }
        rows.append(row)
        candidate = (ensemble_rmse, float(power), weights)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2], pd.DataFrame(rows)


def train_branch(x, y, weights, source, groups, feature_names, config, target_type):
    train_index, validation_index = internal_split_indices(
        source,
        groups,
        float(config["internal_validation_fraction"]),
        int(config["random_seed"]),
    )
    internal_imputer = SimpleImputer(strategy="median")
    x_train = internal_imputer.fit_transform(x[train_index])
    x_validation = internal_imputer.transform(x[validation_index])
    provisional_models = fit_models(
        build_learners(config),
        x_train,
        y[train_index],
        weights[train_index],
    )
    validation_predictions = predict_models(provisional_models, x_validation)
    selected_weights, validation_table = select_pwe_weights(
        y[validation_index],
        validation_predictions,
        weights[validation_index],
        config["pwe_candidate_powers"],
    )
    final_imputer = SimpleImputer(strategy="median")
    x_all = final_imputer.fit_transform(x)
    final_models = fit_models(build_learners(config), x_all, y, weights)
    return {
        "feature_names": feature_names,
        "target_type": target_type,
        "imputer": final_imputer,
        "models": final_models,
        "weights": selected_weights,
        "internal_validation": validation_table,
    }


def predict_branch(package, frame, lower, upper, residual_lower, residual_upper):
    x = package["imputer"].transform(frame[package["feature_names"]].to_numpy(dtype=np.float64))
    base_predictions = predict_models(package["models"], x)
    prediction = sum(
        package["weights"][name] * base_predictions[name]
        for name in package["weights"]
    )
    if package["target_type"] == "residual":
        return np.clip(prediction, residual_lower, residual_upper)
    return np.clip(prediction, lower, upper)


def station_weight(grid_count, station_count, alpha, lower, upper):
    if station_count <= 0:
        raise ValueError("No training-station records")
    value = alpha * float(grid_count) / float(station_count)
    return float(np.clip(value, lower, upper))


def apply_directional_constraint(delta, direction, lam, epsilon, lower, upper):
    delta = np.asarray(delta, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    output = delta.copy()
    active = np.isfinite(delta) & np.isfinite(direction) & (np.abs(direction) > epsilon)
    forced = np.sign(direction) * np.abs(delta)
    output[active] = (1.0 - lam) * delta[active] + lam * forced[active]
    return np.clip(output, lower, upper)


def adaptive_weight(prior, threshold, smoothing, minimum_prior, maximum_weight):
    prior = np.asarray(prior, dtype=np.float64)
    threshold = np.asarray(threshold, dtype=np.float64)
    z = np.clip((prior - threshold) / max(smoothing, 1e-12), -60.0, 60.0)
    weight = maximum_weight / (1.0 + np.exp(-z))
    weight[(~np.isfinite(prior)) | (~np.isfinite(threshold)) | (prior < minimum_prior)] = 0.0
    return np.clip(weight, 0.0, maximum_weight)


def prepare_inputs(config):
    columns = config["columns"]
    predictors = list(columns["predictors"])
    generated = {"Month_sin", "Month_cos", "Year_norm"}
    supplied_predictors = [name for name in predictors if name not in generated]
    station = read_table(config["paths"]["station_samples"])
    grid = read_table(config["paths"]["grid_samples"])
    station_core = [
        columns["station_id"],
        columns["year"],
        columns["month"],
        columns["observed_sm"],
        columns["coarse_sm"],
        columns["pm_prior_raw"],
        columns["direction_signal"],
    ]
    grid_core = [
        columns["year"],
        columns["month"],
        columns["coarse_sm"],
        columns["pm_prior_raw"],
    ]
    require_columns(station, station_core + supplied_predictors, "station_samples")
    require_columns(grid, grid_core + supplied_predictors, "grid_samples")
    station = numeric_frame(
        station,
        [
            columns["year"],
            columns["month"],
            columns["observed_sm"],
            columns["coarse_sm"],
            columns["pm_prior_raw"],
            columns["direction_signal"],
            *supplied_predictors,
        ],
    )
    grid = numeric_frame(
        grid,
        [
            columns["year"],
            columns["month"],
            columns["coarse_sm"],
            columns["pm_prior_raw"],
            *supplied_predictors,
        ],
    )
    station = station.dropna(subset=station_core).copy()
    grid = grid.dropna(subset=grid_core).copy()
    station[columns["station_id"]] = station[columns["station_id"]].astype(str)
    station = add_temporal_features(
        station,
        columns["year"],
        columns["month"],
        int(config["period"]["start_year"]),
        int(config["period"]["end_year"]),
    )
    grid = add_temporal_features(
        grid,
        columns["year"],
        columns["month"],
        int(config["period"]["start_year"]),
        int(config["period"]["end_year"]),
    )
    lower = float(config["soil_moisture"]["lower"])
    upper = float(config["soil_moisture"]["upper"])
    validate_sm(station[columns["observed_sm"]], lower, upper, "observed_sm")
    validate_sm(station[columns["coarse_sm"]], lower, upper, "station coarse_sm")
    validate_sm(station[columns["pm_prior_raw"]], lower, upper, "station pm_prior_raw")
    validate_sm(grid[columns["coarse_sm"]], lower, upper, "grid coarse_sm")
    validate_sm(grid[columns["pm_prior_raw"]], lower, upper, "grid pm_prior_raw")
    return station, grid, predictors


def main():
    args = parse_args()
    config = load_yaml(args.config)
    output_root = Path(config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(config, output_root / "configuration.json")

    columns = config["columns"]
    station, grid, predictors = prepare_inputs(config)
    train_station, test_station, train_ids, test_ids = split_stations(
        station,
        columns["station_id"],
        float(config["split"]["holdout_fraction"]),
        int(config["model"]["random_seed"]),
    )

    split_manifest = {
        "station_split_unit": columns["station_id"],
        "random_seed": int(config["model"]["random_seed"]),
        "holdout_fraction": float(config["split"]["holdout_fraction"]),
        "training_station_count": len(train_ids),
        "holdout_station_count": len(test_ids),
        "overlap_count": len(train_ids & test_ids),
        "training_station_sha256": stable_hash(train_ids),
        "holdout_station_sha256": stable_hash(test_ids),
        "holdout_labels_used_for_training": False,
        "holdout_labels_used_for_prior_calibration": False,
        "holdout_labels_used_for_pwe_selection": False,
    }
    write_json(split_manifest, output_root / "station_split_manifest.json")
    write_table(train_station, output_root / "training_station_records.csv")
    write_table(test_station, output_root / "holdout_station_records.csv")

    start_year = int(config["period"]["start_year"])
    end_year = int(config["period"]["end_year"])
    if len(train_station) < int(config["prior_calibration"]["minimum_samples"]):
        raise ValueError("Insufficient training-station records for prior calibration")
    calibration_x = calibration_matrix(
        train_station[columns["pm_prior_raw"]],
        train_station[columns["year"]],
        train_station[columns["month"]],
        start_year,
        end_year,
    )
    calibration_y = train_station[columns["observed_sm"]].to_numpy(dtype=np.float64)
    calibrator = Ridge(alpha=float(config["prior_calibration"]["ridge_alpha"]))
    calibrator.fit(calibration_x, calibration_y)

    for frame in [train_station, test_station, grid]:
        matrix = calibration_matrix(
            frame[columns["pm_prior_raw"]],
            frame[columns["year"]],
            frame[columns["month"]],
            start_year,
            end_year,
        )
        frame["PM_prior_calibrated"] = np.clip(
            calibrator.predict(matrix),
            float(config["soil_moisture"]["lower"]),
            float(config["soil_moisture"]["upper"]),
        )

    calibration_diagnostics = {
        "training_records": int(len(train_station)),
        "ridge_alpha": float(config["prior_calibration"]["ridge_alpha"]),
        "coefficient": calibrator.coef_.tolist(),
        "intercept": float(calibrator.intercept_),
        "raw_prior_metrics": metrics(
            train_station[columns["observed_sm"]],
            train_station[columns["pm_prior_raw"]],
        ),
        "calibrated_prior_metrics": metrics(
            train_station[columns["observed_sm"]],
            train_station["PM_prior_calibrated"],
        ),
    }
    write_json(calibration_diagnostics, output_root / "prior_calibration.json")
    joblib.dump(calibrator, output_root / "pm_prior_calibrator.joblib")

    station_sample_weight = station_weight(
        len(grid),
        len(train_station),
        float(config["training"]["station_total_weight_fraction"]),
        float(config["training"]["station_weight_lower"]),
        float(config["training"]["station_weight_upper"]),
    )

    grid_training = grid.copy()
    station_training = train_station.copy()
    grid_training["_Source"] = "grid"
    station_training["_Source"] = "station"
    grid_group_column = columns.get("grid_group")
    if grid_group_column and grid_group_column in grid_training.columns:
        grid_training["_Group"] = "grid:" + grid_training[grid_group_column].astype(str)
    else:
        grid_training["_Group"] = (
            "grid:"
            + grid_training[columns["year"]].astype(int).astype(str)
            + "-"
            + grid_training[columns["month"]].astype(int).astype(str).str.zfill(2)
        )
    station_training["_Group"] = "station:" + station_training[columns["station_id"]].astype(str)
    grid_training["_Target"] = grid_training[columns["coarse_sm"]]
    station_training["_Target"] = station_training[columns["observed_sm"]]
    grid_training["_Weight"] = 1.0
    station_training["_Weight"] = station_sample_weight

    training = pd.concat([grid_training, station_training], ignore_index=True, sort=False)
    source = training["_Source"].to_numpy()
    groups = training["_Group"].to_numpy()
    sample_weights = training["_Weight"].to_numpy(dtype=np.float64)
    direct_target = training["_Target"].to_numpy(dtype=np.float64)
    residual_target = direct_target - training["PM_prior_calibrated"].to_numpy(dtype=np.float64)

    model_config = config["model"]
    m1_features = predictors
    m2_features = predictors + ["PM_prior_calibrated"]

    m1 = train_branch(
        training[m1_features].to_numpy(dtype=np.float64),
        direct_target,
        sample_weights,
        source,
        groups,
        m1_features,
        model_config,
        "direct",
    )
    m2 = train_branch(
        training[m2_features].to_numpy(dtype=np.float64),
        direct_target,
        sample_weights,
        source,
        groups,
        m2_features,
        model_config,
        "direct",
    )
    m3 = train_branch(
        training[m1_features].to_numpy(dtype=np.float64),
        residual_target,
        sample_weights,
        source,
        groups,
        m1_features,
        model_config,
        "residual",
    )

    packages = {"M1": m1, "M2": m2, "M3": m3}
    joblib.dump(packages, output_root / "sm_m0_m5_model_package.joblib")
    for name, package in packages.items():
        write_table(
            package["internal_validation"],
            output_root / f"{name}_internal_validation.csv",
        )
        write_json(
            package["weights"],
            output_root / f"{name}_pwe_weights.json",
        )

    lower = float(config["soil_moisture"]["lower"])
    upper = float(config["soil_moisture"]["upper"])
    residual_lower = float(config["soil_moisture"]["residual_lower"])
    residual_upper = float(config["soil_moisture"]["residual_upper"])

    test_station["M0"] = test_station[columns["coarse_sm"]].to_numpy(dtype=np.float64)
    test_station["P0"] = test_station["PM_prior_calibrated"].to_numpy(dtype=np.float64)
    test_station["M1"] = predict_branch(
        m1,
        test_station,
        lower,
        upper,
        residual_lower,
        residual_upper,
    )
    test_station["M2"] = predict_branch(
        m2,
        test_station,
        lower,
        upper,
        residual_lower,
        residual_upper,
    )
    delta = predict_branch(
        m3,
        test_station,
        lower,
        upper,
        residual_lower,
        residual_upper,
    )
    test_station["M3_residual"] = delta
    test_station["M3"] = np.clip(test_station["P0"] + delta, lower, upper)
    constrained = apply_directional_constraint(
        delta,
        test_station[columns["direction_signal"]],
        float(config["m4"]["lambda"]),
        float(config["m4"]["sign_epsilon"]),
        residual_lower,
        residual_upper,
    )
    test_station["M4_residual"] = constrained
    test_station["M4"] = np.clip(test_station["P0"] + constrained, lower, upper)

    threshold_table = (
        grid.groupby([columns["year"], columns["month"]])["PM_prior_calibrated"]
        .quantile(float(config["m5"]["gate_percentile"]) / 100.0)
        .rename("M5_threshold")
        .reset_index()
    )
    month_counts = (
        grid.groupby([columns["year"], columns["month"]])
        .size()
        .rename("Prior_pixel_count")
        .reset_index()
    )
    threshold_table = threshold_table.merge(
        month_counts,
        on=[columns["year"], columns["month"]],
        how="left",
    )
    minimum_pixels = int(config["m5"]["minimum_prior_pixels_per_month"])
    if (threshold_table["Prior_pixel_count"] < minimum_pixels).any():
        failed = threshold_table[
            threshold_table["Prior_pixel_count"] < minimum_pixels
        ]
        raise ValueError(
            f"Insufficient monthly prior support: {failed.to_dict(orient='records')}"
        )
    test_station = test_station.merge(
        threshold_table,
        on=[columns["year"], columns["month"]],
        how="left",
        validate="many_to_one",
    )
    if test_station["M5_threshold"].isna().any():
        raise ValueError("Missing M5 threshold for holdout records")
    test_station["M5_weight"] = adaptive_weight(
        test_station["P0"],
        test_station["M5_threshold"],
        float(config["m5"]["smoothing"]),
        float(config["m5"]["minimum_prior"]),
        float(config["m5"]["maximum_weight"]),
    )
    test_station["M5"] = np.clip(
        (1.0 - test_station["M5_weight"]) * test_station["M1"]
        + test_station["M5_weight"] * test_station["M2"],
        lower,
        upper,
    )

    model_columns = ["M0", "P0", "M1", "M2", "M3", "M4", "M5"]
    common = test_station.dropna(
        subset=[columns["observed_sm"], *model_columns]
    ).copy()
    if len(common) == 0:
        raise ValueError("No common-support holdout records")

    overall_rows = []
    for model in model_columns:
        result = metrics(common[columns["observed_sm"]], common[model])
        result["Model"] = model
        overall_rows.append(result)
    overall = pd.DataFrame(overall_rows)

    bootstrap, confidence = cluster_bootstrap(
        common,
        columns["station_id"],
        columns["observed_sm"],
        model_columns,
        int(config["validation"]["bootstrap_replicates"]),
        int(config["model"]["random_seed"]),
    )
    overall = overall.merge(confidence, on="Model", how="left")

    write_table(test_station, output_root / "holdout_predictions_all_available.csv")
    write_table(common, output_root / "holdout_predictions_common_support.csv")
    write_table(overall, output_root / "holdout_metrics_common_support.xlsx")
    write_table(threshold_table, output_root / "m5_monthly_thresholds.csv")
    write_table(bootstrap, output_root / "station_cluster_bootstrap.csv")

    summary = {
        "training_grid_records": int(len(grid)),
        "training_station_records": int(len(train_station)),
        "holdout_station_records": int(len(test_station)),
        "common_support_records": int(len(common)),
        "training_station_count": int(len(train_ids)),
        "holdout_station_count": int(len(test_ids)),
        "station_sample_weight": station_sample_weight,
        "predictors": predictors,
        "selected_pathway_interpretation": {
            "M0": "coarse reference",
            "P0": "training-station-calibrated physical prior",
            "M1": "no-prior statistical pathway",
            "M2": "prior-as-predictor pathway",
            "M3": "residual-anchor pathway",
            "M4": "direction-constrained residual pathway",
            "M5": "label-independent adaptive fusion of M1 and M2",
        },
    }
    write_json(summary, output_root / "run_summary.json")


if __name__ == "__main__":
    main()
