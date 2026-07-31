from __future__ import annotations

import argparse
import calendar
import json
import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None


FEATURES = (
    "temperature",
    "humidity",
    "et0",
    "vpd",
    "water_availability",
    "ndvi",
    "precipitation",
    "slope",
    "soil_texture",
    "lst",
    "albedo",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(output_root: Path) -> logging.Logger:
    output_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sm_prior")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(
        output_root / f"run_{datetime.now():%Y%m%d_%H%M%S}.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def month_iterator(start: str, end: str):
    start_y, start_m = map(int, start.split("-"))
    end_y, end_m = map(int, end.split("-"))
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        yield y, m
        m += 1
        if m == 13:
            y += 1
            m = 1


def format_path(template: str, year: int, month: int, day: int | None = None) -> Path:
    values = {"year": year, "month": month}
    if day is not None:
        values["day"] = day
    return Path(template.format(**values))


def grid_from_raster(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Undefined CRS: {path}")
        return {
            "width": src.width,
            "height": src.height,
            "transform": src.transform,
            "crs": src.crs,
        }


def same_grid(a: dict, b: dict) -> bool:
    return (
        a["width"] == b["width"]
        and a["height"] == b["height"]
        and a["crs"] == b["crs"]
        and np.allclose(tuple(a["transform"])[:6], tuple(b["transform"])[:6])
    )


def study_mask(shapefile: Path, grid: dict) -> np.ndarray:
    if not shapefile.exists():
        raise FileNotFoundError(shapefile)
    gdf = gpd.read_file(shapefile)
    if gdf.empty or gdf.crs is None:
        raise ValueError(f"Invalid study-area shapefile: {shapefile}")
    gdf = gdf.to_crs(grid["crs"])
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        raise ValueError(f"No valid geometries: {shapefile}")
    return rasterize(
        shapes,
        out_shape=(grid["height"], grid["width"]),
        transform=grid["transform"],
        fill=0,
        default_value=1,
        dtype="uint8",
    ).astype(bool)


def fill_inside_mask(
    array: np.ndarray,
    mask: np.ndarray,
    max_fraction: float,
    method: str,
) -> tuple[np.ndarray, float]:
    out = np.asarray(array, dtype=np.float32).copy()
    valid = mask & np.isfinite(out)
    missing = mask & ~np.isfinite(out)
    fraction = float(missing.sum() / max(mask.sum(), 1))
    if not missing.any():
        out[~mask] = np.nan
        return out, 0.0
    if fraction > max_fraction:
        raise ValueError(f"Missing fraction {fraction:.4%} exceeds {max_fraction:.4%}")
    if valid.sum() < 3:
        raise ValueError("Too few valid pixels for filling")

    if method == "linear_nearest":
        yy, xx = np.indices(out.shape)
        points = np.column_stack((yy[valid], xx[valid]))
        values = out[valid]
        targets = np.column_stack((yy[missing], xx[missing]))
        linear = griddata(points, values, targets, method="linear")
        out[missing] = linear
        remaining = mask & ~np.isfinite(out)
        if remaining.any():
            nearest = griddata(
                points,
                values,
                np.column_stack((yy[remaining], xx[remaining])),
                method="nearest",
            )
            out[remaining] = nearest
    elif method == "nearest":
        indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        out[missing] = out[tuple(indices[:, missing])]
    else:
        raise ValueError(method)

    out[~mask] = np.nan
    if np.any(mask & ~np.isfinite(out)):
        raise ValueError("Filling failed")
    return out, fraction


def read_band(
    path: Path,
    band: int,
    target_grid: dict,
    target_mask: np.ndarray,
    resampling: Resampling,
    max_fill_fraction: float,
) -> tuple[np.ndarray, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        if band < 1 or band > src.count:
            raise ValueError(f"Invalid band {band} for {path}")
        source = src.read(band).astype(np.float32)
        if src.nodata is not None:
            source[source == src.nodata] = np.nan
        destination = np.full(
            (target_grid["height"], target_grid["width"]),
            np.nan,
            dtype=np.float32,
        )
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=target_grid["transform"],
            dst_crs=target_grid["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )
    destination, fraction = fill_inside_mask(
        destination,
        target_mask,
        max_fraction=max_fill_fraction,
        method="nearest",
    )
    return destination, fraction


def write_raster(path: Path, array: np.ndarray, grid: dict, nodata: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array, dtype=np.float32).copy()
    data[~np.isfinite(data)] = nodata
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": grid["width"],
        "height": grid["height"],
        "transform": grid["transform"],
        "crs": grid["crs"],
        "nodata": nodata,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def convert_pressure(values: np.ndarray, unit: str) -> np.ndarray:
    factors = {"Pa": 0.001, "hPa": 0.1, "kPa": 1.0}
    if unit not in factors:
        raise ValueError(f"Unsupported pressure unit: {unit}")
    return values * factors[unit]


def convert_vapor_pressure(values: np.ndarray, unit: str) -> np.ndarray:
    return convert_pressure(values, unit)


def convert_radiation(values: np.ndarray, unit: str) -> np.ndarray:
    factors = {
        "J_m-2_day-1": 1e-6,
        "kJ_m-2_day-1": 1e-3,
        "MJ_m-2_day-1": 1.0,
        "W_m-2": 0.0864,
    }
    if unit not in factors:
        raise ValueError(f"Unsupported radiation unit: {unit}")
    return values * factors[unit]


def convert_precipitation(values: np.ndarray, unit: str, days: int) -> np.ndarray:
    if unit == "mm_day-1":
        return values
    if unit == "mm_month-1":
        return values / days
    raise ValueError(f"Unsupported precipitation unit: {unit}")


def convert_sm(values: np.ndarray, unit: str) -> np.ndarray:
    if unit == "percent":
        return values
    if unit == "fraction":
        return values * 100.0
    if unit == "scaled_percent_0.01":
        return values * 0.01
    raise ValueError(f"Unsupported SM unit: {unit}")


def saturation_vapor_pressure(temp_c: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def delta_svp(temp_c: np.ndarray) -> np.ndarray:
    es = saturation_vapor_pressure(temp_c)
    return 4098.0 * es / np.maximum((temp_c + 237.3) ** 2, 1e-6)


def fao56_et0(
    temp_c: np.ndarray,
    rh: np.ndarray,
    wind2: np.ndarray,
    pressure_kpa: np.ndarray,
    rn_mj: np.ndarray,
    g_mj: np.ndarray,
    es_kpa: np.ndarray | None,
    ea_kpa: np.ndarray | None,
    clip_available_energy: bool,
) -> tuple[np.ndarray, np.ndarray]:
    es_calc = saturation_vapor_pressure(temp_c)
    es = es_calc if es_kpa is None else np.where(
        np.isfinite(es_kpa) & (es_kpa > 0),
        es_kpa,
        es_calc,
    )
    ea_calc = es * np.clip(rh, 0, 100) / 100.0
    ea = ea_calc if ea_kpa is None else np.where(
        np.isfinite(ea_kpa) & (ea_kpa >= 0) & (ea_kpa <= 1.5 * es),
        ea_kpa,
        ea_calc,
    )
    vpd = np.maximum(es - ea, 0.0)
    delta = delta_svp(temp_c)
    gamma = 0.000665 * pressure_kpa
    energy = rn_mj - g_mj
    if clip_available_energy:
        energy = np.maximum(energy, 0.0)
    wind2 = np.maximum(wind2, 0.0)
    denominator = np.maximum(delta + gamma * (1.0 + 0.34 * wind2), 1e-6)
    numerator = (
        0.408 * delta * energy
        + gamma * (900.0 / (temp_c + 273.0)) * wind2 * vpd
    )
    et0 = np.maximum(numerator / denominator, 0.0)
    et0[~np.isfinite(et0)] = np.nan
    return et0.astype(np.float32), vpd.astype(np.float32)


def aggregate_daily_sm(
    cfg: dict,
    year: int,
    month: int,
    shapefile: Path,
) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    days_in_month = calendar.monthrange(year, month)[1]
    daily_files = []
    for day in range(1, days_in_month + 1):
        path = format_path(cfg["paths"]["sm_daily"], year, month, day)
        if path.exists():
            daily_files.append(path)

    if len(daily_files) < int(cfg["quality"]["min_days_per_month"]):
        raise ValueError(
            f"{year}-{month:02d}: only {len(daily_files)} daily SM files"
        )

    source_grid = grid_from_raster(daily_files[0])
    coarse_mask = study_mask(shapefile, source_grid)
    stack = []

    for path in daily_files:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            current_grid = {
                "width": src.width,
                "height": src.height,
                "transform": src.transform,
                "crs": src.crs,
            }
            if not same_grid(current_grid, source_grid):
                aligned = np.full(
                    (source_grid["height"], source_grid["width"]),
                    np.nan,
                    dtype=np.float32,
                )
                reproject(
                    source=data,
                    destination=aligned,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=np.nan,
                    dst_transform=source_grid["transform"],
                    dst_crs=source_grid["crs"],
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                data = aligned
            stack.append(data)

    data_stack = np.stack(stack)
    valid_count = np.sum(np.isfinite(data_stack), axis=0)
    required_count = int(
        np.ceil(days_in_month * float(cfg["quality"]["min_pixel_day_fraction"]))
    )
    with np.errstate(invalid="ignore"):
        monthly = np.nanmean(data_stack, axis=0)
    monthly[valid_count < required_count] = np.nan
    monthly = convert_sm(monthly, cfg["units"]["sm"])
    monthly[~coarse_mask] = np.nan
    monthly, fill_fraction = fill_inside_mask(
        monthly,
        coarse_mask,
        float(cfg["quality"]["max_coarse_fill_fraction"]),
        "linear_nearest",
    )
    sm_min = float(np.nanmin(monthly[coarse_mask]))
    sm_max = float(np.nanmax(monthly[coarse_mask]))
    if not (0 <= sm_min < sm_max <= 100):
        raise ValueError(f"Invalid monthly SM range: {sm_min}, {sm_max}")

    qc = {
        "daily_files": len(daily_files),
        "required_pixel_days": required_count,
        "coarse_fill_fraction": fill_fraction,
        "coarse_min_percent": sm_min,
        "coarse_max_percent": sm_max,
        "coarse_mean_percent": float(np.nanmean(monthly[coarse_mask])),
    }
    return monthly, source_grid, coarse_mask, qc


def reproject_average(
    source: np.ndarray,
    source_grid: dict,
    target_grid: dict,
) -> np.ndarray:
    destination = np.full(
        (target_grid["height"], target_grid["width"]),
        np.nan,
        dtype=np.float32,
    )
    reproject(
        source=source,
        destination=destination,
        src_transform=source_grid["transform"],
        src_crs=source_grid["crs"],
        src_nodata=np.nan,
        dst_transform=target_grid["transform"],
        dst_crs=target_grid["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return destination


class ResidualTransferModel:
    def __init__(self, cfg: dict):
        self.k = int(cfg["model"]["neighbors"])
        self.epsilon = float(cfg["model"]["distance_epsilon"])
        self.batch_size = int(cfg["model"]["batch_size"])
        self.backend = cfg["model"]["backend"]
        self.scaler = StandardScaler()
        self.regression = Ridge(alpha=float(cfg["model"]["ridge_alpha"]))
        self.x_low = None
        self.residuals = None
        self.nn = None
        self.device = None

    def fit(self, x_low: np.ndarray, y_low: np.ndarray):
        x_scaled = self.scaler.fit_transform(x_low)
        self.regression.fit(x_scaled, y_low)
        background = self.regression.predict(x_scaled)
        residuals = y_low - background
        self.x_low = x_scaled.astype(np.float32)
        self.residuals = residuals.astype(np.float32)

        use_torch = (
            self.backend in {"auto", "torch"}
            and torch is not None
            and torch.cuda.is_available()
        )
        if use_torch:
            self.device = torch.device("cuda")
            self.x_low_t = torch.as_tensor(self.x_low, device=self.device)
            self.residuals_t = torch.as_tensor(self.residuals, device=self.device)
        else:
            self.backend = "sklearn"
            self.nn = NearestNeighbors(
                n_neighbors=self.k,
                metric="euclidean",
                n_jobs=-1,
            )
            self.nn.fit(self.x_low)

    def predict(self, x_high: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x_high).astype(np.float32)
        background = self.regression.predict(x_scaled).astype(np.float32)
        transferred = np.empty(len(x_scaled), dtype=np.float32)

        if self.backend == "sklearn":
            for start in tqdm(range(0, len(x_scaled), self.batch_size)):
                end = min(start + self.batch_size, len(x_scaled))
                distances, indices = self.nn.kneighbors(x_scaled[start:end])
                weights = 1.0 / (distances ** 2 + self.epsilon)
                weights /= np.sum(weights, axis=1, keepdims=True)
                transferred[start:end] = np.sum(
                    weights * self.residuals[indices],
                    axis=1,
                )
        else:
            with torch.no_grad():
                for start in tqdm(range(0, len(x_scaled), self.batch_size)):
                    end = min(start + self.batch_size, len(x_scaled))
                    batch = torch.as_tensor(
                        x_scaled[start:end],
                        device=self.device,
                    )
                    distances = torch.cdist(batch, self.x_low_t)
                    values, indices = torch.topk(
                        distances,
                        self.k,
                        dim=1,
                        largest=False,
                    )
                    weights = 1.0 / (values ** 2 + self.epsilon)
                    weights /= torch.sum(weights, dim=1, keepdim=True)
                    residuals = self.residuals_t[indices]
                    transferred[start:end] = torch.sum(
                        weights * residuals,
                        dim=1,
                    ).cpu().numpy()

        return background + transferred


def build_feature_stack(
    feature_map: dict[str, np.ndarray],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack([feature_map[name] for name in FEATURES], axis=-1)
    valid = mask & np.all(np.isfinite(stack), axis=-1)
    return stack, valid


def process_month(
    cfg: dict,
    logger: logging.Logger,
    target_grid: dict,
    target_mask: np.ndarray,
    year: int,
    month: int,
):
    output_dir = Path(cfg["paths"]["output_root"]) / f"{year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    shapefile = Path(cfg["paths"]["study_area"])

    sm_low, coarse_grid, coarse_mask, qc = aggregate_daily_sm(
        cfg,
        year,
        month,
        shapefile,
    )

    paths = {
        name: format_path(template, year, month)
        for name, template in cfg["paths"]["monthly_inputs"].items()
    }
    meteor_path = paths["meteorology"]
    meteor_bands = cfg["bands"]["meteorology"]
    max_input_fill = float(cfg["quality"]["max_input_fill_fraction"])
    fill_fractions = {}

    meteor = {}
    for name in ("temperature", "humidity", "wind2", "pressure", "rn", "g", "es", "ea"):
        band = meteor_bands.get(name)
        if band is None and name in {"es", "ea"}:
            meteor[name] = None
            continue
        data, fraction = read_band(
            meteor_path,
            int(band),
            target_grid,
            target_mask,
            Resampling.bilinear,
            max_input_fill,
        )
        meteor[name] = data
        fill_fractions[f"meteorology_{name}"] = fraction

    days = calendar.monthrange(year, month)[1]
    pressure = convert_pressure(meteor["pressure"], cfg["units"]["pressure"])
    rn = convert_radiation(meteor["rn"], cfg["units"]["rn"])
    g = convert_radiation(meteor["g"], cfg["units"]["g"])
    es = None if meteor["es"] is None else convert_vapor_pressure(
        meteor["es"],
        cfg["units"]["es"],
    )
    ea = None if meteor["ea"] is None else convert_vapor_pressure(
        meteor["ea"],
        cfg["units"]["ea"],
    )

    et0, vpd = fao56_et0(
        meteor["temperature"],
        meteor["humidity"],
        meteor["wind2"],
        pressure,
        rn,
        g,
        es,
        ea,
        bool(cfg["model"]["clip_available_energy"]),
    )

    feature_specs = {
        "ndvi": ("ndvi", int(cfg["bands"]["ndvi"])),
        "precipitation": ("precipitation", int(cfg["bands"]["precipitation"])),
        "slope": ("terrain", int(cfg["bands"]["slope"])),
        "soil_texture": ("soil", int(cfg["bands"]["soil_texture"])),
        "lst": ("lst", int(cfg["bands"]["lst"])),
        "albedo": ("albedo", int(cfg["bands"]["albedo"])),
    }

    feature_map = {
        "temperature": meteor["temperature"],
        "humidity": meteor["humidity"],
        "et0": et0,
        "vpd": vpd,
    }

    for feature, (path_key, band) in feature_specs.items():
        data, fraction = read_band(
            paths[path_key],
            band,
            target_grid,
            target_mask,
            Resampling.bilinear,
            max_input_fill,
        )
        feature_map[feature] = data
        fill_fractions[feature] = fraction

    feature_map["precipitation"] = convert_precipitation(
        feature_map["precipitation"],
        cfg["units"]["precipitation"],
        days,
    )
    feature_map["water_availability"] = (
        feature_map["precipitation"] - feature_map["et0"]
    )

    high_stack, high_valid = build_feature_stack(feature_map, target_mask)
    if high_valid.sum() != target_mask.sum():
        raise ValueError("High-resolution feature stack is incomplete")

    low_feature_map = {}
    for name in FEATURES:
        low = reproject_average(feature_map[name], target_grid, coarse_grid)
        low[~coarse_mask] = np.nan
        low_feature_map[name] = low

    low_stack, low_valid = build_feature_stack(low_feature_map, coarse_mask)
    low_valid &= np.isfinite(sm_low)
    if low_valid.sum() < int(cfg["quality"]["min_training_pixels"]):
        raise ValueError(f"Too few training pixels: {low_valid.sum()}")

    rng = np.random.default_rng(int(cfg["model"]["random_seed"]))
    low_indices = np.flatnonzero(low_valid)
    max_samples = int(cfg["model"]["max_samples"])
    if len(low_indices) > max_samples:
        low_indices = rng.choice(low_indices, max_samples, replace=False)

    x_low = low_stack.reshape(-1, len(FEATURES))[low_indices]
    y_low = sm_low.reshape(-1)[low_indices]
    x_high = high_stack[high_valid]

    model = ResidualTransferModel(cfg)
    model.fit(x_low, y_low)
    predictions = model.predict(x_high)

    sm_high = np.full(target_mask.shape, np.nan, dtype=np.float32)
    sm_high[high_valid] = predictions
    sm_min = float(np.nanmin(sm_low[coarse_mask]))
    sm_max = float(np.nanmax(sm_low[coarse_mask]))
    sm_high = np.clip(sm_high, sm_min, sm_max)
    sm_high[~target_mask] = np.nan

    missing_fraction = float(
        np.sum(target_mask & ~np.isfinite(sm_high)) / max(target_mask.sum(), 1)
    )
    if missing_fraction > float(cfg["quality"]["max_output_missing_fraction"]):
        raise ValueError(f"Output missing fraction: {missing_fraction:.4%}")
    if missing_fraction > 0:
        sm_high, _ = fill_inside_mask(
            sm_high,
            target_mask,
            float(cfg["quality"]["max_output_missing_fraction"]),
            "nearest",
        )

    nodata = float(cfg["output"]["nodata"])
    write_raster(
        output_dir / f"SM_PM_500m_{year}_{month:02d}.tif",
        sm_high,
        target_grid,
        nodata,
    )
    write_raster(
        output_dir / f"ET0_FAO56_500m_{year}_{month:02d}.tif",
        np.where(target_mask, et0, np.nan),
        target_grid,
        nodata,
    )
    write_raster(
        output_dir / f"G_FAO56_500m_{year}_{month:02d}.tif",
        np.where(target_mask, g, np.nan),
        target_grid,
        nodata,
    )
    write_raster(
        output_dir / f"WAI_500m_{year}_{month:02d}.tif",
        np.where(target_mask, feature_map["water_availability"], np.nan),
        target_grid,
        nodata,
    )

    if bool(cfg["output"]["write_soil_water"]):
        soil_water = sm_high / 100.0 * float(cfg["output"]["soil_depth_m"]) * 1000.0
        write_raster(
            output_dir / f"Soil_Water500m_{year}_{month:02d}.tif",
            soil_water,
            target_grid,
            nodata,
        )

    qc.update({
        "year": year,
        "month": month,
        "training_pixels": int(len(x_low)),
        "target_pixels": int(target_mask.sum()),
        "output_missing_fraction": missing_fraction,
        "output_min_percent": float(np.nanmin(sm_high[target_mask])),
        "output_max_percent": float(np.nanmax(sm_high[target_mask])),
        "output_mean_percent": float(np.nanmean(sm_high[target_mask])),
        "input_fill_fractions": fill_fractions,
        "ridge_alpha": float(cfg["model"]["ridge_alpha"]),
        "neighbors": int(cfg["model"]["neighbors"]),
        "backend": model.backend,
    })

    with open(
        output_dir / f"QC_{year}_{month:02d}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(qc, f, indent=2)

    logger.info(
        "%04d-%02d complete | training=%d | mean=%.4f",
        year,
        month,
        len(x_low),
        qc["output_mean_percent"],
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_root = Path(cfg["paths"]["output_root"])
    logger = setup_logger(output_root)

    target_reference = Path(cfg["paths"]["target_reference"])
    target_grid = grid_from_raster(target_reference)
    target_mask = study_mask(Path(cfg["paths"]["study_area"]), target_grid)

    for year, month in month_iterator(cfg["period"]["start"], cfg["period"]["end"]):
        try:
            process_month(
                cfg,
                logger,
                target_grid,
                target_mask,
                year,
                month,
            )
        except Exception:
            logger.exception("%04d-%02d failed", year, month)
            if bool(cfg["runtime"]["stop_on_error"]):
                raise


if __name__ == "__main__":
    main()
