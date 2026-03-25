"""
helpers/models.py
=================
All ML model training, evaluation, comparison, spatial prediction,
and validation functions for the WaPOR ETa downscaling pipeline.

Imported and called by ml_pipeline.py — do not run directly.
"""

import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import rasterio
from rasterio.warp import reproject as rasterio_reproject, Resampling
from rasterio.mask import mask as rasterio_mask
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Matplotlib style ─────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"       : 120,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.size"        : 10,
})

# ── Shared constants ─────────────────────────────────────────
FEATURE_COLS = ["NDVI", "EVI", "SAVI", "NDWI", "NDMI", "LST"]
TARGET_COL   = "ETa"
NODATA_VALUE = -9999
RANDOM_STATE = 42
TEST_SIZE    = 0.2
MONTH_LABEL  = "oct2023"   # suffix used in all output file names

# Models that require StandardScaler-normalised input
SCALED_MODELS = {"Linear Regression", "MLP"}

# Maps model display name → file stem used in output paths
NAME_TO_FILE = {
    "Linear Regression": "LinearRegression",
    "Random Forest"    : "RandomForest",
    "XGBoost"          : "XGBoost",
    "MLP"              : "MLP",
}


# =============================================================
# UTILITIES
# =============================================================

def print_section(title: str) -> None:
    """Print a formatted section banner for console readability."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard regression metrics.

    Parameters
    ----------
    y_true : array-like   Observed / reference values
    y_pred : array-like   Model predictions

    Returns
    -------
    dict
        R2        : coefficient of determination
        RMSE      : root mean squared error
        rRMSE_pct : RMSE as % of the mean observed value
        MAE       : mean absolute error
        Bias      : mean signed error (positive = over-prediction)
    """
    r2        = r2_score(y_true, y_pred)
    rmse      = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rrmse_pct = (rmse / float(np.mean(y_true))) * 100.0
    mae       = float(mean_absolute_error(y_true, y_pred))
    bias      = float(np.mean(y_pred - y_true))
    return {"R2": r2, "RMSE": rmse, "rRMSE_pct": rrmse_pct,
            "MAE": mae, "Bias": bias}


def save_model(model, name: str, out_dir: str) -> str:
    """
    Persist a fitted estimator to disk using joblib.

    Parameters
    ----------
    model   : fitted sklearn / XGBoost estimator
    name    : short identifier used in the filename
    out_dir : directory where the .pkl file is written

    Returns
    -------
    str  Absolute path of the saved file
    """
    path = os.path.join(out_dir, f"model_{name}_{MONTH_LABEL}.pkl")
    joblib.dump(model, path)
    print(f"    [saved] {path}")
    return path


# =============================================================
# STEP 1 — PREPROCESS TRAINING DATA
# =============================================================

def preprocess_training_data(df: pd.DataFrame, out_dir: str = ".") -> tuple:
    """
    Clean the in-memory training DataFrame and return feature / target arrays.

    Data arrives directly from extract_training_data() — no CSV is involved.
    Any remaining NaN or fill values (-9999) are dropped as a safety check.

    Parameters
    ----------
    df      : pd.DataFrame  Raw pixel-pair DataFrame with columns:
                            NDVI, EVI, SAVI, NDWI, NDMI, LST, ETa
    out_dir : str           Directory where the correlation plot is saved.

    Returns
    -------
    X        : pd.DataFrame  Feature matrix  (n_samples × 6)
    y        : pd.Series     Target vector   (n_samples,)
    df_clean : pd.DataFrame  Cleaned DataFrame (for EDA / inspection)
    """
    print_section("STEP 5 — Preprocess Training Data")
    print(f"  Input shape : {df.shape}")

    # Verify required columns
    required = FEATURE_COLS + [TARGET_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")

    df = df[required].copy()

    # Drop any residual NaN or fill values
    df.replace(NODATA_VALUE, np.nan, inplace=True)
    n_before = len(df)
    df.dropna(inplace=True)
    print(f"  Rows dropped (NaN / fill) : {n_before - len(df)}")
    print(f"  Clean shape : {df.shape}")

    # Descriptive statistics
    print("\n  --- Feature & Target Statistics ---")
    print(df.describe().round(4).to_string())

    # Pearson correlation of each feature against ETa
    print("\n  --- Pearson Correlation vs ETa ---")
    corr = (df.corr()[[TARGET_COL]]
              .drop(TARGET_COL)
              .sort_values(TARGET_COL, ascending=False))
    print(corr.round(4).to_string())

    # Correlation heat-map
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    corr_path = os.path.join(out_dir, "correlation_matrix.png")
    plt.savefig(corr_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {corr_path}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y, df


# =============================================================
# STEP 2 — TRAIN/TEST SPLIT AND FEATURE SCALING
# =============================================================

def split_and_scale(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Split data 80/20 and fit a StandardScaler on the training set only.

    Tree-based models (Random Forest, XGBoost) are scale-invariant and
    receive raw values.  Linear Regression and MLP use scaled values.
    The scaler is fitted on X_train only to prevent data leakage.

    Parameters
    ----------
    X : pd.DataFrame  Feature matrix
    y : pd.Series     Target vector

    Returns
    -------
    dict with keys:
        X_train_raw, X_test_raw  — unscaled numpy arrays
        X_train_sc,  X_test_sc   — scaled numpy arrays
        y_train,     y_test      — target numpy arrays
        scaler                   — fitted StandardScaler
    """
    print_section("STEP 2 — Train/Test Split & Feature Scaling")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    print(f"  Train : {X_train_raw.shape[0]} samples")
    print(f"  Test  : {X_test_raw.shape[0]} samples")

    # Fit scaler on training split only
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train_raw)
    X_test_sc   = scaler.transform(X_test_raw)

    print("  StandardScaler fitted on training set.")
    print(f"  Means  : {dict(zip(FEATURE_COLS, scaler.mean_.round(3)))}")
    print(f"  Stdevs : {dict(zip(FEATURE_COLS, scaler.scale_.round(3)))}")

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw" : X_test_raw,
        "X_train_sc" : X_train_sc,
        "X_test_sc"  : X_test_sc,
        "y_train"    : y_train,
        "y_test"     : y_test,
        "scaler"     : scaler,
    }


# =============================================================
# STEP 3 — MODEL TRAINING
# =============================================================

def train_linear_regression(data: dict, out_dir: str) -> dict:
    """
    Train an Ordinary Least Squares Linear Regression baseline.

    Uses StandardScaler-normalised features so coefficients are
    comparable across variables with different physical units.

    Parameters
    ----------
    data    : dict  Output of split_and_scale()
    out_dir : str   Directory for saving the fitted model

    Returns
    -------
    dict  keys: model, metrics, name, train_time, y_pred_test
    """
    print_section("STEP 3a — Linear Regression (Baseline)")

    model = LinearRegression()
    t0 = time.time()
    try:
        model.fit(data["X_train_sc"], data["y_train"])
    except Exception as e:
        print(f"[ERROR] LinearRegression failed: {e}")
        traceback.print_exc()
        return {}
    train_time = time.time() - t0

    y_pred  = model.predict(data["X_test_sc"])
    metrics = compute_metrics(data["y_test"], y_pred)

    print(f"  Train time : {train_time:.2f} s")
    print(f"  R²         : {metrics['R2']:.4f}")
    print(f"  RMSE       : {metrics['RMSE']:.4f} mm/month")
    print(f"  rRMSE      : {metrics['rRMSE_pct']:.2f} %")

    # Coefficients table
    coef_df = (pd.DataFrame({"Feature": FEATURE_COLS,
                              "Coefficient": model.coef_.round(6)})
                 .sort_values("Coefficient", ascending=False))
    print("\n  Coefficients (scaled features):")
    print(coef_df.to_string(index=False))
    print(f"  Intercept : {model.intercept_:.4f}")

    # save_model(model, "LinearRegression", out_dir)

    return {"model": model, "metrics": metrics,
            "name": "Linear Regression",
            "train_time": train_time, "y_pred_test": y_pred}


def train_random_forest(data: dict, out_dir: str) -> dict:
    """
    Train a Random Forest ensemble regressor.

    Tree-based; uses raw (unscaled) features.
    Feature importances are computed via mean impurity decrease.

    Parameters
    ----------
    data    : dict  Output of split_and_scale()
    out_dir : str   Directory for saving the fitted model

    Returns
    -------
    dict  keys: model, metrics, name, train_time, y_pred_test
    """
    print_section("STEP 3b — Random Forest")

    model = RandomForestRegressor(
        n_estimators    =200,
        max_depth       =None,      # grow until pure leaves
        min_samples_split=5,
        min_samples_leaf =2,
        max_features    ="sqrt",    # sqrt(n_features) per split
        n_jobs          =-1,        # parallelise across all cores
        random_state    =RANDOM_STATE,
    )
    t0 = time.time()
    try:
        model.fit(data["X_train_raw"], data["y_train"])
    except Exception as e:
        print(f"[ERROR] RandomForest failed: {e}")
        traceback.print_exc()
        return {}
    train_time = time.time() - t0

    y_pred  = model.predict(data["X_test_raw"])
    metrics = compute_metrics(data["y_test"], y_pred)

    print(f"  Train time : {train_time:.2f} s")
    print(f"  R²         : {metrics['R2']:.4f}")
    print(f"  RMSE       : {metrics['RMSE']:.4f} mm/month")
    print(f"  rRMSE      : {metrics['rRMSE_pct']:.2f} %")

    imp_df = (pd.DataFrame({"Feature": FEATURE_COLS,
                             "Importance": model.feature_importances_.round(6)})
                .sort_values("Importance", ascending=False))
    print("\n  Feature Importances (mean impurity decrease):")
    print(imp_df.to_string(index=False))

    # Bar chart of importances
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp_df["Feature"], imp_df["Importance"], color="steelblue")
    ax.set_xlabel("Importance (mean impurity decrease)")
    ax.set_title("Random Forest — Feature Importances")
    ax.invert_yaxis()
    fig.tight_layout()
    rf_imp_path = os.path.join(out_dir, "rf_feature_importance.png")
    plt.savefig(rf_imp_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {rf_imp_path}")

    # save_model(model, "RandomForest", out_dir)

    return {"model": model, "metrics": metrics,
            "name": "Random Forest",
            "train_time": train_time, "y_pred_test": y_pred}


def train_xgboost(data: dict, out_dir: str) -> dict:
    """
    Train an XGBoost gradient-boosted tree regressor.

    Hyperparameters: n_estimators=200, max_depth=6, lr=0.05,
    subsample=0.8, colsample_bytree=0.8.
    Tree-based; uses raw (unscaled) features.

    Parameters
    ----------
    data    : dict  Output of split_and_scale()
    out_dir : str   Directory for saving the fitted model

    Returns
    -------
    dict  keys: model, metrics, name, train_time, y_pred_test
    """
    print_section("STEP 3c — XGBoost")

    model = xgb.XGBRegressor(
        n_estimators    =200,
        max_depth       =6,
        learning_rate   =0.05,
        subsample       =0.8,
        colsample_bytree=0.8,
        random_state    =RANDOM_STATE,
        n_jobs          =-1,
        verbosity       =0,
    )
    t0 = time.time()
    try:
        model.fit(
            data["X_train_raw"], data["y_train"],
            eval_set=[(data["X_test_raw"], data["y_test"])],
            verbose=False,
        )
    except Exception as e:
        print(f"[ERROR] XGBoost failed: {e}")
        traceback.print_exc()
        return {}
    train_time = time.time() - t0

    y_pred  = model.predict(data["X_test_raw"])
    metrics = compute_metrics(data["y_test"], y_pred)

    print(f"  Train time : {train_time:.2f} s")
    print(f"  R²         : {metrics['R2']:.4f}")
    print(f"  RMSE       : {metrics['RMSE']:.4f} mm/month")
    print(f"  rRMSE      : {metrics['rRMSE_pct']:.2f} %")

    # Gain-based feature importances from the booster
    imp = model.get_booster().get_score(importance_type="gain")
    imp_df = (pd.DataFrame({"Feature": list(imp.keys()),
                             "Gain": list(imp.values())})
                .sort_values("Gain", ascending=False))
    print("\n  Feature Importances (gain):")
    print(imp_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp_df["Feature"], imp_df["Gain"], color="darkorange")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("XGBoost — Feature Importances")
    ax.invert_yaxis()
    fig.tight_layout()
    xgb_imp_path = os.path.join(out_dir, "xgb_feature_importance.png")
    plt.savefig(xgb_imp_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {xgb_imp_path}")

    # save_model(model, "XGBoost", out_dir)

    return {"model": model, "metrics": metrics,
            "name": "XGBoost",
            "train_time": train_time, "y_pred_test": y_pred}


def train_mlp(data: dict, out_dir: str) -> dict:
    """
    Train a Multi-Layer Perceptron (MLP) neural network regressor.

    Architecture: 128 → 64 neurons, ReLU activation, Adam solver.
    Uses StandardScaler-normalised features (MLP is scale-sensitive).
    Early stopping prevents overfitting; 10 % of training data is
    held out internally as a validation set.

    Parameters
    ----------
    data    : dict  Output of split_and_scale()
    out_dir : str   Directory for saving the fitted model

    Returns
    -------
    dict  keys: model, metrics, name, train_time, y_pred_test
    """
    print_section("STEP 3d — MLP Neural Network")

    model = MLPRegressor(
        hidden_layer_sizes =(128, 64),
        activation         ="relu",
        solver             ="adam",
        learning_rate_init =1e-3,
        max_iter           =500,
        early_stopping     =True,
        validation_fraction=0.1,
        n_iter_no_change   =20,
        random_state       =RANDOM_STATE,
    )
    t0 = time.time()
    try:
        model.fit(data["X_train_sc"], data["y_train"])
    except Exception as e:
        print(f"[ERROR] MLP failed: {e}")
        traceback.print_exc()
        return {}
    train_time = time.time() - t0

    y_pred  = model.predict(data["X_test_sc"])
    metrics = compute_metrics(data["y_test"], y_pred)

    print(f"  Train time : {train_time:.2f} s")
    print(f"  Iterations : {model.n_iter_}")
    print(f"  R²         : {metrics['R2']:.4f}")
    print(f"  RMSE       : {metrics['RMSE']:.4f} mm/month")
    print(f"  rRMSE      : {metrics['rRMSE_pct']:.2f} %")

    # save_model(model, "MLP", out_dir)

    return {"model": model, "metrics": metrics,
            "name": "MLP",
            "train_time": train_time, "y_pred_test": y_pred}


# =============================================================
# STEP 4 — MODEL COMPARISON
# =============================================================

def compare_models(results: list, data: dict, out_dir: str) -> str:
    """
    Build a metrics summary table, grouped bar chart, and a 2×2
    predicted-vs-observed scatter grid. Returns the best model name.

    Parameters
    ----------
    results : list of dicts   Return values from train_* functions
    data    : dict            Output of split_and_scale() (for y_test)
    out_dir : str             Directory for saved figures / CSV

    Returns
    -------
    str  Display name of the best-performing model (highest R²)
    """
    print_section("STEP 4 — Model Comparison")

    # Build summary table
    rows = []
    for r in results:
        if not r:
            continue
        m = r["metrics"]
        rows.append({
            "Model"           : r["name"],
            "R²"              : round(m["R2"],        4),
            "RMSE (mm/month)" : round(m["RMSE"],      4),
            "rRMSE (%)"       : round(m["rRMSE_pct"], 2),
            "Train time (s)"  : round(r["train_time"], 2),
        })

    summary_df = pd.DataFrame(rows)
    print("\n  --- Performance Summary (test set) ---")
    print(summary_df.to_string(index=False))

    csv_out = os.path.join(out_dir, "model_comparison_summary.csv")
    summary_df.to_csv(csv_out, index=False)
    print(f"  [saved] {csv_out}")

    # Grouped bar chart — R², RMSE, rRMSE
    colors      = ["steelblue", "darkorange", "seagreen", "crimson"]
    model_names = summary_df["Model"].tolist()
    fig, axes   = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (col, label) in zip(axes, [("R²", "R²"),
                                        ("RMSE (mm/month)", "RMSE"),
                                        ("rRMSE (%)", "rRMSE (%)")]):
        bars = ax.bar(model_names, summary_df[col],
                      color=colors[:len(model_names)])
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xticklabels(model_names, rotation=20, ha="right")
        for bar, val in zip(bars, summary_df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Comparison — Test Set Metrics", fontweight="bold")
    fig.tight_layout()
    bar_path = os.path.join(out_dir, "model_comparison_barchart.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {bar_path}")

    # 2×2 scatter: predicted vs observed
    valid   = [r for r in results if r]
    n       = len(valid)
    nrows   = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 5 * nrows))
    # axes is always an ndarray (ncols=2 is fixed), flatten handles 1-D and 2-D
    axes_flat = np.array(axes).flatten()
    y_test    = data["y_test"]

    for ax, r in zip(axes_flat, valid):
        y_pred = r["y_pred_test"]
        m      = r["metrics"]
        ax.scatter(y_test, y_pred, alpha=0.4, s=15,
                   color="steelblue", edgecolors="none")
        lims = [min(y_test.min(), y_pred.min()),
                max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=1.2, label="1:1")
        ax.set_xlabel("Observed ETa (mm/month)")
        ax.set_ylabel("Predicted ETa (mm/month)")
        ax.set_title(r["name"])
        ax.legend(fontsize=8)
        txt = (f"R² = {m['R2']:.3f}\n"
               f"RMSE = {m['RMSE']:.2f} mm\n"
               f"rRMSE = {m['rRMSE_pct']:.1f}%")
        ax.text(0.05, 0.93, txt, transform=ax.transAxes, fontsize=8,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="lightyellow", alpha=0.8))

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Predicted vs Observed ETa — Test Set", fontweight="bold")
    fig.tight_layout()
    scat_path = os.path.join(out_dir, "scatter_predicted_vs_observed.png")
    plt.savefig(scat_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {scat_path}")

    # Identify best model
    best_idx  = summary_df["R²"].idxmax()
    best_name = summary_df.loc[best_idx, "Model"]
    print(f"\n  Best model : {best_name}  "
          f"(R²={summary_df.loc[best_idx,'R²']:.4f})")
    return best_name


# =============================================================
# STEP 5 — APPLY MODELS TO 30 m RASTER
# =============================================================

def predict_raster(
    raster_path: str,
    model_results: list,
    scaler: StandardScaler,
    best_name: str,
    out_dir: str,
    aoi_path: str = None,
) -> None:
    """
    Apply every trained model to the 30 m Landsat index stack and
    write one downscaled ETa GeoTIFF per model.

    Only valid pixels (all 6 bands finite and not NoData) are passed
    to the model; invalid pixels are filled with −9999 in the output.

    If aoi_path is provided, each output raster is clipped to the AOI
    boundary (the original study-area polygon, without any buffer) so
    the final maps match the exact scheme extent.

    Parameters
    ----------
    raster_path   : str            Path to 6-band 30 m predictor GeoTIFF
                                   Band order: NDVI, EVI, SAVI, NDWI, NDMI, LST
    model_results : list of dicts  Return values of train_* functions
    scaler        : StandardScaler Fitted scaler (applied to LR & MLP inputs)
    best_name     : str            Name of best model (flagged in console)
    out_dir       : str            Output directory for GeoTIFFs
    aoi_path      : str or None    Path to AOI GeoJSON for final clipping.
                                   If None, outputs are not clipped.
    """
    print_section("STEP 5 — Apply Models to 30 m Raster")
    print(f"  Input: {raster_path}")

    try:
        with rasterio.open(raster_path) as src:
            profile    = src.profile.copy()
            n_bands    = src.count
            rows, cols = src.height, src.width
            nodata_in  = src.nodata
            data_stack = src.read()   # shape: (n_bands, rows, cols)
            print(f"  Shape  : {rows} × {cols}")
            print(f"  Bands  : {n_bands}")
            print(f"  CRS    : {src.crs}")
    except Exception as e:
        print(f"[ERROR] Cannot read raster: {e}")
        traceback.print_exc()
        return

    if n_bands < 6:
        print(f"[ERROR] Expected ≥6 bands, found {n_bands}.")
        return

    # Build valid-pixel mask — all 6 bands must be finite and non-NoData
    valid_mask = np.ones((rows, cols), dtype=bool)
    nd = nodata_in if nodata_in is not None else NODATA_VALUE
    for b in range(6):
        band = data_stack[b]
        valid_mask &= np.isfinite(band) & (band != nd)

    n_valid = int(valid_mask.sum())
    print(f"  Valid pixels : {n_valid} / {rows * cols} "
          f"({100 * n_valid / (rows * cols):.1f} %)")

    # Extract valid pixels → (n_valid, 6)
    X_raster   = data_stack[:6].transpose(1, 2, 0)   # (rows, cols, 6)
    X_valid    = X_raster[valid_mask]
    X_valid_sc = scaler.transform(X_valid)            # pre-scaled for LR/MLP

    # Output profile — single-band float32
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32",
                       nodata=NODATA_VALUE, compress="lzw")

    for r in model_results:
        if not r:
            continue
        model_name = r["name"]
        model      = r["model"]
        tag        = NAME_TO_FILE.get(model_name, model_name.replace(" ", ""))

        print(f"\n  Predicting [{model_name}] ...")

        # Use scaled input for LR / MLP; raw for RF / XGBoost
        X_in = X_valid_sc if model_name in SCALED_MODELS else X_valid

        try:
            y_hat = model.predict(X_in).astype(np.float32)
        except Exception as e:
            print(f"  [ERROR] Prediction failed: {e}")
            continue

        # Reconstruct full spatial grid
        pred_grid = np.full((rows, cols), NODATA_VALUE, dtype=np.float32)
        pred_grid[valid_mask] = y_hat

        out_path = os.path.join(
            out_dir, f"downscaled_ETa_30m_{tag}_{MONTH_LABEL}.tif"
        )
        try:
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(pred_grid, 1)
        except Exception as e:
            print(f"  [ERROR] Write failed: {e}")
            continue

        # Clip output to the original AOI boundary (no buffer)
        if aoi_path:
            try:
                aoi_gdf  = gpd.read_file(aoi_path).dissolve().to_crs(out_profile["crs"])
                aoi_geom = [aoi_gdf.geometry.iloc[0]]

                with rasterio.open(out_path) as src:
                    clipped, clip_transform = rasterio_mask(
                        src, aoi_geom, crop=True, nodata=NODATA_VALUE
                    )
                    clip_profile = src.profile.copy()
                    clip_profile.update(
                        height   =clipped.shape[1],
                        width    =clipped.shape[2],
                        transform=clip_transform,
                    )

                # Overwrite the file with the clipped version
                with rasterio.open(out_path, "w", **clip_profile) as dst:
                    dst.write(clipped)

                print(f"  Clipped to AOI boundary.")
            except Exception as e:
                print(f"  [WARN] AOI clipping failed: {e} — unclipped raster kept.")

        flag = "  ← BEST MODEL" if model_name == best_name else ""
        print(f"  [saved] {out_path}{flag}")

    print("\n  All downscaled maps written.")


# =============================================================
# STEP 6 — VALIDATION AGAINST WaPOR LEVEL 3 (20 m)
# =============================================================

def validate_against_l3(
    l3_path: str,
    out_dir: str,
    model_names: list,
) -> None:
    """
    Validate each downscaled 30 m ETa map against the WaPOR L3
    AETI product, resampled from 20 m to 30 m via bilinear interpolation.

    NOTE: WaPOR L3 is an independent model-based product, not
    in-situ ground truth.  Both L1 and L3 derive from the same
    ETLook model family and carry inherent uncertainty.  These
    metrics reflect inter-product agreement, not absolute accuracy.

    Parameters
    ----------
    l3_path     : str   Path to WaPOR L3 AETI GeoTIFF (20 m)
    out_dir     : str   Directory that holds downscaled GeoTIFFs
    model_names : list  Display names of the models to validate
    """
    print_section("STEP 6 — Validation vs WaPOR L3 (20 m)")
    print(f"  L3 reference : {l3_path}")
    print("  NOTE: L3 is a reference product, not ground truth.")

    try:
        with rasterio.open(l3_path) as s:
            l3_nodata = s.nodata
            print(f"  L3 shape  : {s.height} × {s.width}")
            print(f"  L3 CRS    : {s.crs}")
    except Exception as e:
        print(f"[ERROR] Cannot read L3 raster: {e}")
        return

    val_rows = []
    fig_val, axes_val = plt.subplots(2, 2, figsize=(10, 10),
                                     constrained_layout=True)
    axes_flat = axes_val.flatten()
    plot_idx  = 0

    for model_name in model_names:
        tag       = NAME_TO_FILE.get(model_name, model_name.replace(" ", ""))
        pred_path = os.path.join(
            out_dir, f"downscaled_ETa_30m_{tag}_{MONTH_LABEL}.tif"
        )

        if not os.path.exists(pred_path):
            print(f"  [SKIP] Not found: {pred_path}")
            continue

        # Read the 30 m downscaled map
        with rasterio.open(pred_path) as pred_src:
            pred_arr       = pred_src.read(1).astype(np.float64)
            pred_nd        = pred_src.nodata
            pred_height    = pred_src.height
            pred_width     = pred_src.width
            pred_transform = pred_src.transform
            pred_crs       = pred_src.crs

        # Warp L3 onto the 30 m prediction grid (bilinear)
        l3_on_30m = np.empty((pred_height, pred_width), dtype=np.float64)
        with rasterio.open(l3_path) as l3_src:
            rasterio_reproject(
                source        =rasterio.band(l3_src, 1),
                destination   =l3_on_30m,
                src_transform =l3_src.transform,
                src_crs       =l3_src.crs,
                dst_transform =pred_transform,
                dst_crs       =pred_crs,
                resampling    =Resampling.bilinear,
            )

        # Valid-pixel mask: both arrays must be finite and non-NoData
        l3_nd  = l3_nodata  if l3_nodata  is not None else NODATA_VALUE
        p_nd   = pred_nd    if pred_nd    is not None else NODATA_VALUE
        valid  = (np.isfinite(pred_arr)  & (pred_arr  != p_nd)  &
                  np.isfinite(l3_on_30m) & (l3_on_30m != l3_nd) &
                  (l3_on_30m > 0))

        y_pred_v = pred_arr[valid]
        y_l3_v   = l3_on_30m[valid]

        if len(y_pred_v) < 10:
            print(f"  [WARN] Too few valid pixels for {model_name}: {len(y_pred_v)}")
            continue

        m = compute_metrics(y_l3_v, y_pred_v)
        val_rows.append({
            "Model"          : model_name,
            "R² (vs L3)"     : round(m["R2"],        4),
            "RMSE (mm/month)": round(m["RMSE"],      4),
            "MAE (mm/month)" : round(m["MAE"],        4),
            "Bias (mm/month)": round(m["Bias"],       4),
            "n_pixels"       : len(y_pred_v),
        })

        # Scatter plot
        if plot_idx < 4:
            ax = axes_flat[plot_idx]
            ax.scatter(y_l3_v, y_pred_v, alpha=0.3, s=10,
                       color="steelblue", edgecolors="none")
            lims = [min(y_l3_v.min(), y_pred_v.min()),
                    max(y_l3_v.max(), y_pred_v.max())]
            ax.plot(lims, lims, "r--", lw=1.2, label="1:1")
            ax.set_xlabel("WaPOR L3 ETa (mm/month)")
            ax.set_ylabel("Downscaled ETa (mm/month)")
            ax.set_title(model_name)
            ax.legend(fontsize=8)
            txt = (f"R² = {m['R2']:.3f}\n"
                   f"RMSE = {m['RMSE']:.2f} mm\n"
                   f"MAE = {m['MAE']:.2f} mm\n"
                   f"Bias = {m['Bias']:+.2f} mm")
            ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=8,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="lightyellow", alpha=0.8))
            plot_idx += 1

    for ax in axes_flat[plot_idx:]:
        ax.set_visible(False)

    fig_val.suptitle(
        "Validation: Downscaled ETa vs WaPOR L3\n"
        "(L3 is a model product — not independent ground truth)",
        fontweight="bold",
    )
    val_fig = os.path.join(out_dir, "validation_scatter_vs_L3.png")
    plt.savefig(val_fig, bbox_inches="tight")
    plt.close(fig_val)
    print(f"  [saved] {val_fig}")

    if val_rows:
        val_df = pd.DataFrame(val_rows)
        print("\n  --- Validation Summary (vs WaPOR L3) ---")
        print(val_df.to_string(index=False))
        csv_out = os.path.join(out_dir, "validation_summary_vs_L3.csv")
        val_df.to_csv(csv_out, index=False)
        print(f"  [saved] {csv_out}")
    else:
        print("  [WARN] No valid validation results.")
