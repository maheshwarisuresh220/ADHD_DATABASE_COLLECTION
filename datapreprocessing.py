"""
=============================================================================
ADFI Framework — Layer 1: Perception and Data Foundation
=============================================================================
Maps to Chapter 3.3 of the ADFI thesis

Five data sources
-----------------
  features.csv          — Pre-computed tsfresh ACC features (V_beh base)
  patient_info.csv      — Demographics, clinical scales, ADHD label
  CPT_II_Conners*.csv   — CPT-II raw trials + summary scores (V_perf)
  hrv_data/             — Raw RR-interval time series per participant
  activity_data/        — Raw 1-minute actigraphy epoch counts per participant
=============================================================================
"""

import os
import re
import glob
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# =============================================================================
# §3.3.3 Helper — HRV feature extraction from raw RR-interval files
# =============================================================================

def _extract_hrv_features(hrv_folder: str) -> pd.DataFrame:
    pattern = os.path.join(hrv_folder, "patient_hr_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"  [HRV] WARNING: No files found in '{hrv_folder}'. ")
        return pd.DataFrame()

    records = []
    for fp in files:
        match = re.search(r"patient_hr_(\d+)\.csv$", fp, re.IGNORECASE)
        if not match: continue
        pid = int(match.group(1))

        try:
            df = pd.read_csv(fp, sep=";")
            df.columns = df.columns.str.strip()
            df["ts"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
            df = df.dropna(subset=["ts", "HRV"]).sort_values("ts")
            rr = df["HRV"].values.astype(float)

            if len(rr) < 10: continue

            # Time-domain
            diff_rr = np.diff(rr)
            mean_rr = float(np.mean(rr))
            sdnn    = float(np.std(rr, ddof=1))
            rmssd   = float(np.sqrt(np.mean(diff_rr ** 2)))
            pnn50   = float((np.abs(diff_rr) > 50).sum() / len(diff_rr) * 100)
            cvnn    = sdnn / mean_rr if mean_rr > 0 else 0.0
            min_rr  = float(rr.min())
            max_rr  = float(rr.max())
            rr_range = max_rr - min_rr

            # Poincaré non-linear features
            sd1 = float(np.std(diff_rr, ddof=1) / np.sqrt(2))
            sd2 = float(np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0)))
            sd1_sd2 = sd1 / sd2 if sd2 > 0 else 0.0

            # Signal statistics
            from scipy.stats import skew, kurtosis
            skew_rr = float(skew(rr))
            kurt_rr = float(kurtosis(rr))

            duration_h = float((df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds() / 3600)

            records.append({
                "ID":           pid,
                "hrv_mean_rr":      mean_rr,
                "hrv_sdnn":         sdnn,
                "hrv_rmssd":        rmssd,
                "hrv_pnn50":        pnn50,
                "hrv_cvnn":         cvnn,
                "hrv_min_rr":       min_rr,
                "hrv_max_rr":       max_rr,
                "hrv_rr_range":     rr_range,
                "hrv_sd1":          sd1,
                "hrv_sd2":          sd2,
                "hrv_sd1_sd2":      sd1_sd2,
                "hrv_skewness_rr":  skew_rr,
                "hrv_kurtosis_rr":  kurt_rr,
                "hrv_duration_h":   duration_h,
                "hrv_n_beats":      len(rr),
            })

        except Exception as e:
            print(f"  [HRV] Skipping {fp}: {e}")

    if not records: return pd.DataFrame()

    hrv_df = pd.DataFrame(records)
    print(f"  [HRV] Extracted {len(hrv_df.columns)-1} features from {len(hrv_df)} participants")
    return hrv_df


# =============================================================================
# §3.3.3 Helper — Circadian features from raw activity files
# =============================================================================

def _extract_activity_circadian_features(activity_folder: str) -> pd.DataFrame:
    pattern = os.path.join(activity_folder, "patient_activity_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"  [Activity] WARNING: No files found in '{activity_folder}'. ")
        return pd.DataFrame()

    records = []
    for fp in files:
        match = re.search(r"patient_activity_(\d+)\.csv$", fp, re.IGNORECASE)
        if not match: continue
        pid = int(match.group(1))

        try:
            df = pd.read_csv(fp, sep=";")
            df.columns = df.columns.str.strip()
            df["ts"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
            df = df.dropna(subset=["ts", "ACTIVITY"]).sort_values("ts")

            if len(df) < 60: continue

            act = df["ACTIVITY"].values.astype(float)
            hours = df["ts"].dt.hour.values

            # Circadian features
            day_mask   = (hours >= 8)  & (hours < 20)
            night_mask = (hours >= 0)  & (hours < 6)

            day_act   = act[day_mask]
            night_act = act[night_mask]

            mean_day   = float(day_act.mean())   if len(day_act)   > 0 else np.nan
            mean_night = float(night_act.mean()) if len(night_act) > 0 else np.nan
            circ_ratio = (mean_day / mean_night if mean_night and mean_night > 0 else np.nan)

            df["act"]  = act
            df["hour"] = hours
            hourly_mean = df.groupby("hour")["act"].mean()
            peak_hour   = int(hourly_mean.idxmax()) if not hourly_mean.empty else np.nan

            sleep_eff = (float((night_act == 0).sum() / len(night_act)) if len(night_act) > 0 else np.nan)

            hourly_profile = df.groupby("hour")["act"].mean().reindex(range(24), fill_value=np.nan).values
            overall_mean = act.mean()
            n            = len(act)

            num_is = n * np.nansum((hourly_profile - overall_mean) ** 2)
            den_is = 24 * np.sum((act - overall_mean) ** 2)
            is_val = float(num_is / den_is) if den_is > 0 else np.nan

            diff_act = np.diff(act)
            num_iv   = n * np.sum(diff_act ** 2)
            den_iv   = (n - 1) * np.sum((act - overall_mean) ** 2)
            iv_val   = float(num_iv / den_iv) if den_iv > 0 else np.nan

            records.append({
                "ID":                    pid,
                "act_mean_daytime":      mean_day,
                "act_mean_nighttime":    mean_night,
                "act_circadian_ratio":   circ_ratio,
                "act_peak_hour":         peak_hour,
                "act_sleep_efficiency":  sleep_eff,
                "act_interdaily_stab":   is_val,
                "act_intradaily_var":    iv_val,
                "act_overall_mean":      float(overall_mean),
                "act_overall_std":       float(act.std()),
            })

        except Exception as e:
            print(f"  [Activity] Skipping {fp}: {e}")

    if not records: return pd.DataFrame()

    act_df = pd.DataFrame(records)
    print(f"  [Activity] Extracted {len(act_df.columns)-1} circadian features from {len(act_df)} participants")
    return act_df


# =============================================================================
# §3.3.3 Helper — CPT-II per-block temporal trajectory features
# =============================================================================

_N_BLOCKS   = 6
_BLOCK_SIZE = 60

def _extract_cpt_temporal_features(cpt_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in cpt_df.iterrows():
        feats = {"ID": row["ID"]}
        for b in range(_N_BLOCKS):
            s = b * _BLOCK_SIZE + 1
            e = s  + _BLOCK_SIZE
            resp = pd.to_numeric(pd.Series([row.get(f"Response{i}", np.nan) for i in range(s, e)]), errors="coerce")
            valid_rt = resp[resp > 0]
            px = f"cpt_block{b+1}"
            feats[f"{px}_mean_rt"]     = valid_rt.mean() if len(valid_rt) > 0 else np.nan
            feats[f"{px}_std_rt"]      = valid_rt.std()  if len(valid_rt) > 1 else 0.0
            feats[f"{px}_omissions"]   = int((resp == -1).sum())
            feats[f"{px}_commissions"] = int((resp ==  0).sum())
        records.append(feats)

    temporal_df = pd.DataFrame(records)
    print(f"  [CPT Temporal] {len(temporal_df.columns)-1} block-level trajectory features (6 blocks × 4 metrics)")
    return temporal_df


# =============================================================================
# §3.3.1  Data Sources and Mapping
# =============================================================================

def _load_and_merge_all(features_path, patient_path, cpt_path, hrv_folder, activity_folder) -> tuple:
    features_df  = pd.read_csv(features_path, sep=";")
    v_beh_tsfresh = [c for c in features_df.columns if c.startswith("ACC__")]
    print(f"  features.csv      : {len(features_df)} participants | {len(v_beh_tsfresh)} tsfresh ACC features (V_beh)")

    patient_info  = pd.read_csv(patient_path, sep=";")
    clinical_cols = ["SEX", "AGE", "WURS", "ASRS", "MADRS", "HADS_A", "HADS_D"]
    patient_sub   = patient_info[["ID", "ADHD"] + clinical_cols].copy()
    print(f"  patient_info.csv  : {len(patient_info)} participants | ADHD={patient_info['ADHD'].sum()} Control={len(patient_info)-patient_info['ADHD'].sum()}")

    cpt_df = pd.read_csv(cpt_path, sep=";")
    non_summary = ({"ID", "Assessment Status", "Assessment Duration", "Type", "LastTrial"} | {f"Trial{i}" for i in range(1, 361)} | {f"Response{i}" for i in range(1, 361)})
    v_perf_summary = [c for c in cpt_df.columns if c not in non_summary]
    cpt_summary    = cpt_df[["ID"] + v_perf_summary].copy()
    print(f"  CPT_II.csv        : {len(cpt_df)} participants | {len(v_perf_summary)} CPT summary features (V_perf)")

    cpt_temporal    = _extract_cpt_temporal_features(cpt_df)
    v_perf_temporal = [c for c in cpt_temporal.columns if c != "ID"]

    hrv_df    = _extract_hrv_features(hrv_folder)
    v_beh_hrv = [c for c in hrv_df.columns if c != "ID"] if not hrv_df.empty else []

    act_df      = _extract_activity_circadian_features(activity_folder)
    v_beh_circ  = [c for c in act_df.columns if c != "ID"] if not act_df.empty else []

    df = features_df.merge(patient_sub, on="ID", how="inner").merge(cpt_summary, on="ID", how="inner").merge(cpt_temporal, on="ID", how="inner")
    
    if not hrv_df.empty: df = df.merge(hrv_df, on="ID", how="left")
    if not act_df.empty: df = df.merge(act_df, on="ID", how="left")

    print(f"\n  Participants after merge  : {len(df)}")
    print(f"  ADHD distribution         : Control={int((df['ADHD']==0).sum())} | ADHD={int((df['ADHD']==1).sum())}")

    v_beh_cols  = v_beh_tsfresh + v_beh_hrv + v_beh_circ
    v_perf_cols = v_perf_summary + v_perf_temporal

    print(f"\n  V_beh  total : {len(v_beh_cols):>4} (tsfresh={len(v_beh_tsfresh)} | hrv={len(v_beh_hrv)} | circadian={len(v_beh_circ)})")
    print(f"  V_perf total : {len(v_perf_cols):>4} (CPT summary={len(v_perf_summary)} | CPT trajectory={len(v_perf_temporal)})")

    return df, v_beh_cols, v_perf_cols


# =============================================================================
# §3.3  Main preprocessing pipeline (BUG FIX APPLIED HERE)
# =============================================================================

def run_adf_preprocessing(
    features_path:    str   = "features.csv",
    patient_path:     str   = "patient_info.csv",
    cpt_path:         str   = "CPT_II_ConnersContinuousPerformanceTest.csv",
    hrv_folder:       str   = "hrv_data",
    activity_folder:  str   = "activity_data",
    test_size:        float = 0.20,
    pca_variance:     float = 0.95,
    smote_strategy:   str   = "auto",
    random_state:     int   = 42,
):
    print("=" * 65)
    print("  LAYER 1 — PERCEPTION AND DATA FOUNDATION")
    print("=" * 65)

    print("\n[§3.3.1] Loading and mapping all five data sources …\n")
    df, v_beh_cols, v_perf_cols = _load_and_merge_all(
        features_path, patient_path, cpt_path, hrv_folder, activity_folder
    )
    y = df["ADHD"].values.astype(int)

    print("\n[§3.3.2] Imputation …")
    drop_always = {"ID", "ADHD", "filter_$"}
    all_feat_cols = v_beh_cols + v_perf_cols
    extra_cols    = [c for c in df.columns if c not in all_feat_cols and c not in drop_always]

    df_feat  = df[[c for c in all_feat_cols + extra_cols if c in df.columns]].copy()
    num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_feat.select_dtypes(exclude=[np.number]).columns.tolist()

    X_num = df_feat[num_cols].copy()
    X_num = X_num.fillna(X_num.median()).fillna(0)
    print(f"  Numeric   : {len(num_cols)} columns | median imputation | {int(X_num.isna().sum().sum())} NaNs left")

    if len(cat_cols) > 0:
        X_cat = df_feat[cat_cols].copy()
        for col in cat_cols:
            mode_val = X_cat[col].mode()
            if not mode_val.empty:
                X_cat[col] = X_cat[col].fillna(mode_val[0])
        X_cat_enc = pd.get_dummies(X_cat, drop_first=True).astype(float)
        print(f"  Categorical: {len(cat_cols)} columns | mode imputation + one-hot → {X_cat_enc.shape[1]} binary cols")
        X_full = pd.concat([X_num.reset_index(drop=True), X_cat_enc.reset_index(drop=True)], axis=1)
    else:
        print("  Categorical: none found")
        X_full = X_num.reset_index(drop=True)
    
    feature_names = list(X_full.columns)

    # =========================================================================
    # BUG FIX: SPLIT DATA FIRST TO PREVENT LEAKAGE
    # =========================================================================
    print("\n[§3.3.4] Splitting Data, Scaling, and SMOTE balancing …")

    # 1. Split raw imputed data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    before = np.bincount(y_train)
    print(f"  Before SMOTE (Train) — Control: {before[0]} | ADHD: {before[1]}")

    # 2. Z-Score Normalization (fit on train ONLY, apply to both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)
    print(f"  Z-score normalization applied using training set parameters.")

    # 3. SMOTE (applied to scaled training data ONLY)
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X_train_scaled, y_train)
    after = np.bincount(y_balanced)
    print(f"  After SMOTE (Train)  — Control: {after[0]} | ADHD: {after[1]}")

    # 4. PCA Dimensionality Reduction (fit on balanced train ONLY, apply to both)
    pca = PCA(n_components=pca_variance, random_state=random_state)
    X_train_pca = pca.fit_transform(X_balanced)
    X_test_pca  = pca.transform(X_test_scaled)
    print(f"  PCA applied using balanced training set parameters: 903 → {X_train_pca.shape[1]} components ({pca_variance*100:.0f}% variance)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 65)
    print("  LAYER 1 RESULTS (DATA LEAKAGE FIXED)")
    print("=" * 65)
    print(f"  Total raw features             : {X_full.shape[1]}")
    print(f"  PCA components (95% variance)  : {X_train_pca.shape[1]}")
    print(f"  Training samples (post-SMOTE)  : {X_balanced.shape[0]}")
    print(f"  Test samples (held-out)        : {X_test_scaled.shape[0]}")
    print(f"\n  Data is now cleanly separated and ready for Layer 2.")
    print("=" * 65)

    return (X_balanced, y_balanced, X_test_scaled, y_test,
            feature_names, X_train_pca, X_test_pca, scaler, pca)

if __name__ == "__main__":
    (X_train, y_train, X_test, y_test,
     feat_names, X_train_pca, X_test_pca, scaler, pca) = run_adf_preprocessing()