from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_score, recall_score, confusion_matrix, classification_report, precision_recall_curve,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import TimeSeriesSplit
import utils
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import KFold
from graph import build_graph, train_gnn
import optuna
from sklearn.metrics import average_precision_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../src')


## =============== XGBOOST ===============

def make_xgb(
    n_estimators:          int   = 2000,
    max_depth:             int   = 6,
    learning_rate:         float = 0.02,
    subsample:             float = 0.8,
    colsample_bytree:      float = 0.8,
    colsample_bylevel:     float = 1.0,    
    colsample_bynode:      float = 1.0,    
    scale_pos_weight:      float = 1.0,
    min_child_weight:      int   = 1,      
    max_delta_step:        int   = 0,     
    gamma:                 float = 0.0,    
    reg_alpha:             float = 0.0,    
    reg_lambda:            float = 1.0,    
    early_stopping_rounds: int   = 100,
    enable_categorical:    bool  = True,
    eval_metric:           str   = "aucpr",   
    random_state:          int   = 42,
    device:                str   = "cuda",
    **kwargs,
) -> XGBClassifier:
    """ . 
    """
    return XGBClassifier(
        n_estimators          = n_estimators,
        max_depth             = max_depth,
        learning_rate         = learning_rate,
        subsample             = subsample,
        colsample_bytree      = colsample_bytree,
        colsample_bylevel     = colsample_bylevel,   
        colsample_bynode      = colsample_bynode,    
        scale_pos_weight      = scale_pos_weight,
        min_child_weight      = min_child_weight,    
        max_delta_step        = max_delta_step,      
        gamma                 = gamma,               
        reg_alpha             = reg_alpha,           
        reg_lambda            = reg_lambda,          
        eval_metric           = eval_metric,
        early_stopping_rounds = early_stopping_rounds,
        tree_method           = "hist",
        device                = device,
        random_state          = random_state,
        enable_categorical    = enable_categorical,

    )


def align_categories(
        X_train: pd.DataFrame,
        X_val:   pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    """
    X_train = X_train.copy()
    X_val   = X_val.copy()
    cat_cols = X_train.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        train_cats   = X_train[col].cat.categories
        X_train[col] = X_train[col].cat.set_categories(train_cats)
        X_val[col]   = pd.Categorical(X_val[col], categories=train_cats)

    return X_train, X_val


def train_xgboost(
        X_train: pd.DataFrame,
        y_train,
        X_val:   pd.DataFrame,
        y_val,
        params:  dict,
        verbose = True) -> tuple:
    excluded_keys = {"spw_mode",}
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)

    spw_mode = params.get("spw_mode", "full")
    if spw_mode == "full":
        spw = n_neg / max(n_pos, 1)           
    elif spw_mode == "sqrt":
        spw = np.sqrt(n_neg / max(n_pos, 1))  
    else:
        spw = 1.0                              

    # print(f"  spw_mode={spw_mode} → scale_pos_weight={spw:.2f}")

    X_train, X_val = align_categories(X_train, X_val)

    xgb_params = {k: v for k, v in params.items()
                  if k not in excluded_keys}

    assert len(X_train.select_dtypes(include=["object"]).columns) == 0, \
    f"Object columns in X_train: {X_train.select_dtypes(include=['object']).columns.tolist()}"

    assert len(X_val.select_dtypes(include=["object"]).columns) == 0, \
        f"Object columns in X_val: {X_val.select_dtypes(include=['object']).columns.tolist()}"

    model = make_xgb(
        **xgb_params,
        scale_pos_weight = spw,
        # scale_pos_weight = scale_pos_weight,
        # early_stopping_rounds = params.get("early_stopping_rounds", 100),
        eval_metric      = "aucpr",
        verbosity        = 0,
        random_state     = 42,
        device           = "cuda" if torch.cuda.is_available() else "cpu",)

    model.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        verbose   = 100 if verbose else False,
    )

    probs   = model.predict_proba(X_val)[:, 1]

    return model, probs


def analyze_feature_importance(
        xgb_model,
        X_train:        pd.DataFrame,
        hgnn_model      = None,
        top_n:          int   = 30,
        importance_type: str  = "gain",     # "weight" | "gain" | "cover"
        drop_threshold: float = 0.0001,     # feature - candidate to drop
) -> pd.DataFrame:
    """
    """
    booster    = xgb_model.get_booster()
    score_dict = booster.get_score(importance_type=importance_type)

    imp = pd.Series(0.0, index=X_train.columns)
    for k, v in score_dict.items():
        if k in imp.index:
            imp[k] = v

    imp = imp.sort_values(ascending=False)
    total = imp.sum()

    def get_feature_type(col: str) -> str:
        if col.startswith("gnn_"):
            return "GNN"
        elif col.startswith("V"):
            return "V_feature"
        elif col.startswith("C"):
            return "C_feature"
        elif col.startswith("D"):
            return "D_feature"
        elif col.startswith("id_"):
            return "id_feature"
        elif col.startswith("M"):
            return "M_feature"
        elif col.startswith("card"):
            return "card"
        elif col.startswith("addr"):
            return "addr"
        elif col.startswith("amt"):
            return "amount"
        elif col.startswith("tx_") or col.startswith("log_tx"):
            return "velocity"
        elif col in ["TransactionAmt", "TransactionDT", "log_dt_prev", "is_burst"]:
            return "temporal_amount"
        else:
            return "other"

    df_imp = pd.DataFrame({
        "feature":    imp.index,
        "importance": imp.values,
        "pct":        imp.values / (total + 1e-8) * 100,
        "type":       [get_feature_type(c) for c in imp.index],
        "cumulative": imp.values.cumsum() / (total + 1e-8) * 100,
    })

    print(f"Feature Importance TOP {top_n} ({importance_type})")
    print(f"{'='*100}")
    print(f"{'Feature':<35} {'Type':<15} {'Imp':>8} {'Pct':>7} {'Cum%':>7}")
    print(f"{'─'*100}")
    for _, row in df_imp.head(top_n).iterrows():
        print(f"  {row['feature']:<33} {row['type']:<15} "
              f"{row['importance']:>8.4f} {row['pct']:>6.2f}% {row['cumulative']:>6.1f}%")

    print(f"\n{'='*40}")
    print("Importance by feature type:")
    print(f"{'─'*40}")
    type_imp = df_imp.groupby("type")["importance"].sum().sort_values(ascending=False)
    for ftype, total_imp in type_imp.items():
        pct = total_imp / (total + 1e-8) * 100
        print(f"  {ftype:<20}: {total_imp:>8.4f}  ({pct:.1f}%)")

    if hgnn_model is not None and hasattr(hgnn_model, "stream_names"):
        print(f"\n{'='*40}")
        print("GNN per-stream importance:")
        print(f"{'─'*40}")
        for stream in hgnn_model.stream_names:
            cols      = [c for c in df_imp["feature"]
                        if c.startswith(f"gnn_{stream}_")]
            valid     = [c for c in cols if c in imp.index]
            stream_total = imp[valid].sum() if valid else 0.0
            pct       = stream_total / (total + 1e-8) * 100
            print(f"  {stream:<30}: {stream_total:.4f}  ({pct:.1f}%)")

    zero_imp    = df_imp[df_imp["importance"] == 0.0]
    low_imp     = df_imp[
        (df_imp["importance"] > 0) &
        (df_imp["importance"] < drop_threshold * total)
    ]

    if len(zero_imp) > 0:
        print(f"\n  Zero importance ({len(zero_imp):>4} features")
        for col in zero_imp["feature"].tolist()[:20]:
            ftype = get_feature_type(col)
            print(f"    {col:<35} [{ftype}]")
        if len(zero_imp) > 20:
            print(f"    ... і ще {len(zero_imp)-20}")

    if len(low_imp) > 0:
        print(f"\n  Low importance ({len(low_imp):>4} features")
        for _, row in low_imp.iterrows():
            print(f"    {row['feature']:<35} [{row['type']:<12}] "
                  f"{row['pct']:.3f}%")
            

    # 90% Coverage
    coverage_90 = (df_imp["cumulative"] <= 90).sum()
    print(f"\n  90% importance covered by top {coverage_90} features "
          f"from {len(df_imp)} total")
    return df_imp



def get_features_to_drop(
        df_imp:           pd.DataFrame,
        # # exclude_prefixes: list  = ["gnn_"],
        # drop_zero:        bool  = True,
        # drop_low:         bool  = False,
        drop_threshold:   float = 0.0001,
) -> dict:
    """
    """
    total = df_imp["importance"].sum()
    result = {
        "zero":     [],
        "low":      [],
        "all":      [],
        "metadata": {
            "total_features":    len(df_imp),
            # "exclude_prefixes":  exclude_prefixes,
            "drop_threshold":    drop_threshold,
        }
    }

    result["zero"] = [
        c for c in df_imp[df_imp["importance"] == 0.0]["feature"].tolist()
            # if not is_excluded(c)
        ]

    result["low"] = [
        c for c in df_imp[
            (df_imp["importance"] > 0) &
            (df_imp["importance"] < drop_threshold * total)
        ]["feature"].tolist()
        # if not is_excluded(c)
        ]

    result["all"] = list(set(result["zero"] + result["low"]))

    return result


def optimize_xgboost(
        X_train:    pd.DataFrame,
        y_train:    pd.Series,
        X_val:      pd.DataFrame,
        y_val:      pd.Series,
        n_trials:   int = 50,
        timeout:    int = 3600,   
        base_params: dict = None,
        search_space_override: dict = None,
        seed_trial: dict = None,
) -> tuple[dict, optuna.Study]:
    """
    Optuna hyperparameter optimization для XGBoost. 
    Metric: PR-AUC
    """
    base_params = base_params or {}

    fixed_params = {
        "n_estimators":          2000,
        "early_stopping_rounds": 100,
        "spw_mode":              base_params.get("spw_mode", "sqrt"),
    }

    default_search_space = {
        "max_depth":         {"type": "int",   "low": 4,    "high": 10},
        "min_child_weight":  {"type": "int",   "low": 1,    "high": 20},
        "max_delta_step":    {"type": "int",   "low": 0,    "high": 5},
        "subsample":         {"type": "float", "low": 0.5,  "high": 1.0},
        "colsample_bytree":  {"type": "float", "low": 0.5,  "high": 1.0},
        "colsample_bylevel": {"type": "float", "low": 0.5,  "high": 1.0},
        "colsample_bynode":  {"type": "float", "low": 0.5,  "high": 1.0},
        "learning_rate":     {"type": "float", "low": 0.005,"high": 0.1,  "log": True},
        "gamma":             {"type": "float", "low": 0.0,  "high": 5.0},
        "reg_alpha":         {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        "reg_lambda":        {"type": "float", "low": 1e-3, "high": 20.0, "log": True},
    }

    search_space = search_space_override or default_search_space

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "n_estimators":          2000,
            "early_stopping_rounds": 100,
            # "eval_metric":           "aucpr",
            "spw_mode":              base_params.get("spw_mode", "sqrt"),

            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),

            "subsample":         trial.suggest_float("subsample",         0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bynode":  trial.suggest_float("colsample_bynode",  0.5, 1.0),

            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "gamma":         trial.suggest_float("gamma",         0.0,   5.0),
            "reg_alpha":     trial.suggest_float("reg_alpha",     1e-3,  10.0, log=True),
            "reg_lambda":    trial.suggest_float("reg_lambda",    1e-3,  20.0, log=True),
        }

        merged_params = {
            **base_params,
            **trial_params,
        }

        try:
            _, probas = train_xgboost(
                X_train, y_train, X_val, y_val, merged_params,
                verbose=False,
            )
            pr_auc = average_precision_score(y_val, probas)
            return pr_auc
        except Exception as e:
            print(f"  Trial failed: {e}")
            raise optuna.exceptions.TrialPruned()

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials  = 5,
        n_warmup_steps    = 10,
    )

    study = optuna.create_study(
        direction = "maximize",
        sampler   = sampler,
        pruner    = pruner,
        study_name = "xgboost_prauc",
    )

    if seed_trial is not None:
        valid_seed = {
            k: v for k, v in seed_trial.items()
            if k in search_space
        }
        study.enqueue_trial(valid_seed)
        print(f"  Seeded with {len(valid_seed)} params from previous best")

    # First Trial Diagnostic
    print("── Running diagnostic trial ──")
    try:
        first_trial = study.ask()
        value       = objective(first_trial)
        study.tell(first_trial, value)
        print(f"  Diagnostic trial PR-AUC: {value:.4f}")
    except Exception as e:
        print(f"  Diagnostic trial FAILED: {e}")
        raise RuntimeError(f"Objective function failed: {e}")
    
    print(f"Starting Optuna optimization: {n_trials} trials, timeout={timeout}s")
    print(f"Metric: PR-AUC\n")

    study.optimize(
        objective,
        n_trials  = n_trials - 1,
        timeout   = timeout,
        show_progress_bar = True,
        catch             = (Exception,),
    )

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nCompleted trials: {len(completed)} / {len(study.trials)}")

    if not completed:
        raise RuntimeError("No trials completed — check objective function")
    
    best_trial = study.best_trial
    print(f"\n{'='*55}")
    print(f"Best PR-AUC:  {best_trial.value:.4f}")
    print(f"Best params:")
    for k, v in best_trial.params.items():
        print(f"  {k:<25}: {v}")

    print(f"\nTop 5 trials:")
    top5 = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
         reverse=True,)[:5]
    for i, t in enumerate(top5):
        print(f"  #{i+1}: PR-AUC={t.value:.4f}  trial={t.number}")

    return best_trial.params, study


def train_xgboost_full(
        X_train: pd.DataFrame,
        y_train,
        params: dict,
        verbose: bool = True,
):
    excluded_keys = {
        "spw_mode",
        "early_stopping_rounds",
        "eval_metric",
        "verbosity",
        "random_state",
        "device",
        "scale_pos_weight",
    }
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw_mode = params.get("spw_mode", "full")

    if spw_mode == "full":
        spw = n_neg / max(n_pos, 1)
    elif spw_mode == "sqrt":
        spw = np.sqrt(n_neg / max(n_pos, 1))
    elif spw_mode == "none":
        spw = 1.0
    else:
        raise ValueError(f"Unknown spw_mode: {spw_mode}")

    print(f"  spw_mode={spw_mode} → scale_pos_weight={spw:.2f}")

    obj_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    assert not obj_cols, f"Object columns in X_train: {obj_cols}"

    xgb_params = {
        k: v for k, v in params.items()
        if k not in excluded_keys
    }

    model = make_xgb(
        **xgb_params,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        verbosity=0,
        random_state=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        early_stopping_rounds=None,
    )

    model.fit(
        X_train,
        y_train,
        verbose=verbose,
    )

    return model


## =============== Metrics ===============

def compute_metrics(
        probas: np.ndarray,
        y_true,
) -> dict:
    """
    """
    y_np   = np.asarray(y_true).ravel()
    probas = np.asarray(probas).ravel()

    assert len(probas) == len(y_np), \
        f"probas {len(probas)} != y_true {len(y_np)}"

    auc    = float(roc_auc_score(y_np, probas))
    pr_auc = float(average_precision_score(y_np, probas))

    precisions, recalls, pr_thresholds = precision_recall_curve(y_np, probas)

    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0,
    )
    best_idx  = int(np.argmax(f1_scores[:-1]))  
    best_thr  = float(pr_thresholds[best_idx])
    best_f1   = float(f1_scores[best_idx])
    best_prec = float(precisions[best_idx])
    best_rec  = float(recalls[best_idx])

    n_pos = int(y_np.sum())
    n_neg = int(len(y_np) - n_pos)

    metrics = {
        "auc":       auc,
        "pr_auc":    pr_auc,
        "best_thr":  best_thr,
        "f1":        best_f1,
        "precision": best_prec,
        "recall":    best_rec,
        "n_pos":     n_pos,
        "n_neg":     n_neg,
    }

    print("=" * 72)
    print(f"{'Metric':<18} {'Value':>10}")
    print("-" * 72)
    print(f"{'ROC-AUC':<18} {auc:>10.4f}")
    print(f"{'PR-AUC':<18} {pr_auc:>10.4f}")
    print(f"{'Best threshold':<18} {best_thr:>10.4f}")
    print(f"{'F1':<18} {best_f1:>10.4f}")
    print(f"{'Precision':<18} {best_prec:>10.4f}")
    print(f"{'Recall':<18} {best_rec:>10.4f}")
    print(f"{'Fraud / Non-fraud':<18} {n_pos:>10,} / {n_neg:,}")
    print("=" * 72)

    thresholds = [best_thr, 0.3, 0.5, 0.7]
    thresholds = sorted(set(round(t, 4) for t in thresholds))
    rows = []
    for thr in thresholds:
        y_pred = (probas > thr).astype(int)

        rows.append({
            "threshold": thr,
            "precision": precision_score(y_np, y_pred, zero_division=0),
            "recall": recall_score(y_np, y_pred, zero_division=0),
            "f1": f1_score(y_np, y_pred, zero_division=0),
            "fraud_pred_%": 100 * y_pred.mean(),
        })

    thr_df = pd.DataFrame(rows)

    print("\nThreshold comparison:")
    print(thr_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


    print(f"\nClassification report at best threshold={best_thr:.4f}:")
    y_pred_best = (probas > best_thr).astype(int)
    print(classification_report(
        y_np,
        y_pred_best,
        target_names=["non-fraud", "fraud"],
        digits=4,
    ))

    y_pred_best = (probas > best_thr).astype(int)
    # cm = confusion_matrix(y_np, y_pred_best)

    # fig, ax = plt.subplots(figsize=(4, 4))
    # ConfusionMatrixDisplay(
    #     cm,
    #     display_labels=["non-fraud", "fraud"],
    # ).plot(
    #     cmap="Blues",
    #     ax=ax,
    #     values_format="d",
    # )
    # ax.set_title(f"Confusion Matrix at best threshold={best_thr:.2f}")
    # plt.tight_layout()
    # plt.show()
    # plt.close(fig)

    return metrics











# def prob_to_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
#     """probability → logit = log(p / (1-p))"""
#     p = np.clip(p, eps, 1 - eps)
#     return np.log(p / (1 - p))


# def normalize_logits(x: np.ndarray) -> np.ndarray:
#     """Нормалізує logits до однакового масштабу."""
#     std = x.std()
#     return x / std if std > 1e-8 else x


# # def weighted_ensemble_logits(
# #         gnn_logits:  np.ndarray,
# #         xgb_probas:  np.ndarray,
# #         y_true,
# #         n_grid:      int         = 101,
# #         n_refine:    int         = 21,
# #         holdout_gnn: np.ndarray  = None,
# #         holdout_xgb: np.ndarray  = None,
# #         y_holdout                = None,
# # ) -> tuple:
# #     y_np = y_true.values if hasattr(y_true, "values") else np.array(y_true)

# #     # ── Конвертація і нормалізація ────────────────────────────
# #     gnn_raw = gnn_logits.ravel()
# #     xgb_raw = prob_to_logit(xgb_probas.ravel())

# #     # ✅ fit scale на val
# #     gnn_std = gnn_raw.std()
# #     xgb_std = xgb_raw.std()

# #     gnn_r = gnn_raw / (gnn_std + 1e-8)
# #     xgb_r = xgb_raw / (xgb_std + 1e-8)

# #     assert len(gnn_r) == len(y_np)
# #     assert len(xgb_r) == len(y_np)

# #     if gnn_std < 1e-8:
# #         print("  ⚠️  GNN logits константні")
# #     if xgb_std < 1e-8:
# #         print("  ⚠️  XGB logits константні")

# #     # ── Grid search на val ────────────────────────────────────
# #     # ✅ sigmoid зайвий для AUC — рахуємо по raw combined
# #     def combined(w: float) -> np.ndarray:
# #         return w * gnn_r + (1 - w) * xgb_r

# #     best_auc, best_w = 0.0, 0.5
# #     for w in np.linspace(0, 1, n_grid):
# #         try:
# #             auc = roc_auc_score(y_np, combined(w))
# #         except Exception:
# #             continue
# #         if auc > best_auc:
# #             best_auc, best_w = auc, w

# #     # ── Fine-tune ─────────────────────────────────────────────
# #     lo = max(0.0, best_w - 0.05)
# #     hi = min(1.0, best_w + 0.05)
# #     for w in np.linspace(lo, hi, n_refine):
# #         try:
# #             auc = roc_auc_score(y_np, combined(w))
# #         except Exception:
# #             continue
# #         if auc > best_auc:
# #             best_auc, best_w = auc, w

# #     # ✅ sigmoid тільки для фінальних probas
# #     val_probas = stable_sigmoid(combined(best_w))

# #     # ── Діагностика на val ────────────────────────────────────
# #     gnn_only_auc = roc_auc_score(y_np, gnn_r)
# #     xgb_only_auc = roc_auc_score(y_np, xgb_r)

# #     print("─" * 50)
# #     print(f"  [VAL — підбір w]")
# #     print(f"  GNN  only AUC : {gnn_only_auc:.4f}")
# #     print(f"  XGB  only AUC : {xgb_only_auc:.4f}")
# #     print(f"  Ensemble AUC  : {best_auc:.4f}  ⚠️  overfit на val")
# #     print(f"  best_w (GNN)  : {best_w:.3f}")
# #     print(f"  best_w (XGB)  : {1 - best_w:.3f}")
# #     print("─" * 50)

# #     # ── Holdout з фіксованим best_w і val scale ───────────────
# #     holdout_probas  = None
# #     holdout_metrics = None

# #     if holdout_gnn is not None and holdout_xgb is not None \
# #             and y_holdout is not None:
# #         y_hold = (
# #             y_holdout.values
# #             if hasattr(y_holdout, "values")
# #             else np.array(y_holdout)
# #         )

# #         # ✅ apply val scale на holdout — no leakage
# #         hold_gnn = holdout_gnn.ravel() / (gnn_std + 1e-8)
# #         hold_xgb = prob_to_logit(holdout_xgb.ravel()) / (xgb_std + 1e-8)

# #         hold_combined  = best_w * hold_gnn + (1 - best_w) * hold_xgb
# #         holdout_probas = stable_sigmoid(hold_combined)

# #         hold_auc    = roc_auc_score(y_hold, hold_combined)  # sigmoid не потрібен
# #         hold_pr_auc = average_precision_score(y_hold, holdout_probas)

# #         print(f"  [HOLDOUT — чесна оцінка w={best_w:.3f}]")
# #         print(f"  Ensemble ROC-AUC : {hold_auc:.4f}")
# #         print(f"  Ensemble PR-AUC  : {hold_pr_auc:.4f}")
# #         print("─" * 50)

# #         holdout_metrics = compute_metrics(holdout_probas, y_hold)
# #     else:
# #         print("  ⚠️  Holdout не передано — val AUC є overfit оцінкою")
# #         print("─" * 50)

# #     return best_w, val_probas


# def weighted_ensemble_logits(
#         gnn_logits: np.ndarray,
#         xgb_probas: np.ndarray,
#         y_true,
#         n_grid:   int   = 101,
#         n_refine: int   = 21,
#         metric:   str   = "pr_auc",      # ✅ "pr_auc" | "roc_auc"
#         w_max:    float = 1.0,            # ✅ обмеження ваги GNN (0.3 для constrained)
# ) -> tuple[float, np.ndarray]:
#     """
#     Ensemble через logit space.
#     ✅ Z-score нормалізація (mean + std)
#     ✅ Оптимізація по PR-AUC або ROC-AUC
#     ✅ Constrained або unconstrained вага GNN
#     ⚠️ best_w підбирається на val — overfit оцінка.
#     """
#     y_np = y_true.values if hasattr(y_true, "values") else np.array(y_true)

#     gnn_raw = gnn_logits.ravel()
#     xgb_raw = prob_to_logit(xgb_probas.ravel())

#     # ✅ Z-score: fit mean+std на val
#     gnn_mean, gnn_std = gnn_raw.mean(), gnn_raw.std()
#     xgb_mean, xgb_std = xgb_raw.mean(), xgb_raw.std()

#     gnn_r = (gnn_raw - gnn_mean) / (gnn_std + 1e-8)
#     xgb_r = (xgb_raw - xgb_mean) / (xgb_std + 1e-8)

#     assert len(gnn_r) == len(y_np)
#     assert len(xgb_r) == len(y_np)

#     if gnn_std < 1e-8:
#         print("  ⚠️  GNN logits константні")
#     if xgb_std < 1e-8:
#         print("  ⚠️  XGB logits константні")

#     def combined(w: float) -> np.ndarray:
#         return w * gnn_r + (1 - w) * xgb_r

#     # ✅ метрика для підбору ваги
#     def score(w: float) -> float:
#         c = combined(w)
#         try:
#             if metric == "pr_auc":
#                 return average_precision_score(y_np, stable_sigmoid(c))
#             else:
#                 return roc_auc_score(y_np, c)
#         except Exception:
#             return 0.0

#     # ── Grid search ───────────────────────────────────────────
#     # ✅ w_max обмежує діапазон пошуку ваги GNN
#     best_score, best_w = 0.0, 0.5
#     for w in np.linspace(0.0, w_max, n_grid):
#         s = score(w)
#         if s > best_score:
#             best_score, best_w = s, w

#     # ── Fine-tune ─────────────────────────────────────────────
#     lo = max(0.0,   best_w - 0.05)
#     hi = min(w_max, best_w + 0.05)
#     for w in np.linspace(lo, hi, n_refine):
#         s = score(w)
#         if s > best_score:
#             best_score, best_w = s, w

#     val_probas = stable_sigmoid(combined(best_w))

#     # ── Діагностика ───────────────────────────────────────────
#     gnn_roc = roc_auc_score(y_np, gnn_r)
#     xgb_roc = roc_auc_score(y_np, xgb_r)
#     ens_roc = roc_auc_score(y_np, combined(best_w))

#     gnn_pr  = average_precision_score(y_np, stable_sigmoid(gnn_r))
#     xgb_pr  = average_precision_score(y_np, stable_sigmoid(xgb_r))
#     ens_pr  = average_precision_score(y_np, val_probas)

#     print("─" * 55)
#     print(f"  [VAL — підбір w по {metric}]  ⚠️ overfit на val")
#     print(f"  {'':20s} {'ROC-AUC':>8} {'PR-AUC':>8}")
#     print(f"  {'GNN only':20s} {gnn_roc:>8.4f} {gnn_pr:>8.4f}")
#     print(f"  {'XGB only':20s} {xgb_roc:>8.4f} {xgb_pr:>8.4f}")
#     print(f"  {'Ensemble':20s} {ens_roc:>8.4f} {ens_pr:>8.4f}")
#     print(f"  best_w (GNN)  : {best_w:.3f}  (w_max={w_max})")
#     print(f"  best_w (XGB)  : {1 - best_w:.3f}")
#     print("─" * 55)

#     # ✅ зберігаємо artifacts для застосування на holdout
#     artifacts = {
#         "gnn_mean": gnn_mean, "gnn_std": gnn_std,
#         "xgb_mean": xgb_mean, "xgb_std": xgb_std,
#         "best_w":   best_w,
#         "metric":   metric,
#         "w_max":    w_max,
#     }

#     return best_w, val_probas, artifacts


# def apply_ensemble(
#         gnn_logits:  np.ndarray,
#         xgb_probas:  np.ndarray,
#         artifacts:   dict,
# ) -> np.ndarray:
#     """
#     Застосовує ensemble з val artifacts на holdout/test.
#     ✅ Використовує val mean/std — no leakage.
#     """
#     gnn_raw = gnn_logits.ravel()
#     xgb_raw = prob_to_logit(xgb_probas.ravel())

#     # ✅ apply val scale на holdout
#     gnn_r = (gnn_raw - artifacts["gnn_mean"]) / (artifacts["gnn_std"] + 1e-8)
#     xgb_r = (xgb_raw - artifacts["xgb_mean"]) / (artifacts["xgb_std"] + 1e-8)

#     w = artifacts["best_w"]
#     return stable_sigmoid(w * gnn_r + (1 - w) * xgb_r)

# # ════════════════════════════════════════════════════════════════
# # 9. Stacking Model (Meta-learning)

# def probas_to_logits(probas: np.ndarray) -> np.ndarray:
#     """probas → logits з clip для numerical stability."""
#     p = np.clip(probas, 1e-7, 1 - 1e-7)
#     return np.clip(np.log(p / (1 - p)), -20, 20)


# def get_oof_gnn_logits(
#         train_df:     pd.DataFrame,
#         graph_config,
#         model_config,
#         train_config,
#         device,
#         n_splits:     int = 5,
#         graph_config_copy_fn = None,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     OOF GNN logits для train — TimeSeriesSplit.
#     Повертає (oof_logits, oof_mask) — mask=True тільки для покритих рядків.
#     """
#     N          = len(train_df)
#     oof_logits = np.zeros(N, dtype=np.float32)
#     oof_mask   = np.zeros(N, dtype=bool)
#     tscv       = TimeSeriesSplit(n_splits=n_splits)
#     fold_aucs  = []

#     for fold, (tr_idx, vl_idx) in enumerate(tscv.split(np.arange(N))):
#         print(f"\n{'═'*55}")
#         print(f"  OOF GNN Fold {fold+1}/{n_splits} | "
#               f"train={len(tr_idx):,} | val={len(vl_idx):,}")
#         print(f"{'═'*55}")

#         # ✅ використовуємо передану функцію
#         if graph_config_copy_fn is not None:
#             fold_graph_config = graph_config_copy_fn(graph_config)
#         else:
#             raise ValueError(
#             "graph_config_copy_fn не передано — "
#             "GraphConfig недоступний в models.py. "
#             "Передайте graph.copy_graph_config як аргумент."
#         )

#         fold_train = train_df.iloc[tr_idx].reset_index(drop=True)
#         fold_val   = train_df.iloc[vl_idx].reset_index(drop=True)

#         # # ✅ копіюємо graph_config — зберігаємо всі налаштування
#         # # але очищаємо артефакти щоб не було cross-fold contamination
#         # fold_graph_config = GraphConfig(
#         #     graph_features       = graph_config.graph_features,
#         #     entity_cols          = graph_config.entity_cols,
#         #     entity_feat_cols     = graph_config.entity_feat_cols,
#         #     edge_weight_type     = graph_config.edge_weight_type,
#         #     temporal_enabled     = graph_config.temporal_enabled,
#         #     temporal_threshold   = graph_config.temporal_threshold,
#         #     temporal_group_cols  = graph_config.temporal_group_cols,
#         #     node_norm            = graph_config.node_norm,
#         #     # ✅ артефакти порожні — fit на fold_train
#         # )

#         with utils.timer(f"Build Graph fold {fold+1}"):
#             fold_graph = build_graph(
#                 fold_train, fold_val, fold_graph_config
#             )

#         with utils.timer(f"Train GNN fold {fold+1}"):
#             _, fold_artifacts = train_gnn(
#                 data         = fold_graph,
#                 y_train      = fold_train["isFraud"],
#                 device       = device,
#                 train_config = train_config,
#                 model_config = model_config,
#                 graph_config = fold_graph_config,
#             )

#         oof_logits[vl_idx] = fold_artifacts["val_logits"]
#         oof_mask[vl_idx]   = True   # ✅ позначаємо покриті рядки

#         fold_auc = roc_auc_score(
#             fold_val["isFraud"].values,
#             stable_sigmoid(fold_artifacts["val_logits"])
#         )
#         fold_aucs.append(fold_auc)
#         print(f"  Fold {fold+1} AUC: {fold_auc:.4f}")

#         del fold_graph, fold_artifacts
#         utils.clear_memory()

#     n_covered = oof_mask.sum()
#     print(f"\n  OOF coverage: {n_covered:,}/{N:,} ({n_covered/N:.1%})")
#     print(f"  Uncovered (first segment): {N - n_covered:,} rows — excluded from stacking")
#     print(f"  Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
#     print(f"  Mean: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

#     oof_auc = roc_auc_score(
#         train_df["isFraud"].values[oof_mask],
#         stable_sigmoid(oof_logits[oof_mask])
#     )
#     print(f"  OOF GNN AUC (covered only): {oof_auc:.4f}")

#     return oof_logits, oof_mask


# def get_oof_xgb_logits(
#         X_train:  pd.DataFrame,
#         y_train,
#         params:   dict,
#         n_splits: int = 5,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     OOF XGB logits для train — TimeSeriesSplit.
#     Повертає (oof_logits, oof_mask).
#     """
#     y_np       = np.asarray(y_train).ravel()
#     oof_logits = np.zeros(len(X_train), dtype=np.float32)
#     oof_mask   = np.zeros(len(X_train), dtype=bool)
#     tscv       = TimeSeriesSplit(n_splits=n_splits)

#     exclude      = {"spw_mode"}
#     params_clean = {k: v for k, v in params.items() if k not in exclude}
#     spw_mode     = params.get("spw_mode", "full")
#     fold_aucs    = []

#     for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X_train)):
#         X_tr = X_train.iloc[tr_idx].copy()
#         X_vl = X_train.iloc[vl_idx].copy()
#         y_tr = y_np[tr_idx]
#         y_vl = y_np[vl_idx]

#         X_tr, X_vl = align_categories(X_tr, X_vl)

#         n_pos = int(y_tr.sum())
#         n_neg = int(len(y_tr) - n_pos)

#         if spw_mode == "sqrt":
#             spw = np.sqrt(n_neg / max(n_pos, 1))
#         elif spw_mode == "full":
#             spw = n_neg / max(n_pos, 1)
#         else:
#             spw = 1.0

#         fold_model = make_xgb(**params_clean, scale_pos_weight=spw)
#         fold_model.fit(
#             X_tr, y_tr,
#             eval_set = [(X_vl, y_vl)],
#             verbose  = 200,
#         )

#         oof_probas         = fold_model.predict_proba(X_vl)[:, 1]
#         oof_logits[vl_idx] = probas_to_logits(oof_probas)
#         oof_mask[vl_idx]   = True   # ✅

#         fold_auc = roc_auc_score(y_vl, oof_probas)
#         fold_aucs.append(fold_auc)
#         print(f"  Fold {fold+1}/{n_splits} | "
#               f"train={len(tr_idx):,} | val={len(vl_idx):,} | "
#               f"n_pos={n_pos} | AUC={fold_auc:.4f}")

#         del fold_model
#         utils.clear_memory()

#     n_covered = oof_mask.sum()
#     print(f"\n  OOF coverage: {n_covered:,}/{len(X_train):,} ({n_covered/len(X_train):.1%})")
#     print(f"  Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
#     print(f"  Mean: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

#     oof_auc = roc_auc_score(y_np[oof_mask], stable_sigmoid(oof_logits[oof_mask]))
#     print(f"  OOF XGB AUC (covered only): {oof_auc:.4f}")

#     return oof_logits, oof_mask


# def train_stacking(
#         train_gnn_logits: np.ndarray,
#         train_xgb_logits: np.ndarray,
#         y_train,
#         val_gnn_logits:   np.ndarray,
#         val_xgb_logits:   np.ndarray,
#         y_val,
#         params:           dict,
#         oof_mask:         np.ndarray = None,   # ✅ маска покритих рядків
# ) -> tuple:
#     """
#     Stacking meta-model на OOF logits — без leakage.
#     oof_mask — тренуємо тільки на покритих рядках.
#     """
#     y_tr = y_train.values if hasattr(y_train, "values") else np.array(y_train)
#     y_vl = y_val.values   if hasattr(y_val,   "values") else np.array(y_val)

#     # ✅ застосовуємо маску якщо передана
#     if oof_mask is not None:
#         assert len(oof_mask) == len(y_tr), \
#             f"oof_mask {len(oof_mask)} != y_train {len(y_tr)}"
#         train_gnn_logits = train_gnn_logits[oof_mask]
#         train_xgb_logits = train_xgb_logits[oof_mask]
#         y_tr             = y_tr[oof_mask]
#         print(f"  Stacking train size (OOF covered): {len(y_tr):,}")

#     assert len(train_gnn_logits) == len(train_xgb_logits) == len(y_tr)
#     assert len(val_gnn_logits)   == len(val_xgb_logits)   == len(y_vl)

#     # ✅ z-score: fit на train, apply на val
#     def fit_normalize(
#             train_x: np.ndarray,
#             val_x:   np.ndarray,
#     ) -> tuple[np.ndarray, np.ndarray]:
#         mean = train_x.mean()
#         std  = train_x.std() + 1e-8
#         return (train_x - mean) / std, (val_x - mean) / std

#     gnn_tr_n, gnn_vl_n = fit_normalize(train_gnn_logits, val_gnn_logits)
#     xgb_tr_n, xgb_vl_n = fit_normalize(train_xgb_logits, val_xgb_logits)

#     def make_meta_features(
#             gnn: np.ndarray,
#             xgb: np.ndarray,
#     ) -> np.ndarray:
#         gnn_p = stable_sigmoid(gnn)
#         xgb_p = stable_sigmoid(xgb)
#         return np.column_stack([
#             gnn,                      # нормалізовані логіти
#             xgb,
#             gnn_p,                    # ймовірності
#             xgb_p,
#             gnn_p * xgb_p,            # взаємодія
#             np.abs(gnn_p - xgb_p),   # розбіжність
#         ])

#     X_meta_train = make_meta_features(gnn_tr_n, xgb_tr_n)
#     X_meta_val   = make_meta_features(gnn_vl_n, xgb_vl_n)

#     n_pos = int(y_tr.sum())
#     n_neg = int(len(y_tr) - n_pos)

#     exclude      = {"spw_mode"}
#     params_clean = {k: v for k, v in params.items() if k not in exclude}
#     spw_mode     = params.get("spw_mode", "full")

#     if spw_mode == "sqrt":
#         spw = np.sqrt(n_neg / max(n_pos, 1))
#     elif spw_mode == "full":
#         spw = n_neg / max(n_pos, 1)
#     else:
#         spw = 1.0

#     meta_model = make_xgb(**params_clean, scale_pos_weight=spw)
#     meta_model.fit(
#         X_meta_train, y_tr,
#         eval_set = [(X_meta_val, y_vl)],
#         verbose  = 100,
#     )

#     meta_probas = meta_model.predict_proba(X_meta_val)[:, 1]

#     meta_auc = roc_auc_score(y_vl, meta_probas)
#     gnn_auc  = roc_auc_score(y_vl, stable_sigmoid(gnn_vl_n))
#     xgb_auc  = roc_auc_score(y_vl, stable_sigmoid(xgb_vl_n))
#     delta    = meta_auc - max(gnn_auc, xgb_auc)

#     print("─" * 50)
#     print(f"  GNN  only AUC : {gnn_auc:.4f}")
#     print(f"  XGB  only AUC : {xgb_auc:.4f}")
#     print(f"  Stacking AUC  : {meta_auc:.4f}  "
#           f"({'↑' if delta >= 0 else '↓'}{abs(delta):.4f} vs best base)")
#     print("─" * 50)

#     return meta_model, meta_probas




# def compare_models(metrics_dict: dict, y_val) -> pd.DataFrame:
#     """
#     Порівняння всіх моделей в одній таблиці.
#     Приймає вже розраховані metrics з compute_metrics.

#     metrics_dict = {
#         "GNN only":      compute_metrics(stable_sigmoid(val_gnn_logits), y_val, print_report=False),
#         "XGB Baseline":  compute_metrics(xgb_base.predict_proba(X_val)[:, 1], y_val, print_report=False),
#         "XGB + GNN emb": compute_metrics(xgb_gnn.predict_proba(X_val_gnn)[:, 1], y_val, print_report=False),
#         "Ensemble":      compute_metrics(ensemble_probas, y_val, print_report=False),
#         "Stacking":      compute_metrics(stacking_probas, y_val, print_report=False),
#     }
#     """
#     y_np  = np.array(y_val)
#     n_pos = int(y_np.sum())
#     n_neg = int(len(y_np) - n_pos)

#     rows = []
#     for name, m in metrics_dict.items():
#         rows.append({
#             "Model":     name,
#             "AUC":       round(m["auc"],       4),
#             "PR-AUC":    round(m["pr_auc"],    4),
#             "Best thr":  round(m["best_thr"],  2),
#             "F1":        round(m["f1"],        4),
#             "Precision": round(m["precision"], 4),
#             "Recall":    round(m["recall"],    4),
#         })

#     df = pd.DataFrame(rows).set_index("Model")

#     # ── Print ─────────────────────────────────────────────────
#     print("\n" + "=" * 75)
#     print(f"  MODEL COMPARISON  |  {n_pos:,} fraud / {n_neg:,} non-fraud")
#     print("=" * 75)
#     print(df.to_string(float_format=lambda x: f"{x:.4f}"))
#     print("=" * 75)

#     # ── Best per metric ───────────────────────────────────────
#     highlight_cols = [c for c in df.columns if c != "Best thr"]

#     print("\n  Best per metric:")
#     for col in highlight_cols:
#         best_val   = df[col].max()
#         best_model = df[col].idxmax()
#         print(f"    {col:<12}: {best_val:.4f}  ← {best_model}")

#     # ── Delta vs baseline (перша модель) ─────────────────────
#     baseline_name = list(metrics_dict.keys())[0]
#     if len(df) > 1:
#         print(f"\n  Delta vs '{baseline_name}':")
#         for col in ["AUC", "PR-AUC", "F1"]:
#             baseline_val = df.loc[baseline_name, col]
#             for model_name in df.index[1:]:
#                 delta  = df.loc[model_name, col] - baseline_val
#                 symbol = "↑" if delta > 0 else "↓"
#                 print(f"    {model_name:<22} {col:<8}: "
#                       f"{symbol}{abs(delta):.4f}")

#     print()
#     return df





