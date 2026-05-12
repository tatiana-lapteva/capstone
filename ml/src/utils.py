# src/utils.py
import os
import gc
import time
import random
import numpy as np
import torch
import joblib
import json
import os
import pandas as pd
from pathlib import Path
import sys
sys.path.append('../src')

### =============== DEVICE =============== 
def setup_device() -> torch.device:
    """
    Налаштовує GPU/CPU і повертає device.
    Викликати на початку кожного ноутбука.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True,max_split_size_mb:512"
        )
        torch.cuda.empty_cache()
        gc.collect()

    return device



### =============== SEED =============== 

def set_seed(seed: int = 42) -> None:
    """Фіксує всі джерела випадковості."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)
    # ✅ для PyG NeighborLoader
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"✅ Seed fixed: {seed}")



### =============== TIMER ===============

def timer(label: str):
    """Context manager для вимірювання часу блоку коду."""
    class _Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            mins, secs = divmod(elapsed, 60)
            print(f"⏱  {label}: {int(mins)}m {secs:.1f}s")
    return _Timer()



### =============== MEMORY ===============

def clear_memory() -> None:
    """Звільняє GPU і CPU пам'ять між великими блоками."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        print(f"GPU memory: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated  "
              f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
        


### =============== ARTIFACTS ===============

def save_artifact(
        data:      object,
        save_path: str,
        metadata:  dict = None,
) -> None:
    """
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext  = path.suffix.lower()

    if ext == ".json":
        if isinstance(data, pd.DataFrame):
            payload = data.to_dict(orient="records")
        elif isinstance(data, (list, dict)):
            payload = data
        else:
            payload = str(data)

        artifact = {
            "created_at": pd.Timestamp.now().isoformat(),
            "metadata":   metadata or {},
            "data":       payload,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)

    elif ext == ".pkl":
        joblib.dump(data, path)

    elif ext == ".npy":
        np.save(path, data)

    elif ext == ".pt":
        torch.save(data, path)

    else:
        raise ValueError(f"Unsupported extension: {ext}. Use .json/.pkl/.npy/.pt")

    size_mb = path.stat().st_size / 1e6
    print(f"Artifact saved: {path.name:<40} ({size_mb:.2f} MB)")


def load_artifact(load_path: str, device: str = "cpu") -> object:
    """
    """
    path = Path(load_path)
    assert path.exists(), f"File not found: {path}"
    ext = path.suffix.lower()

    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    elif ext == ".pkl":
        data = joblib.load(path)
    elif ext == ".npy":
        data = np.load(path, allow_pickle=False)
    elif ext == ".pt":
        data = torch.load(path, map_location=device)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    size_mb = path.stat().st_size / 1e6
    print(f"Artifact loaded: {path.name:<40} ({size_mb:.2f} MB)")
    return data

def save_zero_importance_features(
        df_imp:    pd.DataFrame,
        save_path: str = "zero_importance_features.json",
        exclude_prefixes: list = ["gnn_"],  
) -> list:
    """
    Save list of features with zero-importance into JSON file.
    """
    zero_cols = df_imp[df_imp["importance"] == 0.0]["feature"].tolist()

    zero_cols_filtered = [
        c for c in zero_cols
        if not any(c.startswith(p) for p in exclude_prefixes)
    ]

    artifact = {
        "created_at":      pd.Timestamp.now().isoformat(),
        "total_features":  len(df_imp),
        "zero_count":      len(zero_cols_filtered),
        "excluded_prefixes": exclude_prefixes,
        "zero_features":   zero_cols_filtered,
    }

    path = Path(save_path)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Saved {len(zero_cols_filtered)} zero-importance features: {path}")
    return zero_cols_filtered
