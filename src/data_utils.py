import pandas as pd
from pathlib import Path
import zipfile

def extract_zip(zip_path: str, extract_dir: str):
    """Extract dataset.zip if not already extracted."""
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    return extract_dir

def load_daily_pickles(pkl_dir: str) -> pd.DataFrame:
    """Load all daily .pkl files and concat."""
    pkl_dir = Path(pkl_dir)
    dfs = []
    for fp in sorted(pkl_dir.glob("*.pkl")):
        dfs.append(pd.read_pickle(fp))
    return pd.concat(dfs, ignore_index=True)

def save_csv(df: pd.DataFrame, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
