from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED_RAW_TABLES = ("races", "pit_stops", "lap_times", "results", "driver_standings", "constructor_standings", "constructors", "circuits")


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=[r"\N", "NULL", "null", ""])


def load_raw_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in EXPECTED_RAW_TABLES if not (data_dir / f"{name}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing Ergast tables: {', '.join(missing)}")
    return {name: load_csv(data_dir / f"{name}.csv") for name in EXPECTED_RAW_TABLES}


def split_label_pools(frame: pd.DataFrame, labeled_fraction: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled, unlabeled = train_test_split(frame, train_size=labeled_fraction, stratify=frame["strategy"], random_state=random_state)
    return labeled.reset_index(drop=True), unlabeled.reset_index(drop=True)
