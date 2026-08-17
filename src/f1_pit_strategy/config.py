from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    output_dir: Path = Path("outputs")
    random_state: int = 42
    labeled_fraction: float = 0.08
    pseudo_thresholds: tuple[float, ...] = (0.55, 0.60, 0.75, 0.85)
    pseudo_rounds: int = 5
    active_rounds: int = 5
    active_batch_size: int = 40
    oracle_unlabeled: bool = False
