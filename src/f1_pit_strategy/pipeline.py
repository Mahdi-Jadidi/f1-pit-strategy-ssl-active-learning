from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .active_learning import detect_plateau, run_active_learning
from .config import ExperimentConfig
from .data import load_raw_tables, split_label_pools
from .features import build_feature_table
from .labeling import make_strategy_labels
from .models import feature_columns, model_templates, select_model
from .semi_supervised import run_pseudo_labeling


def prepare_dataset(data_dir: Path, output_dir: Path, labeled_fraction: float = .08, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled, unlabeled = split_label_pools(make_strategy_labels(build_feature_table(load_raw_tables(data_dir))), labeled_fraction, random_state); output_dir.mkdir(parents=True, exist_ok=True); labeled.to_csv(output_dir / "labeled.csv", index=False); unlabeled.to_csv(output_dir / "unlabeled.csv", index=False); return labeled, unlabeled


def run_pipeline(config: ExperimentConfig) -> pd.DataFrame:
    labeled = pd.read_csv(config.data_dir / "labeled.csv"); unlabeled = pd.read_csv(config.data_dir / "unlabeled.csv"); numeric, categorical = feature_columns(labeled); columns = numeric + categorical
    train, temporary = train_test_split(labeled, test_size=.2, stratify=labeled.strategy, random_state=config.random_state); validation, test = train_test_split(temporary, test_size=.5, stratify=temporary.strategy, random_state=config.random_state)
    train_x, train_y = train[columns], train.strategy; validation_x, validation_y = validation[columns], validation.strategy; test_x, test_y = test[columns], test.strategy
    templates = model_templates(labeled, config.random_state); best_name, baseline_model, baseline_results = select_model(templates, train_x, train_y, validation_x, validation_y); selected_template = templates[best_name]
    pseudo_histories, pseudo_models = [], []
    for threshold in config.pseudo_thresholds:
        history, model = run_pseudo_labeling(selected_template, train_x, train_y, unlabeled[columns], test_x, test_y, threshold, config.pseudo_rounds); pseudo_histories.append(history); pseudo_models.append((history.macro_f1.max(), model, threshold))
    pseudo_results = pd.concat(pseudo_histories, ignore_index=True); best_pseudo = max(pseudo_models, key=lambda item: item[0])
    active_histories, active_models = [], []
    if config.oracle_unlabeled and "strategy" in unlabeled:
        for strategy in ("least_confidence", "margin", "entropy"):
            history, model = run_active_learning(selected_template, train_x, train_y, unlabeled[columns], unlabeled.strategy, test_x, test_y, strategy, config.active_rounds, config.active_batch_size); active_histories.append(history); active_models.append((history.macro_f1.max(), model, strategy))
    config.output_dir.mkdir(parents=True, exist_ok=True); baseline_results.to_csv(config.output_dir / "baseline_metrics.csv", index=False); pseudo_results.to_csv(config.output_dir / "pseudo_labeling_history.csv", index=False); joblib.dump(baseline_model, config.output_dir / "baseline_model.joblib"); joblib.dump(best_pseudo[1], config.output_dir / "pseudo_label_model.joblib")
    summaries = [{"method": f"baseline_{best_name}", "macro_f1": baseline_results.iloc[0].macro_f1}, {"method": f"pseudo_threshold_{best_pseudo[2]}", "macro_f1": best_pseudo[0]}]; plateaus = [{"method": f"pseudo_threshold_{threshold}", **detect_plateau(history)} for threshold, history in ((value, pseudo_results[pseudo_results.threshold == value]) for value in config.pseudo_thresholds)]
    if active_histories:
        active_results = pd.concat(active_histories, ignore_index=True); active_results.to_csv(config.output_dir / "active_learning_history.csv", index=False); best_active = max(active_models, key=lambda item: item[0]); joblib.dump(best_active[1], config.output_dir / "active_learning_model.joblib"); summaries.append({"method": f"active_{best_active[2]}", "macro_f1": best_active[0]}); plateaus.extend({"method": f"active_{strategy}", **detect_plateau(active_results[active_results.query_strategy == strategy])} for strategy in active_results.query_strategy.unique())
    summary = pd.DataFrame(summaries).sort_values("macro_f1", ascending=False); summary.to_csv(config.output_dir / "comparative_summary.csv", index=False); pd.DataFrame(plateaus).to_csv(config.output_dir / "plateau_summary.csv", index=False); return summary
