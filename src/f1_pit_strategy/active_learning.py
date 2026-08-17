import numpy as np
import pandas as pd

from .models import fit_estimator, metrics


def uncertainty(probabilities: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "least_confidence": return 1 - probabilities.max(axis=1)
    if strategy == "margin":
        ordered = np.sort(probabilities, axis=1); return 1 - (ordered[:, -1] - ordered[:, -2])
    if strategy == "entropy":
        safe = np.clip(probabilities, 1e-12, 1); return -(safe * np.log(safe)).sum(axis=1)
    raise ValueError(f"Unknown query strategy: {strategy}")


def run_active_learning(template, train_x, train_y, pool_x, pool_y, test_x, test_y, strategy: str, rounds: int, batch_size: int) -> tuple[pd.DataFrame, object]:
    current_x, current_y = train_x.reset_index(drop=True), train_y.reset_index(drop=True); pool_x, pool_y = pool_x.reset_index(drop=True), pool_y.reset_index(drop=True); model = fit_estimator(template, current_x, current_y); history = [{"query_strategy": strategy, "round": 0, "queried_total": 0, "total_labeled": len(current_y), "pool_remaining": len(pool_x), **metrics(test_y, model.predict(test_x))}]; best_model, best_score = model, history[0]["macro_f1"]
    for round_number in range(1, rounds + 1):
        if pool_x.empty: break
        selected = np.argsort(uncertainty(model.predict_proba(pool_x), strategy))[-min(batch_size, len(pool_x)):][::-1]; current_x = pd.concat([current_x, pool_x.iloc[selected]], ignore_index=True); current_y = pd.concat([current_y, pool_y.iloc[selected]], ignore_index=True); keep = np.ones(len(pool_x), dtype=bool); keep[selected] = False; pool_x, pool_y = pool_x.loc[keep].reset_index(drop=True), pool_y.loc[keep].reset_index(drop=True); model = fit_estimator(template, current_x, current_y); record = {"query_strategy": strategy, "round": round_number, "queried_total": len(selected), "total_labeled": len(current_y), "pool_remaining": len(pool_x), **metrics(test_y, model.predict(test_x))}; history.append(record)
        if record["macro_f1"] > best_score: best_model, best_score = model, record["macro_f1"]
    return pd.DataFrame(history), best_model


def detect_plateau(history: pd.DataFrame, tolerance: float = .005, patience: int = 2) -> dict:
    ordered = history.sort_values("round").reset_index(drop=True); improvements = ordered["macro_f1"].diff().fillna(np.inf)
    for index in range(patience, len(ordered)):
        if (improvements.iloc[index - patience + 1:index + 1].abs() < tolerance).all():
            row = ordered.iloc[index]; return {"round": int(row["round"]), "total_labeled": int(row["total_labeled"]), "macro_f1": float(row["macro_f1"]), "plateaued": True}
    row = ordered.iloc[-1]; return {"round": int(row["round"]), "total_labeled": int(row["total_labeled"]), "macro_f1": float(row["macro_f1"]), "plateaued": False}
