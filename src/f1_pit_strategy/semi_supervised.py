from collections import Counter

import numpy as np
import pandas as pd

from .models import fit_estimator, metrics


def run_pseudo_labeling(template, train_x, train_y, pool_x, test_x, test_y, threshold: float, rounds: int) -> tuple[pd.DataFrame, object]:
    current_x, current_y, pool = train_x.reset_index(drop=True), train_y.reset_index(drop=True), pool_x.reset_index(drop=True)
    model = fit_estimator(template, current_x, current_y); history = [{"threshold": threshold, "round": 0, "pool_remaining": len(pool), "added_total": 0, "total_labeled": len(current_y), **metrics(test_y, model.predict(test_x))}]; best_model, best_score = model, history[0]["macro_f1"]
    for round_number in range(1, rounds + 1):
        probabilities = model.predict_proba(pool); confidence = probabilities.max(axis=1); classes = model.named_steps["model"].classes_; predictions = classes[probabilities.argmax(axis=1)]; selected = np.flatnonzero(confidence >= threshold)
        if not len(selected): break
        additions = Counter(predictions[selected]); current_x = pd.concat([current_x, pool.iloc[selected]], ignore_index=True); current_y = pd.concat([current_y, pd.Series(predictions[selected], name="strategy")], ignore_index=True); keep = np.ones(len(pool), dtype=bool); keep[selected] = False; pool = pool.loc[keep].reset_index(drop=True); model = fit_estimator(template, current_x, current_y)
        record = {"threshold": threshold, "round": round_number, "pool_remaining": len(pool), "added_total": len(selected), "total_labeled": len(current_y), **{f"added_{key}": value for key, value in additions.items()}, **metrics(test_y, model.predict(test_x))}; history.append(record)
        if record["macro_f1"] > best_score: best_model, best_score = model, record["macro_f1"]
        if len(selected) < 50: break
    return pd.DataFrame(history), best_model
