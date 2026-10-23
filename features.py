"""
Shared feature extraction logic for Adaptive Round Robin Scheduling.

Used identically by:
- round_robin.py (live dispatch-time feature extraction)
- features.py / training data generation
- train_model.py
- app.py (dashboard visualization)
"""

import pandas as pd

FEATURE_NAMES = [
    "ready_queue_length",
    "average_cpu_burst",
    "recent_workload_behavior",
]


def extract_features(ready_queue, recent_history):
    """
    Extract live dispatch-time features from the scheduler state.

    Parameters
    ----------
    ready_queue : list or deque of Process objects
        Processes currently in the ready queue, INCLUDING the process about to run.
    recent_history : list or deque of Process objects
        Recent processes executed/completed (window size up to 10).

    Returns
    -------
    dict
        Dictionary containing:
        - ready_queue_length: int, total count of processes in ready queue
        - average_cpu_burst: float, mean remaining burst of processes in ready queue
        - recent_workload_behavior: float in [0.0, 1.0], fraction of the last N<=10
          processes that were short (burst_time <= 5), indicating I/O-bound frequency.
    """
    queue_len = len(ready_queue)
    if queue_len > 0:
        avg_burst = sum(p.remaining_burst for p in ready_queue) / float(queue_len)
    else:
        avg_burst = 0.0

    if recent_history and len(recent_history) > 0:
        short_count = sum(1 for p in recent_history if p.burst_time <= 5)
        recent_behavior = short_count / float(len(recent_history))
    else:
        recent_behavior = 0.0

    return {
        "ready_queue_length": int(queue_len),
        "average_cpu_burst": float(round(avg_burst, 4)),
        "recent_workload_behavior": float(round(recent_behavior, 4)),
    }


def features_to_dataframe(feature_dict):
    """
    Convert a feature dictionary into a 1-row DataFrame with the exact column order.
    """
    return pd.DataFrame([[
        feature_dict["ready_queue_length"],
        feature_dict["average_cpu_burst"],
        feature_dict["recent_workload_behavior"],
    ]], columns=FEATURE_NAMES)


def verify_feature_names(model):
    """
    Verify that the model was trained on the exact feature names and order.
    Raises ValueError with a clear explanation if there is a mismatch.
    """
    if hasattr(model, "feature_names_in_"):
        expected = list(FEATURE_NAMES)
        actual = list(model.feature_names_in_)
        if actual != expected:
            raise ValueError(
                f"Model feature mismatch! Expected feature columns {expected}, "
                f"but loaded model has {actual}. Retrain model with train_model.py."
            )
    return True
