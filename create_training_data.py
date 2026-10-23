"""
Generate training dataset for Adaptive Round Robin Scheduling.

Creates dispatch-time states with features:
- ready_queue_length
- average_cpu_burst
- recent_workload_behavior

Optimal quantum label is selected from candidates {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20}
by simulating the remaining workload from that state and finding the quantum that
minimizes:
    cost = w1 * avg_response + w2 * avg_waiting + w3 * context_switch_overhead
where w1 = 1.0, w2 = 1.0, w3 = 2.0.

Workload seeds are distinct from evaluation workloads to prevent data leakage.
"""

import os
import copy
import random
import numpy as np
import pandas as pd
from process import Process
from features import extract_features, FEATURE_NAMES

CANDIDATE_QUANTA = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
W_RESPONSE = 1.0
W_WAITING = 1.0
W_CS_OVERHEAD = 2.0


def simulate_remaining_cost(ready_queue, unarrived, current_time, quantum, penalty=1):
    """
    Simulate remaining workload from current dispatch state to completion under quantum.
    Returns composite cost = W_RESPONSE*avg_response + W_WAITING*avg_waiting + W_CS_OVERHEAD*cs_overhead.
    """
    rq = [
        Process(p.pid, p.arrival_time, p.burst_time)
        for p in ready_queue
    ]
    # carry over current remaining burst and start time
    for target, source in zip(rq, ready_queue):
        target.remaining_burst = source.remaining_burst
        target.start_time = source.start_time
        target.response_time = source.response_time

    unarr = [
        Process(p.pid, p.arrival_time, p.burst_time)
        for p in unarrived
    ]

    t = current_time
    context_switches = 0
    completed_procs = []
    next_unarr = 0
    total_unarr = len(unarr)

    while rq or next_unarr < total_unarr:
        while next_unarr < total_unarr and unarr[next_unarr].arrival_time <= t:
            rq.append(unarr[next_unarr])
            next_unarr += 1

        if not rq:
            if next_unarr < total_unarr:
                t = unarr[next_unarr].arrival_time
                continue
            break

        curr = rq.pop(0)
        if curr.start_time == -1:
            curr.start_time = t
            curr.response_time = t - curr.arrival_time

        exec_time = min(quantum, curr.remaining_burst)
        t += exec_time
        curr.remaining_burst -= exec_time

        while next_unarr < total_unarr and unarr[next_unarr].arrival_time <= t:
            rq.append(unarr[next_unarr])
            next_unarr += 1

        if curr.remaining_burst == 0:
            curr.completion_time = t
            curr.turnaround_time = curr.completion_time - curr.arrival_time
            curr.waiting_time = curr.turnaround_time - curr.burst_time
            completed_procs.append(curr)
            next_to_run = rq[0] if rq else None
        else:
            rq.append(curr)
            next_to_run = rq[0]

        if rq and next_to_run is not None and next_to_run is not curr:
            t += penalty
            context_switches += 1
            while next_unarr < total_unarr and unarr[next_unarr].arrival_time <= t:
                rq.append(unarr[next_unarr])
                next_unarr += 1

    if not completed_procs:
        return 0.0

    avg_resp = sum(p.response_time for p in completed_procs) / len(completed_procs)
    avg_wait = sum(p.waiting_time for p in completed_procs) / len(completed_procs)
    cs_overhead = context_switches * penalty

    return W_RESPONSE * avg_resp + W_WAITING * avg_wait + W_CS_OVERHEAD * cs_overhead


def evaluate_candidate_costs(ready_queue, current_time, penalty=1, unarrived=None, candidate_quanta=None):
    """
    Evaluate candidate quanta for a dispatch state.
    - Non-clairvoyant lookahead: simulates ONLY the observable ready queue.
    - Uses the specified context-switch penalty.
    - If only 1 process is observable, all candidates are equivalent:
      returns a single collapsed row "Q >= max_rem (equivalent)" with cost=0.0 and regret=0.0.
    - If >1 processes are observable, candidates Q >= max_rem collapse into one row:
      "Q >= max_rem (>= max burst max_rem)".
    - Resolves ties within 1% of min cost by choosing the LARGEST tied quantum (minimizing context switches).
    """
    if candidate_quanta is None:
        candidate_quanta = CANDIDATE_QUANTA
    if not ready_queue:
        return [], 5, 0.0, False

    max_rem = max(p.remaining_burst for p in ready_queue)

    # Special case: only 1 process observable in ready queue
    if len(ready_queue) == 1:
        candidate_q = next((q for q in candidate_quanta if q >= max_rem), candidate_quanta[-1])
        label = f"Q >= {max_rem} (equivalent)"
        cost_results = [{
            "quantum": candidate_q,
            "collapse_threshold": max_rem,
            "label": label,
            "is_collapsed": True,
            "cost": 0.0,
            "is_best": True,
            "is_true_min": True,
            "is_cheapest": True,
            "note": "only 1 process observable",
        }]
        return cost_results, candidate_q, 0.0, False

    # Multiple processes observable: collapse candidates Q >= max_rem
    collapsed_quanta = []
    collapsed_done = False
    for q in candidate_quanta:
        if q < max_rem:
            collapsed_quanta.append((q, False, f"Q={q}"))
        else:
            if not collapsed_done:
                collapsed_quanta.append((q, True, f"Q >= {max_rem} (>= max burst {max_rem})"))
                collapsed_done = True

    if not collapsed_quanta:
        collapsed_quanta = [(candidate_quanta[0], False, f"Q={candidate_quanta[0]}")]

    cost_results = []
    for q, is_collapsed, label in collapsed_quanta:
        # Non-clairvoyant lookahead: unarrived is empty list!
        cost = simulate_remaining_cost(ready_queue, [], current_time, q, penalty=penalty)
        cost_results.append({
            "quantum": q,
            "collapse_threshold": max_rem if is_collapsed else None,
            "label": label,
            "is_collapsed": is_collapsed,
            "cost": cost,
            "note": "",
        })

    true_min_item = min(cost_results, key=lambda x: x["cost"])
    true_min_cost = true_min_item["cost"]
    true_min_quantum = true_min_item["quantum"]

    threshold = true_min_cost * 1.01 if true_min_cost > 0 else 1e-6
    tied_candidates = [r for r in cost_results if r["cost"] <= threshold]
    is_tied = len(tied_candidates) > 1

    # Winner: pick LARGEST quantum among tied candidates within 1% (fewest switches)
    winner_item = max(tied_candidates, key=lambda x: x["quantum"])
    best_quantum = winner_item["quantum"]

    for r in cost_results:
        r["is_best"] = (r["quantum"] == best_quantum)
        r["is_true_min"] = (r["quantum"] == true_min_quantum)
        r["is_cheapest"] = r["is_best"]

    return cost_results, best_quantum, true_min_cost, is_tied


def find_candidate_match(cand_table, quantum):
    """Find the candidate row matching the given quantum (handling collapsed rows)."""
    if not cand_table:
        return None
    if len(cand_table) == 1:
        return cand_table[0]
    for r in cand_table:
        if not r.get("is_collapsed") and r["quantum"] == quantum:
            return r
    for r in cand_table:
        if r.get("is_collapsed") and quantum >= r.get("collapse_threshold", r["quantum"]):
            return r
    return cand_table[0]


def generate_training_workload(seed, w_type, num_procs=15):
    rng = random.Random(seed)
    procs = []
    for i in range(1, num_procs + 1):
        arr = rng.randint(0, num_procs * 4)
        if w_type == "io":
            burst = rng.randint(1, 5)
        elif w_type == "cpu":
            burst = rng.randint(15, 45)
        elif w_type == "mixed":
            burst = rng.randint(1, 5) if rng.random() < 0.5 else rng.randint(15, 45)
        elif w_type == "bursty":
            arr = 0 if i <= 5 else (rng.randint(20, 40) if i <= 10 else rng.randint(60, 90))
            burst = rng.randint(2, 25)
        else:
            burst = rng.randint(2, 30)
        procs.append(Process(f"T{i}", arr, burst))
    procs.sort(key=lambda p: p.arrival_time)
    return procs


def build_dataset(num_workloads=40):
    dataset_rows = []
    workload_types = ["io", "cpu", "mixed", "bursty", "general"]

    for w_idx in range(num_workloads):
        seed = 1000 + w_idx
        w_type = workload_types[w_idx % len(workload_types)]
        n_procs = 8 + (w_idx % 12) * 2  # 8 to 30 processes
        procs = generate_training_workload(seed, w_type, n_procs)

        # Step through workload dispatch by dispatch
        curr_time = 0
        rq = []
        unarrived = list(procs)
        recent_history = []
        window_size = 10

        while rq or unarrived:
            while unarrived and unarrived[0].arrival_time <= curr_time:
                rq.append(unarrived.pop(0))

            if not rq:
                if unarrived:
                    curr_time = unarrived[0].arrival_time
                    continue
                break

            # Live features
            feats = extract_features(rq, recent_history)

            # Evaluate each candidate quantum via evaluate_candidate_costs (collapsing & tie-breaking)
            cand_table, best_q, min_cost, is_tied = evaluate_candidate_costs(
                rq, curr_time, penalty=1
            )

            dataset_rows.append({
                "workload_id": f"w_{w_idx:03d}",
                "ready_queue_length": feats["ready_queue_length"],
                "average_cpu_burst": feats["average_cpu_burst"],
                "recent_workload_behavior": feats["recent_workload_behavior"],
                "optimal_quantum": best_q,
            })

            # Advance by executing best_q
            curr = rq.pop(0)
            if curr.start_time == -1:
                curr.start_time = curr_time
                curr.response_time = curr_time - curr.arrival_time

            exec_time = min(best_q, curr.remaining_burst)
            curr_time += exec_time
            curr.remaining_burst -= exec_time

            while unarrived and unarrived[0].arrival_time <= curr_time:
                rq.append(unarrived.pop(0))

            recent_history.append(curr)
            if len(recent_history) > window_size:
                recent_history.pop(0)

            if curr.remaining_burst == 0:
                curr.completion_time = curr_time
                curr.turnaround_time = curr.completion_time - curr.arrival_time
                curr.waiting_time = curr.turnaround_time - curr.burst_time
                next_to_run = rq[0] if rq else None
            else:
                rq.append(curr)
                next_to_run = rq[0]

            if rq and next_to_run is not None and next_to_run is not curr:
                curr_time += 1  # penalty 1
                while unarrived and unarrived[0].arrival_time <= curr_time:
                    rq.append(unarrived.pop(0))

    df = pd.DataFrame(dataset_rows)
    return df


if __name__ == "__main__":
    print("Generating state-level training data with lookahead cost optimization...")
    df = build_dataset(num_workloads=50)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data.csv")
    df.to_csv(out_path, index=False)
    print(f"training_data.csv saved successfully! Total records: {len(df)}")
    print(f"Distinct workloads: {df['workload_id'].nunique()}")
    print("\nClass distribution of optimal_quantum:")
    print(df["optimal_quantum"].value_counts().sort_index())