"""
Metrics calculation module for Round Robin scheduling.

Provides a single source of truth for:
- Waiting time, turnaround time, response time (avg and max)
- Context switch count and overhead (time units)
- Throughput (processes per simulation time unit)
- CPU utilization (executed time / elapsed time)
- Jain's fairness index computed over per-process slowdown (turnaround / burst)
- Per-process metrics table with sanity checks and formatting
"""

import pandas as pd
import numpy as np


METRIC_COLUMNS = [
    "PID", "Arrival", "Burst", "Start", "Completion",
    "Waiting", "Turnaround", "Response",
]

METRICS_CAPTION = (
    "Turnaround = Completion − Arrival | "
    "Waiting = Turnaround − Burst | "
    "Response = First Start − Arrival"
)


def compute_metrics(results, timeline, context_switch_penalty=1, context_switches=None):
    """
    Compute system-wide scheduling performance metrics.

    Parameters
    ----------
    results : list of Process
        List of completed Process objects.
    timeline : list of dict
        Simulation timeline containing execution, switch, and idle blocks.
    context_switch_penalty : int
        Penalty duration per context switch.
    context_switches : int, optional
        Pre-counted context switches from simulator.

    Returns
    -------
    dict
        Comprehensive metrics dictionary.
    """
    n = len(results)
    if n == 0:
        return {
            "avg_waiting": 0.0,
            "max_waiting": 0.0,
            "avg_turnaround": 0.0,
            "max_turnaround": 0.0,
            "avg_response": 0.0,
            "max_response": 0.0,
            "context_switches": 0,
            "context_switch_overhead": 0,
            "execution_time": 0,
            "idle_time": 0,
            "elapsed_time": 0,
            "makespan": 0,
            "throughput": 0.0,
            "cpu_utilization": 0.0,
            "fairness": 1.0,
            "min_slowdown": 1.0,
            "max_slowdown": 1.0,
        }

    # Time measurements
    execution_time = sum(
        item["end"] - item["start"]
        for item in timeline
        if item.get("kind") == "execution"
    )
    context_switch_time = sum(
        item["end"] - item["start"]
        for item in timeline
        if item.get("kind") == "switch"
    )
    idle_time = sum(
        item["end"] - item["start"]
        for item in timeline
        if item.get("kind") == "idle"
    )

    elapsed_time = execution_time + context_switch_time + idle_time
    if context_switches is not None:
        cs_count = int(context_switches)
    else:
        switch_count = sum(1 for item in timeline if item.get("kind") == "switch")
        if switch_count > 0:
            cs_count = switch_count
        else:
            # When penalty == 0, count transitions between different execution processes
            execs = [item["label"] for item in timeline if item.get("kind") == "execution"]
            cs_count = sum(1 for i in range(len(execs) - 1) if execs[i] != execs[i + 1])
    context_switches = cs_count

    # Waiting, Turnaround, Response
    waiting_times = [p.waiting_time for p in results]
    turnaround_times = [p.turnaround_time for p in results]
    response_times = [p.response_time for p in results]

    avg_waiting = float(np.mean(waiting_times))
    max_waiting = float(np.max(waiting_times))
    avg_turnaround = float(np.mean(turnaround_times))
    max_turnaround = float(np.max(turnaround_times))
    avg_response = float(np.mean(response_times))
    max_response = float(np.max(response_times))

    # Makespan and throughput
    # makespan = time of last completion - time of first arrival
    min_arrival = min(p.arrival_time for p in results)
    max_completion = max(p.completion_time for p in results)
    makespan = max(1, max_completion - min_arrival)
    throughput = float(n / makespan)

    # CPU utilization = executed time / elapsed time
    cpu_utilization = float(execution_time / elapsed_time) if elapsed_time > 0 else 0.0

    # Jain's fairness index over per-process slowdown (turnaround / burst):
    # s_i = turnaround_i / burst_i
    # J = (sum s_i)^2 / (n * sum s_i^2), range (0, 1], higher is fairer
    slowdowns = [
        float(p.turnaround_time) / float(p.burst_time) if p.burst_time > 0 else 1.0
        for p in results
    ]
    sum_s = sum(slowdowns)
    sum_s_sq = sum(s ** 2 for s in slowdowns)
    if sum_s_sq > 0:
        fairness = float((sum_s ** 2) / (n * sum_s_sq))
    else:
        fairness = 1.0
    fairness = min(1.0, max(0.0, fairness))
    min_slowdown = float(np.min(slowdowns))
    max_slowdown = float(np.max(slowdowns))

    return {
        "avg_waiting": round(avg_waiting, 2),
        "max_waiting": round(max_waiting, 2),
        "avg_turnaround": round(avg_turnaround, 2),
        "max_turnaround": round(max_turnaround, 2),
        "avg_response": round(avg_response, 2),
        "max_response": round(max_response, 2),
        "context_switches": int(context_switches),
        "context_switch_overhead": int(context_switch_time),
        "execution_time": int(execution_time),
        "idle_time": int(idle_time),
        "elapsed_time": int(elapsed_time),
        "makespan": int(makespan),
        "throughput": round(throughput, 4),
        "cpu_utilization": round(cpu_utilization, 4),
        "fairness": round(fairness, 4),
        "min_slowdown": round(min_slowdown, 2),
        "max_slowdown": round(max_slowdown, 2),
    }


def _compute_executed_by_pid(timeline):
    """
    Compute the total executed CPU time per PID by summing only execution events.
    This is the correct check: sum(end - start) over all execution events for that PID.
    NOT completion - first_start (which incorrectly includes waiting gaps between turns).
    """
    executed = {}
    for ev in timeline:
        if ev.get("kind") != "execution":
            continue
        pid = ev["label"]
        executed[pid] = executed.get(pid, 0) + (ev["end"] - ev["start"])
    return executed


def format_process_table(results, timeline):
    """
    Build the per-process metrics table from the Gantt timeline.

    Start = first execution slice start
    Completion = last execution slice end
    Turnaround = Completion - Arrival
    Waiting = Turnaround - Burst
    Response = Start - Arrival

    Sanity check for executed time:
        sum(end - start) over all execution slices for this PID == Burst
        (NOT completion - first_start, which includes inter-turn waiting gaps)

    Returns (data_df, summary_dict, warnings)
    where data_df has only data rows (no Average row) for clean CSV export,
    and summary_dict has the average values.
    """
    executed_by_pid = _compute_executed_by_pid(timeline)

    by_pid = {}
    for p in results:
        by_pid[p.pid] = {
            "Arrival": int(p.arrival_time),
            "Burst": int(p.burst_time),
            "Start": None,
            "Completion": None,
        }

    for event in timeline:
        if event.get("kind") != "execution":
            continue
        pid = event["label"]
        if pid not in by_pid:
            continue
        entry = by_pid[pid]
        start_t = int(event["start"])
        end_t = int(event["end"])
        if entry["Start"] is None:
            entry["Start"] = start_t
        entry["Completion"] = end_t

    rows = []
    warnings = []
    for pid, entry in by_pid.items():
        start = entry["Start"] if entry["Start"] is not None else 0
        completion = entry["Completion"] if entry["Completion"] is not None else 0
        arrival = entry["Arrival"]
        burst = entry["Burst"]
        executed = executed_by_pid.get(pid, 0)

        turnaround = completion - arrival
        waiting = turnaround - burst
        response = start - arrival

        # F1 corrected sanity check: sum of execution slices == burst
        if executed != burst:
            warnings.append(
                (pid, f"Sum of CPU execution slices ({executed}) ≠ burst time ({burst}).")
            )
        if not (0 <= waiting <= turnaround):
            warnings.append(
                (pid, f"Waiting time {waiting} violates 0 ≤ Waiting ≤ Turnaround ({turnaround}).")
            )

        rows.append({
            "PID": pid,
            "Arrival": int(arrival),
            "Burst": int(burst),
            "Start": int(start),
            "Completion": int(completion),
            "Waiting": int(waiting),
            "Turnaround": int(turnaround),
            "Response": int(response),
        })

    data_df = pd.DataFrame(rows, columns=METRIC_COLUMNS)

    if data_df.empty:
        summary = {"Waiting": 0.0, "Turnaround": 0.0, "Response": 0.0}
        return data_df, summary, warnings

    n = len(data_df)
    summary = {
        "Waiting": round(data_df["Waiting"].sum() / n, 2),
        "Turnaround": round(data_df["Turnaround"].sum() / n, 2),
        "Response": round(data_df["Response"].sum() / n, 2),
    }
    return data_df, summary, warnings


def reference_round_robin(processes, time_quantum, context_switch_penalty):
    """
    Independent reference Round Robin implementation for verification.
    Uses same tie-break rules and conventions as RoundRobinSimulator.
    Returns list of dicts with per-process metrics.
    """
    from process import Process
    procs = sorted(
        [{"pid": p.pid, "arrival": p.arrival_time, "burst": p.burst_time,
          "remaining": p.burst_time, "start": -1, "completion": -1,
          "response": -1, "waiting": 0, "turnaround": 0}
         for p in processes],
        key=lambda p: p["arrival"]
    )
    t = 0
    rq = []
    next_idx = 0
    total = len(procs)
    completed = 0
    timeline_ref = []
    recent_processes = []

    while completed < total:
        while next_idx < total and procs[next_idx]["arrival"] <= t:
            rq.append(procs[next_idx])
            next_idx += 1

        if not rq:
            if next_idx < total:
                t = procs[next_idx]["arrival"]
            continue

        if callable(time_quantum):
            state_rq = []
            for p in rq:
                po = Process(p["pid"], p["arrival"], p["burst"])
                po.remaining_burst = p["remaining"]
                po.start_time = p["start"]
                po.response_time = p["response"]
                state_rq.append(po)
            state = {
                "ready_queue": state_rq,
                "recent_history": list(recent_processes),
                "current_time": t,
            }
            chosen_q = int(max(1, min(20, round(float(time_quantum(state))))))
        else:
            chosen_q = int(time_quantum)

        curr = rq.pop(0)
        if curr["start"] == -1:
            curr["start"] = t
            curr["response"] = t - curr["arrival"]

        exec_t = min(chosen_q, curr["remaining"])
        t += exec_t
        curr["remaining"] -= exec_t

        ro = Process(curr["pid"], curr["arrival"], curr["burst"])
        ro.remaining_burst = curr["remaining"]
        recent_processes.append(ro)
        if len(recent_processes) > 10:
            recent_processes.pop(0)

        # Enqueue arrivals during slice
        while next_idx < total and procs[next_idx]["arrival"] <= t:
            rq.append(procs[next_idx])
            next_idx += 1

        timeline_ref.append({"start": t - exec_t, "end": t, "label": curr["pid"], "kind": "execution", "quantum": chosen_q})

        if curr["remaining"] == 0:
            curr["completion"] = t
            curr["turnaround"] = t - curr["arrival"]
            curr["waiting"] = curr["turnaround"] - curr["burst"]
            completed += 1
            next_to_run = rq[0] if rq else None
        else:
            rq.append(curr)
            next_to_run = rq[0]

        if rq and next_to_run is not None and next_to_run is not curr:
            t += context_switch_penalty
            while next_idx < total and procs[next_idx]["arrival"] <= t:
                rq.append(procs[next_idx])
                next_idx += 1

    return procs, timeline_ref
