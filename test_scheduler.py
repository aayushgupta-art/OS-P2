"""
Comprehensive test suite for Round Robin Simulator and Adaptive Quantum Scheduler.
Runnable with: python test_scheduler.py
Uses plain asserts and detailed diagnostic messages.
"""

import os
import sys
import copy
import random
import joblib
import pandas as pd
import numpy as np

from process import Process
from round_robin import RoundRobinSimulator, FixedQuantumPolicy, AdaptiveQuantumPolicy
from features import extract_features, features_to_dataframe, FEATURE_NAMES, verify_feature_names
from metrics import compute_metrics, format_process_table

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "adaptive_quantum_model.pkl")


class ReferenceRoundRobin:
    """
    Independent reference implementation of Round Robin scheduler
    built strictly according to OS scheduling specification (A2/A3 semantics).
    """

    def __init__(self, time_quantum=4, context_switch_penalty=1, quanta_sequence=None):
        self.time_quantum = time_quantum
        self.context_switch_penalty = context_switch_penalty
        self.quanta_sequence = list(quanta_sequence) if quanta_sequence is not None else None
        self.dispatch_idx = 0

    def run(self, processes):
        # Sort by arrival
        procs = sorted([Process(p.pid, p.arrival_time, p.burst_time) for p in processes],
                       key=lambda p: p.arrival_time)
        t = 0
        rq = []
        completed = []
        timeline = []
        next_unarr = 0
        total = len(procs)
        context_switches = 0

        while len(completed) < total:
            # Enqueue arrivals up to current time
            while next_unarr < total and procs[next_unarr].arrival_time <= t:
                rq.append(procs[next_unarr])
                next_unarr += 1

            # Idle gap if ready queue is empty
            if not rq:
                if next_unarr < total:
                    if t < procs[next_unarr].arrival_time:
                        timeline.append({
                            "start": t,
                            "end": procs[next_unarr].arrival_time,
                            "label": "Idle",
                            "kind": "idle",
                            "ready_queue": [],
                        })
                        t = procs[next_unarr].arrival_time
                continue

            # Determine quantum
            if self.quanta_sequence is not None and self.dispatch_idx < len(self.quanta_sequence):
                q = self.quanta_sequence[self.dispatch_idx]
                self.dispatch_idx += 1
            elif callable(self.time_quantum):
                q = self.time_quantum(rq)
            else:
                q = self.time_quantum

            curr = rq.pop(0)
            if curr.start_time == -1:
                curr.start_time = t
                curr.response_time = t - curr.arrival_time

            exec_time = min(q, curr.remaining_burst)
            exec_start = t
            t += exec_time
            curr.remaining_burst -= exec_time

            # Enqueue arrivals during slice BEFORE preemption
            while next_unarr < total and procs[next_unarr].arrival_time <= t:
                rq.append(procs[next_unarr])
                next_unarr += 1

            timeline.append({
                "start": exec_start,
                "end": t,
                "label": curr.pid,
                "kind": "execution",
                "ready_queue": [p.pid for p in rq],
            })

            if curr.remaining_burst == 0:
                curr.completion_time = t
                curr.turnaround_time = curr.completion_time - curr.arrival_time
                curr.waiting_time = curr.turnaround_time - curr.burst_time
                completed.append(curr)
                next_proc = rq[0] if rq else None
            else:
                rq.append(curr)
                next_proc = rq[0]

            # Context switch only if next process is different
            if rq and next_proc is not None and next_proc is not curr:
                cs_start = t
                t += self.context_switch_penalty
                context_switches += 1

                # Enqueue arrivals during CS
                while next_unarr < total and procs[next_unarr].arrival_time <= t:
                    rq.append(procs[next_unarr])
                    next_unarr += 1

                if self.context_switch_penalty > 0:
                    timeline.append({
                        "start": cs_start,
                        "end": t,
                        "label": "CS",
                        "kind": "switch",
                        "ready_queue": [p.pid for p in rq],
                    })

        return completed, timeline, context_switches


def check_timeline_integrity(timeline, total_burst):
    """Verify timeline has no gaps or overlaps, and executed time == sum of bursts."""
    for i in range(len(timeline) - 1):
        assert timeline[i]["end"] == timeline[i + 1]["start"], (
            f"Timeline gap or overlap at event {i} ({timeline[i]}) to {i+1} ({timeline[i+1]})"
        )
    exec_time = sum(ev["end"] - ev["start"] for ev in timeline if ev["kind"] == "execution")
    assert exec_time == total_burst, f"Executed time ({exec_time}) != total burst ({total_burst})"


def check_process_invariants(procs):
    """Verify Turnaround = Completion - Arrival, Waiting = Turnaround - Burst, Response = Start - Arrival."""
    for p in procs:
        assert p.completion_time >= p.start_time >= p.arrival_time >= 0, f"Process timing sequence violated for {p.pid}"
        assert p.turnaround_time == p.completion_time - p.arrival_time, f"Turnaround mismatch for {p.pid}"
        assert p.waiting_time == p.turnaround_time - p.burst_time, f"Waiting mismatch for {p.pid}"
        assert p.response_time == p.start_time - p.arrival_time, f"Response mismatch for {p.pid}"
        assert 0 <= p.waiting_time <= p.turnaround_time, f"Waiting {p.waiting_time} outside 0..turnaround {p.turnaround_time} for {p.pid}"


def test_1_reference_comparison():
    print("Test 1: Comparing RoundRobinSimulator against independent Reference implementation...")
    rng = random.Random(42)
    for trial in range(10):
        n = rng.randint(4, 15)
        workload = [
            Process(f"P{i}", rng.randint(0, 30), rng.randint(1, 25))
            for i in range(1, n + 1)
        ]
        q = rng.choice([2, 4, 7, 10])
        penalty = rng.choice([0, 1, 2])

        sim = RoundRobinSimulator(time_quantum=q, context_switch_penalty=penalty)
        sim_res = sim.run([Process(p.pid, p.arrival_time, p.burst_time) for p in workload])

        ref = ReferenceRoundRobin(time_quantum=q, context_switch_penalty=penalty)
        ref_res, ref_timeline, ref_switches = ref.run(workload)

        # Sort both by PID
        sim_map = {p.pid: p for p in sim_res}
        ref_map = {p.pid: p for p in ref_res}

        for pid in sim_map:
            sp = sim_map[pid]
            rp = ref_map[pid]
            assert sp.start_time == rp.start_time, f"Trial {trial}: Start mismatch for {pid}: {sp.start_time} vs {rp.start_time}"
            assert sp.completion_time == rp.completion_time, f"Trial {trial}: Completion mismatch for {pid}: {sp.completion_time} vs {rp.completion_time}"
            assert sp.waiting_time == rp.waiting_time, f"Trial {trial}: Waiting mismatch for {pid}: {sp.waiting_time} vs {rp.waiting_time}"
            assert sp.turnaround_time == rp.turnaround_time, f"Trial {trial}: Turnaround mismatch for {pid}: {sp.turnaround_time} vs {rp.turnaround_time}"
            assert sp.response_time == rp.response_time, f"Trial {trial}: Response mismatch for {pid}: {sp.response_time} vs {rp.response_time}"

        assert sim.context_switches == ref_switches, f"Trial {trial}: Context switch mismatch: {sim.context_switches} vs {ref_switches}"
        check_timeline_integrity(sim.timeline, sum(p.burst_time for p in workload))
    print("  -> Passed: Reference comparison matched perfectly across all trials.")


def test_2_process_invariants():
    print("Test 2: Verifying mathematical process invariants...")
    workload = [
        Process("P1", 0, 8),
        Process("P2", 1, 14),
        Process("P3", 2, 2),
        Process("P4", 4, 19),
        Process("P5", 8, 6),
    ]
    sim = RoundRobinSimulator(time_quantum=4, context_switch_penalty=1)
    results = sim.run(workload)
    check_process_invariants(results)
    check_timeline_integrity(sim.timeline, sum(p.burst_time for p in workload))
    print("  -> Passed: Process invariants hold.")


def test_3_regression_presets():
    print("Test 3: Regression test on 5-process preset with penalty 1...")
    # Expected: (Start, Completion, Waiting, Turnaround, Response)
    expected_q4 = {
        "P1": (0, 22, 14, 22, 0),
        "P2": (5, 53, 38, 52, 4),
        "P3": (10, 12, 8, 10, 8),
        "P4": (13, 61, 38, 57, 9),
        "P5": (23, 40, 26, 32, 15),
    }

    procs_q4 = [
        Process("P1", 0, 8),
        Process("P2", 1, 14),
        Process("P3", 2, 2),
        Process("P4", 4, 19),
        Process("P5", 8, 6),
    ]
    sim_q4 = RoundRobinSimulator(time_quantum=4, context_switch_penalty=1)
    results_q4 = sim_q4.run(procs_q4)
    res_map_q4 = {p.pid: (p.start_time, p.completion_time, p.waiting_time, p.turnaround_time, p.response_time) for p in results_q4}
    for pid, exp in expected_q4.items():
        assert res_map_q4[pid] == exp, f"Q=4 mismatch for {pid}: got {res_map_q4[pid]}, expected {exp}"

    expected_q10 = {
        "P1": (0, 8, 0, 8, 0),
        "P2": (9, 45, 30, 44, 8),
        "P3": (20, 22, 18, 20, 18),
        "P4": (23, 55, 32, 51, 19),
        "P5": (34, 40, 26, 32, 26),
    }

    procs_q10 = [
        Process("P1", 0, 8),
        Process("P2", 1, 14),
        Process("P3", 2, 2),
        Process("P4", 4, 19),
        Process("P5", 8, 6),
    ]
    sim_q10 = RoundRobinSimulator(time_quantum=10, context_switch_penalty=1)
    results_q10 = sim_q10.run(procs_q10)
    res_map_q10 = {p.pid: (p.start_time, p.completion_time, p.waiting_time, p.turnaround_time, p.response_time) for p in results_q10}
    for pid, exp in expected_q10.items():
        assert res_map_q10[pid] == exp, f"Q=10 mismatch for {pid}: got {res_map_q10[pid]}, expected {exp}"

    print("  -> Passed: Exact regression values confirmed for Q=4 and Q=10.")


def test_4_ready_queue_snapshot():
    print("Test 4: Ready-queue snapshot & F3 fields verification...")
    procs = [
        Process("P1", 0, 8),
        Process("P2", 1, 14),
        Process("P3", 2, 2),
        Process("P4", 4, 19),
        Process("P5", 8, 6),
    ]
    sim = RoundRobinSimulator(time_quantum=2, context_switch_penalty=0)
    sim.run(procs)

    # Event 1 (P1, 0-2): Waiting at start: none, Arrivals during: P2@1, P3@2, Not arrived: P4@4, P5@8
    ev1 = sim.timeline[0]
    assert ev1["label"] == "P1"
    assert ev1["start"] == 0 and ev1["end"] == 2
    assert ev1["ready_queue"] == []
    assert ev1["arrivals_during"] == [("P2", 1), ("P3", 2)]
    assert ev1["not_arrived"] == [("P4", 4), ("P5", 8)]
    assert ev1["completed_pids"] == []

    # Event 2 (P2, 2-4): Waiting at start: P3, P1, Arrivals during: P4@4, Not arrived: P5@8
    ev2 = sim.timeline[1]
    assert ev2["label"] == "P2"
    assert ev2["start"] == 2 and ev2["end"] == 4
    assert ev2["ready_queue"] == ["P3", "P1"]
    assert ev2["arrivals_during"] == [("P4", 4)]
    assert ev2["not_arrived"] == [("P5", 8)]

    print("  -> Passed: F3 ready_queue snapshot and arrivals_during match specification.")


def test_5_edge_cases():
    print("Test 5: Edge cases...")
    # 5a: Single process -> 0 context switches
    sim = RoundRobinSimulator(time_quantum=3, context_switch_penalty=1)
    p = [Process("P1", 0, 10)]
    sim.run(p)
    assert sim.context_switches == 0, f"Single process had {sim.context_switches} switches, expected 0"
    assert p[0].completion_time == 10
    check_timeline_integrity(sim.timeline, 10)

    # 5b: Idle gap: P1 0 3, P2 10 4
    sim = RoundRobinSimulator(time_quantum=5, context_switch_penalty=1)
    p = [Process("P1", 0, 3), Process("P2", 10, 4)]
    sim.run(p)
    kinds = [ev["kind"] for ev in sim.timeline]
    assert "idle" in kinds, "Idle period missing in idle gap test"
    idle_ev = [ev for ev in sim.timeline if ev["kind"] == "idle"][0]
    assert idle_ev["start"] == 3 and idle_ev["end"] == 10
    check_timeline_integrity(sim.timeline, 7)

    # 5c: Context switch penalty 0
    sim = RoundRobinSimulator(time_quantum=2, context_switch_penalty=0)
    p = [Process("P1", 0, 4), Process("P2", 0, 4)]
    sim.run(p)
    assert all(ev["kind"] != "switch" for ev in sim.timeline), "Found switch event with penalty 0"
    assert sim.context_switches > 0, "Context switch count should still be incremented even if penalty 0"
    check_timeline_integrity(sim.timeline, 8)

    # 5d: All arrive at t=0
    sim = RoundRobinSimulator(time_quantum=4, context_switch_penalty=1)
    p = [Process("P1", 0, 5), Process("P2", 0, 5), Process("P3", 0, 5)]
    sim.run(p)
    check_process_invariants(p)
    check_timeline_integrity(sim.timeline, 15)

    # 5e: Quantum larger than every burst (FCFS degeneracy)
    sim = RoundRobinSimulator(time_quantum=100, context_switch_penalty=1)
    p = [Process("P1", 0, 5), Process("P2", 2, 7), Process("P3", 3, 4)]
    sim.run(p)
    # Each process should complete in its first run without preemption
    p_map = {x.pid: x for x in p}
    assert p_map["P1"].completion_time == 5
    assert p_map["P2"].start_time == 6 and p_map["P2"].completion_time == 13
    assert p_map["P3"].start_time == 14 and p_map["P3"].completion_time == 18
    check_timeline_integrity(sim.timeline, 16)

    # 5f: Quantum = 1
    sim = RoundRobinSimulator(time_quantum=1, context_switch_penalty=1)
    p = [Process("P1", 0, 2), Process("P2", 0, 2)]
    sim.run(p)
    check_process_invariants(p)
    check_timeline_integrity(sim.timeline, 4)
    print("  -> Passed: All edge cases passed.")


def test_6_metrics():
    print("Test 6: Metrics tests on hand-computed examples...")
    p1 = Process("P1", 0, 10)
    p1.completion_time = 10
    p1.turnaround_time = 10
    p1.waiting_time = 0
    p1.response_time = 0

    p2 = Process("P2", 0, 10)
    p2.completion_time = 20
    p2.turnaround_time = 20
    p2.waiting_time = 10
    p2.response_time = 10

    # Equal slowdowns: P1 turnaround/burst = 10/10 = 1.0, P2 = 20/10 = 2.0
    # J = (1 + 2)^2 / (2 * (1 + 4)) = 9 / 10 = 0.90
    timeline = [
        {"start": 0, "end": 10, "label": "P1", "kind": "execution"},
        {"start": 10, "end": 20, "label": "P2", "kind": "execution"},
    ]
    m = compute_metrics([p1, p2], timeline, context_switch_penalty=0)
    assert m["fairness"] == 0.9, f"Fairness expected 0.9, got {m['fairness']}"
    assert m["cpu_utilization"] == 1.0, f"Utilization expected 1.0, got {m['cpu_utilization']}"
    # throughput = 2 / 20 = 0.1 processes per time unit
    assert m["throughput"] == 0.1, f"Throughput expected 0.1, got {m['throughput']}"

    # Equal slowdown case -> fairness must equal 1.0
    p3 = Process("P3", 0, 5)
    p3.completion_time = 10
    p3.turnaround_time = 10  # slowdown = 10/5 = 2.0
    p4 = Process("P4", 0, 10)
    p4.completion_time = 20
    p4.turnaround_time = 20  # slowdown = 20/10 = 2.0
    m_equal = compute_metrics([p3, p4], timeline, context_switch_penalty=0)
    assert m_equal["fairness"] == 1.0, f"Equal slowdown fairness expected 1.0, got {m_equal['fairness']}"
    print("  -> Passed: Metrics match hand-computed values.")


def test_7_adaptive_policy():
    print("Test 7: Adaptive policy tests...")
    model = joblib.load(MODEL_PATH)
    verify_feature_names(model)

    policy = AdaptiveQuantumPolicy(model)
    # Test on empty queue state
    empty_state = {"ready_queue": [], "recent_history": []}
    q_empty = policy(empty_state)
    assert isinstance(q_empty, int) and 1 <= q_empty <= 20, f"Invalid quantum on empty state: {q_empty}"

    # Test on one process queue
    one_state = {"ready_queue": [Process("P1", 0, 7)], "recent_history": []}
    q_one = policy(one_state)
    assert isinstance(q_one, int) and 1 <= q_one <= 20, f"Invalid quantum on single process: {q_one}"

    # Full adaptive simulation
    workload = [
        Process("P1", 0, 8),
        Process("P2", 1, 14),
        Process("P3", 2, 2),
        Process("P4", 4, 19),
        Process("P5", 8, 6),
    ]
    sim = RoundRobinSimulator(quantum_policy=policy, context_switch_penalty=1)
    results = sim.run([Process(p.pid, p.arrival_time, p.burst_time) for p in workload])

    # Invariants
    check_process_invariants(results)
    check_timeline_integrity(sim.timeline, sum(p.burst_time for p in workload))

    # Replay with ReferenceRoundRobin using exact logged per-dispatch quanta
    logged_quanta = [ev["quantum"] for ev in sim.timeline if ev["kind"] == "execution"]
    ref = ReferenceRoundRobin(context_switch_penalty=1, quanta_sequence=logged_quanta)
    ref_results, ref_timeline, ref_switches = ref.run(workload)

    sim_map = {p.pid: p for p in results}
    ref_map = {p.pid: p for p in ref_results}
    for pid in sim_map:
        sp = sim_map[pid]
        rp = ref_map[pid]
        assert sp.start_time == rp.start_time
        assert sp.completion_time == rp.completion_time
        assert sp.waiting_time == rp.waiting_time
        assert sp.turnaround_time == rp.turnaround_time
        assert sp.response_time == rp.response_time

    assert sim.context_switches == ref_switches
    print("  -> Passed: Adaptive policy executes correctly and matches reference replay.")


def test_8_hand_verified_cases():
    print("Test 8: Hand-verified test cases (G1 & Verification specs)...")
    def make_procs():
        return [
            Process("P1", 0, 8),
            Process("P2", 1, 14),
            Process("P3", 2, 2),
            Process("P4", 4, 19),
            Process("P5", 8, 6),
        ]

    # Hand-verified: Penalty 1, Q=5
    p1 = make_procs()
    sim1 = RoundRobinSimulator(time_quantum=5, context_switch_penalty=1)
    res1 = sim1.run(p1)
    m1 = compute_metrics(res1, sim1.timeline, 1, context_switches=sim1.context_switches)
    expected_p1 = {
        "P1": (0, 24, 16, 24, 0),
        "P2": (6, 49, 34, 48, 5),
        "P3": (12, 14, 10, 12, 10),
        "P4": (15, 59, 36, 55, 11),
        "P5": (25, 44, 30, 36, 17),
    }
    for p in res1:
        s, c, w, t, r = expected_p1[p.pid]
        assert (p.start_time, p.completion_time, p.waiting_time, p.turnaround_time, p.response_time) == (s, c, w, t, r)
    assert round(m1["avg_response"], 2) == 8.6
    assert round(m1["avg_waiting"], 2) == 25.2
    assert round(m1["avg_turnaround"], 2) == 35.0
    assert sim1.context_switches == 10
    assert m1["context_switches"] == 10
    assert m1["elapsed_time"] == 59
    assert abs(m1["throughput"] - 5/59) < 1e-4
    assert abs(m1["cpu_utilization"] - 49/59) < 1e-4

    # Hand-verified: Penalty 1, Q=2
    p2 = make_procs()
    sim2 = RoundRobinSimulator(time_quantum=2, context_switch_penalty=1)
    res2 = sim2.run(p2)
    m2 = compute_metrics(res2, sim2.timeline, 1, context_switches=sim2.context_switches)
    assert sim2.context_switches == 21
    assert m2["context_switches"] == 21
    assert m2["elapsed_time"] == 70
    assert round(m2["avg_response"], 2) == 4.8
    assert round(m2["avg_waiting"], 2) == 31.0
    assert round(m2["avg_turnaround"], 2) == 40.8

    # Hand-verified: Penalty 1, Q=19
    p19 = make_procs()
    sim19 = RoundRobinSimulator(time_quantum=19, context_switch_penalty=1)
    res19 = sim19.run(p19)
    m19 = compute_metrics(res19, sim19.timeline, 1, context_switches=sim19.context_switches)
    assert sim19.context_switches == 4
    assert m19["context_switches"] == 4
    assert m19["elapsed_time"] == 53
    assert round(m19["avg_response"], 2) == 18.4
    assert round(m19["avg_waiting"], 2) == 18.4
    assert round(m19["avg_turnaround"], 2) == 28.2
    assert round(m19["fairness"], 4) == 0.5653

    # Hand-verified: Penalty 0, Q=5
    p0 = make_procs()
    sim0 = RoundRobinSimulator(time_quantum=5, context_switch_penalty=0)
    res0 = sim0.run(p0)
    m0 = compute_metrics(res0, sim0.timeline, 0, context_switches=sim0.context_switches)
    expected_p0 = {
        "P1": (0, 20, 12, 20, 0),
        "P2": (5, 40, 25, 39, 4),
        "P3": (10, 12, 8, 10, 8),
        "P4": (12, 49, 26, 45, 8),
        "P5": (20, 36, 22, 28, 12),
    }
    for p in res0:
        s, c, w, t, r = expected_p0[p.pid]
        assert (p.start_time, p.completion_time, p.waiting_time, p.turnaround_time, p.response_time) == (s, c, w, t, r)
    assert round(m0["avg_waiting"], 2) == 18.6
    assert round(m0["avg_turnaround"], 2) == 28.4
    assert round(m0["avg_response"], 2) == 6.4
    assert sim0.context_switches == 10
    assert m0["context_switches"] == 10
    assert len(sim0.timeline) == 12

    # Hand-verified: Baseline Q=1, penalty 0 (49 events, 42 switches, ends 49)
    sim_q1_p0 = RoundRobinSimulator(time_quantum=1, context_switch_penalty=0)
    res_q1_p0 = sim_q1_p0.run(make_procs())
    m_q1_p0 = compute_metrics(res_q1_p0, sim_q1_p0.timeline, 0, context_switches=sim_q1_p0.context_switches)
    expected_q1_p0 = {
        "P1": (0, 26, 18, 26, 0),
        "P2": (1, 42, 27, 41, 0),
        "P3": (3, 8, 4, 6, 1),
        "P4": (6, 49, 26, 45, 2),
        "P5": (11, 31, 17, 23, 3),
    }
    for p in res_q1_p0:
        assert (p.start_time, p.completion_time, p.waiting_time, p.turnaround_time, p.response_time) == expected_q1_p0[p.pid]
    assert sim_q1_p0.context_switches == 42
    assert m_q1_p0["context_switches"] == 42
    assert m_q1_p0["elapsed_time"] == 49
    exec_count_q1_p0 = len([e for e in sim_q1_p0.timeline if e.get("kind") == "execution"])
    assert exec_count_q1_p0 == 49

    # Hand-verified: Baseline Q=1, penalty 1 (42 switches, ends 91)
    sim_q1_p1 = RoundRobinSimulator(time_quantum=1, context_switch_penalty=1)
    res_q1_p1 = sim_q1_p1.run(make_procs())
    m_q1_p1 = compute_metrics(res_q1_p1, sim_q1_p1.timeline, 1, context_switches=sim_q1_p1.context_switches)
    expected_q1_p1 = {
        "P1": (0, 55, 47, 55, 0),
        "P2": (2, 83, 68, 82, 1),
        "P3": (6, 15, 11, 13, 4),
        "P4": (10, 91, 68, 87, 6),
        "P5": (16, 57, 43, 49, 8),
    }
    for p in res_q1_p1:
        assert (p.start_time, p.completion_time, p.waiting_time, p.turnaround_time, p.response_time) == expected_q1_p1[p.pid]
    assert sim_q1_p1.context_switches == 42
    assert m_q1_p1["context_switches"] == 42
    assert m_q1_p1["elapsed_time"] == 91

    # Hand-verified: Replayed AI quanta [1,15,15,15,15,1,1,1,1,1,1,1,1,1,1], penalty 0
    class ReplayPolicy:
        def __init__(self, quanta):
            self.quanta = list(quanta)
            self.idx = 0
        def __call__(self, state):
            if self.idx < len(self.quanta):
                q = self.quanta[self.idx]
                self.idx += 1
                return q
            return 1

    sim_replay_p0 = RoundRobinSimulator(quantum_policy=ReplayPolicy([1,15,15,15,15,1,1,1,1,1,1,1,1,1,1]), context_switch_penalty=0)
    res_replay_p0 = sim_replay_p0.run(make_procs())
    m_replay_p0 = compute_metrics(res_replay_p0, sim_replay_p0.timeline, 0, context_switches=sim_replay_p0.context_switches)
    expected_replay_p0 = {
        "P1": (0, 22, 14, 22, 0),
        "P2": (1, 15, 0, 14, 0),
        "P3": (22, 24, 20, 22, 20),
        "P4": (24, 47, 24, 43, 20),
        "P5": (39, 49, 35, 41, 31),
    }
    for p in res_replay_p0:
        assert (p.start_time, p.completion_time, p.waiting_time, p.turnaround_time, p.response_time) == expected_replay_p0[p.pid]
    assert round(m_replay_p0["avg_waiting"], 2) == 18.6
    assert round(m_replay_p0["avg_turnaround"], 2) == 28.4
    assert round(m_replay_p0["avg_response"], 2) == 14.2

    # Hand-verified: Replayed AI quanta [1,10,20,20,10,20,20,20], penalty 1 (7 switches, ends 56)
    sim_replay_p1 = RoundRobinSimulator(quantum_policy=ReplayPolicy([1,10,20,20,10,20,20,20]), context_switch_penalty=1)
    res_replay_p1 = sim_replay_p1.run(make_procs())
    m_replay_p1 = compute_metrics(res_replay_p1, sim_replay_p1.timeline, 1, context_switches=sim_replay_p1.context_switches)
    expected_replay_p1 = {
        "P1": (0, 20, 12, 20, 0),
        "P2": (2, 46, 31, 45, 1),
        "P3": (21, 23, 19, 21, 19),
        "P4": (24, 56, 33, 52, 20),
        "P5": (35, 41, 27, 33, 27),
    }
    for p in res_replay_p1:
        assert (p.start_time, p.completion_time, p.waiting_time, p.turnaround_time, p.response_time) == expected_replay_p1[p.pid]
    assert sim_replay_p1.context_switches == 7
    assert m_replay_p1["context_switches"] == 7
    assert m_replay_p1["elapsed_time"] == 56

    # Test q < 1 validation
    try:
        RoundRobinSimulator(time_quantum=0)
        assert False, "Should have raised ValueError for quantum 0"
    except ValueError as e:
        assert "Time quantum must be >= 1" in str(e)

    try:
        RoundRobinSimulator(time_quantum=-3)
        assert False, "Should have raised ValueError for negative quantum"
    except ValueError as e:
        assert "Time quantum must be >= 1" in str(e)

    sim_q2_p0 = RoundRobinSimulator(time_quantum=2, context_switch_penalty=0)
    sim_q2_p0.run(make_procs())
    assert sim_q2_p0.context_switches == 21

    sim_q5_p0 = RoundRobinSimulator(time_quantum=5, context_switch_penalty=0)
    sim_q5_p0.run(make_procs())
    assert sim_q5_p0.context_switches == 10

    sim_q10_p0 = RoundRobinSimulator(time_quantum=10, context_switch_penalty=0)
    sim_q10_p0.run(make_procs())
    assert sim_q10_p0.context_switches == 6

    # Adjacent execution events transition rule check
    for sim_test in [sim_q1_p0, sim_q1_p1, sim_q2_p0, sim_q5_p0, sim_q10_p0, sim_replay_p0, sim_replay_p1]:
        exec_pids = [ev["label"] for ev in sim_test.timeline if ev.get("kind") == "execution"]
        transitions = sum(1 for i in range(len(exec_pids) - 1) if exec_pids[i] != exec_pids[i + 1])
        assert sim_test.context_switches == transitions, f"CS count {sim_test.context_switches} != transitions {transitions}"

    print("  -> Passed: All hand-verified cases match specification.")


def test_9_pipeline_consistency():
    print("Test 9: G3 Pipeline consistency test...")
    model = joblib.load(MODEL_PATH)
    policy = AdaptiveQuantumPolicy(model)
    workload = [
        Process("P1", 0, 8),
        Process("P2", 1, 14),
        Process("P3", 2, 2),
        Process("P4", 4, 19),
        Process("P5", 8, 6),
    ]
    sim = RoundRobinSimulator(quantum_policy=policy, context_switch_penalty=1)
    sim.run([Process(p.pid, p.arrival_time, p.burst_time) for p in workload])

    exec_events = [ev for ev in sim.timeline if ev.get("kind") == "execution"]
    assert len(exec_events) == len(sim.feature_data), "Feature data count must match execution events"

    for k, (ev, feat) in enumerate(zip(exec_events, sim.feature_data)):
        q_timeline = ev["quantum"]
        q_feat = feat["predicted_quantum"]
        assert q_timeline == q_feat, f"Event {k}: timeline quantum {q_timeline} != feature quantum {q_feat}"

        df = pd.DataFrame([{
            "ready_queue_length": feat["ready_queue_length"],
            "average_cpu_burst": feat["average_cpu_burst"],
            "recent_workload_behavior": feat["recent_workload_behavior"],
        }])
        pred = model.predict(df)[0]
        pred_q = int(max(1, min(20, round(float(pred)))))
        assert q_timeline == pred_q, f"Event {k}: timeline quantum {q_timeline} != model.predict {pred_q}"

    print("  -> Passed: Pipeline Q matches timeline quantum and model.predict for every dispatch.")


def test_10_m_hand_verified_cases():
    print("Test 10: M hand-verified test cases & candidate-cost rules...")
    from create_training_data import evaluate_candidate_costs, find_candidate_match

    def make_procs():
        return [
            Process("P1", 0, 8),
            Process("P2", 1, 14),
            Process("P3", 2, 2),
            Process("P4", 4, 19),
            Process("P5", 8, 6),
        ]

    # 1. Baseline Q=5, penalty 0
    sim_b5 = RoundRobinSimulator(time_quantum=5, context_switch_penalty=0)
    res_b5 = sim_b5.run(make_procs())
    m_b5 = compute_metrics(res_b5, sim_b5.timeline, 0, context_switches=sim_b5.context_switches)

    assert len([e for e in sim_b5.timeline if e.get("kind") == "execution"]) == 12
    assert sim_b5.context_switches == 10
    assert m_b5["elapsed_time"] == 49

    expected_b5 = {
        "P1": (0, 20, 12, 20, 0),
        "P2": (5, 40, 25, 39, 4),
        "P3": (10, 12, 8, 10, 8),
        "P4": (12, 49, 26, 45, 8),
        "P5": (20, 36, 22, 28, 12),
    }
    for p in res_b5:
        assert (p.start_time, p.completion_time, p.waiting_time, p.turnaround_time, p.response_time) == expected_b5[p.pid]
    assert round(m_b5["fairness"], 4) == 0.9037
    assert round(m_b5["min_slowdown"], 2) == 2.37
    assert round(m_b5["max_slowdown"], 2) == 5.00

    # 2. Replayed AI quanta [8, 20, 15, 15, 8, 4], penalty 0
    class ReplayPolicy:
        def __init__(self, quanta):
            self.quanta = list(quanta)
            self.idx = 0
        def __call__(self, state):
            if self.idx < len(self.quanta):
                q = self.quanta[self.idx]
                self.idx += 1
                return q
            return 1

    sim_ai = RoundRobinSimulator(quantum_policy=ReplayPolicy([8, 20, 15, 15, 8, 4]), context_switch_penalty=0)
    res_ai = sim_ai.run(make_procs())
    m_ai = compute_metrics(res_ai, sim_ai.timeline, 0, context_switches=sim_ai.context_switches)

    assert len([e for e in sim_ai.timeline if e.get("kind") == "execution"]) == 6
    assert sim_ai.context_switches == 5
    assert m_ai["elapsed_time"] == 49

    expected_ai = {
        "P1": (0, 8, 0, 8, 0),
        "P2": (8, 22, 7, 21, 7),
        "P3": (22, 24, 20, 22, 20),
        "P4": (24, 49, 26, 45, 20),
        "P5": (39, 45, 31, 37, 31),
    }
    for p in res_ai:
        assert (p.start_time, p.completion_time, p.waiting_time, p.turnaround_time, p.response_time) == expected_ai[p.pid]

    assert round(m_ai["avg_waiting"], 1) == 16.8
    assert round(m_ai["avg_turnaround"], 1) == 26.6
    assert round(m_ai["avg_response"], 1) == 15.6
    assert round(m_ai["fairness"], 4) == 0.5784
    assert round(m_ai["min_slowdown"], 2) == 1.00
    assert round(m_ai["max_slowdown"], 2) == 11.00

    # Features check
    feat_lens = [f["ready_queue_length"] for f in sim_ai.feature_data]
    feat_bursts = [f["average_cpu_burst"] for f in sim_ai.feature_data]
    assert feat_lens == [1, 4, 3, 2, 2, 1]
    assert feat_bursts == [8.0, 10.25, 9.0, 12.5, 5.0, 4.0]

    # Oversized dispatches count
    exec_evs = [ev for ev in sim_ai.timeline if ev.get("kind") == "execution"]
    oversized = 0
    for ev in exec_evs:
        q = ev.get("quantum", 1)
        snap = ev.get("ready_snapshot", [])
        if snap and q > max(p[2] for p in snap):
            oversized += 1
    assert oversized == 2

    # 3. Single ready process candidate equivalence test
    rq_single = [Process("P1", 0, 8)]
    c_tbl1, best_q1, min_c1, is_tied1 = evaluate_candidate_costs(rq_single, 0, penalty=0)
    assert len(c_tbl1) == 1
    assert c_tbl1[0]["cost"] == 0.0
    assert "equivalent" in c_tbl1[0]["label"]
    assert c_tbl1[0]["note"] == "only 1 process observable"

    # Chosen cost and regret match
    matched1 = find_candidate_match(c_tbl1, 8)
    assert matched1["cost"] == 0.0
    regret1 = matched1["cost"] - c_tbl1[0]["cost"]
    assert regret1 == 0.0

    # 4. Event at t=39 candidate collapsing test (ready = P5 rem 6, P4 rem 4)
    p5 = Process("P5", 8, 6); p5.remaining_burst = 6; p5.start_time = -1; p5.response_time = -1
    p4 = Process("P4", 4, 19); p4.remaining_burst = 4; p4.start_time = 24; p4.response_time = 20
    rq_39 = [p5, p4]
    c_tbl39, best_q39, min_c39, _ = evaluate_candidate_costs(rq_39, 39, penalty=0)

    # Candidates < 6 are distinct, candidates >= 6 collapse into one row
    labels_39 = [r["label"] for r in c_tbl39]
    assert "Q=1" in labels_39
    assert "Q=2" in labels_39
    assert "Q=3" in labels_39
    assert "Q=4" in labels_39
    assert "Q=5" in labels_39
    assert any("Q >= 6" in l for l in labels_39)
    # Check that Q=8 or Q=12 do not appear as separate rows
    assert not any(l == "Q=8" for l in labels_39)
    assert not any(l == "Q=12" for l in labels_39)

    # Chosen quantum Q=8 matches the collapsed row Q >= 6
    matched39 = find_candidate_match(c_tbl39, 8)
    assert matched39.get("is_collapsed") is True
    assert matched39["cost"] == min_c39

    print("  -> Passed: M hand-verified cases, AI features, candidate equivalence & collapsing verified.")


if __name__ == "__main__":
    print("\n=======================================================")
    print("      RUNNING SCHEDULER & METRICS TEST SUITE")
    print("=======================================================")
    test_1_reference_comparison()
    test_2_process_invariants()
    test_3_regression_presets()
    test_4_ready_queue_snapshot()
    test_5_edge_cases()
    test_6_metrics()
    test_7_adaptive_policy()
    test_8_hand_verified_cases()
    test_9_pipeline_consistency()
    test_10_m_hand_verified_cases()
    print("=======================================================")
    print("  ALL 10 TEST SUITES PASSED PERFECTLY WITH ZERO ERRORS!")
    print("=======================================================\n")
