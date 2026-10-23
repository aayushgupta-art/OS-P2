Adaptive_RR_Scheduling: Adaptive Time-Quantum Round Robin Scheduling
================================================================================

1. PROJECT OVERVIEW
-------------------
Project 19 investigates whether dynamically selecting the Round Robin time quantum 
improves system responsiveness while minimizing context-switch overhead.
- Phase 1 (Baseline): Configurable fixed-quantum Round Robin tested across small, 
  medium, and large quantum values and various workloads.
- Phase 2 (Proposed): Dynamic AI-driven adaptive time-quantum policy predicting the 
  quantum per dispatch from live ready-queue state and recent history.

2. FILE OVERVIEW
----------------
├── process.py             → Process / PCB data structure and validation
├── round_robin.py         → Round Robin simulator supporting Fixed & Adaptive policies
├── features.py            → Single source of truth for feature extraction & validation
├── metrics.py             → Single source of truth for OS metrics & per-process table
├── create_training_data.py→ Generates dispatch-state training dataset via lookahead simulation
├── train_model.py         → Trains Random Forest classifier with workload-level train/test split
├── run_experiments.py     → Runs Phase 1 sweep and Phase 2 adaptive vs fixed comparison
├── test_scheduler.py      → Comprehensive test suite with independent reference implementation
├── main.py                → CLI simulation driver
├── app.py                 → Interactive Streamlit web dashboard
├── requirements.txt       → Pinned dependency specifications
├── adaptive_quantum_model.pkl → Exported trained machine learning model
├── model_metrics.json     → Evaluation metrics (held-out accuracy, MAE, mean regret, conflict rate)
├── training_data.csv      → Dispatch-state training dataset
├── results/               → Experimental CSV outputs and visualization charts
│   ├── phase1_fixed_quantum_sweep.csv
│   ├── phase2_adaptive_vs_fixed.csv
│   ├── phase1_response_vs_quantum.png
│   ├── phase1_context_switches_vs_quantum.png
│   ├── phase1_waiting_turnaround_vs_quantum.png
│   ├── phase1_throughput_fairness_vs_quantum.png
│   ├── phase2_adaptive_vs_fixed_bars.png
│   └── summary.txt
└── backup/                → Safe backups of original models and datasets

3. HOW TO RUN
-------------
A. Run Verification Test Suite (all 9 test suites with exact asserts):
    python test_scheduler.py

B. Run CLI Demonstration:
    python main.py

C. Generate Training Dataset (with candidate collapsing & tie-breaking):
    python create_training_data.py

D. Train Random Forest Model & Evaluate Hold-Out Performance:
    python train_model.py

E. Run Full Experiments (Phase 1 sweep & Phase 2 comparison):
    python run_experiments.py

F. Run Interactive Streamlit Web Dashboard:
    streamlit run app.py --server.port 8503

4. METRIC DEFINITIONS & CONTEXT SWITCH COUNTING RULE
----------------------------------------------------
All metrics are computed in metrics.py:
- Turnaround Time: Completion Time - Arrival Time (total time spent in the system).
- Waiting Time   : Turnaround Time - Burst Time (total time spent waiting in ready queue).
- Response Time  : First Execution Start Time - Arrival Time (latency to first CPU dispatch).
- Context Switches: Count of process handoffs to a DIFFERENT process.
  * Exact Counting Rule:
    1. A context switch is counted whenever the CPU finishes executing one process 
       and immediately or subsequently executes a different process (prev_pid != next_pid).
    2. Switches are counted identically regardless of whether penalty > 0 or penalty == 0.
    3. Idle transitions do NOT count as context switches (transitions to or from idle).
    4. If a process exhausts its quantum but is the only process in the ready queue, 
       it continues executing without a context switch.
- Context Switch Overhead: Total simulation time spent on context switching (Switches * Penalty).
- Throughput     : Completed Processes / Makespan (unit: processes per simulation time unit).
                   Makespan = Last Process Completion Time - First Process Arrival Time.
- CPU Utilization: Total Executed CPU Time / Total Elapsed Simulation Time (%).
- Jain's Fairness Index:
    Computed over per-process slowdown s_i = (Turnaround_i / Burst_i):
        J = (sum_{i=1}^n s_i)^2 / (n * sum_{i=1}^n s_i^2)
    Range: (0, 1]. A score of 1.0 represents perfectly equal slowdown across all processes.
    Also reports Min and Max slowdown.

5. HOW LABELS ARE GENERATED
---------------------------
In create_training_data.py, labels are generated at each dispatch state:
1. Candidate Quantum Pool: {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20}.
2. Candidate Collapsing: Any candidate quantum Q >= max(remaining_burst in ready_queue) 
   produces identical execution slices. They are collapsed into a single representative 
   candidate to eliminate evaluation redundancy.
3. Non-Clairvoyant Lookahead Horizon:
   Simulation evaluates the currently observable ready queue until all present tasks finish.
   Future unarrived processes are NOT assumed (pure non-clairvoyant state).
4. Composite Cost Function:
   cost = 1.0 * avg_response + 1.0 * avg_waiting + 2.0 * context_switch_overhead
   where context_switch_overhead = context_switches * penalty.
5. Tie-Breaking Rule:
   Candidates within 1% of the true minimum cost are considered tied. Ties are broken 
   by selecting the LARGEST tied quantum, which directly minimizes context-switch overhead.
6. Workload-Level Train/Test Split:
   Dataset is split 80% train / 20% test strictly by workload_id, preventing data leakage 
   between dispatches of the same workload.

6. HOW THE ADAPTIVE POLICY WORKS
--------------------------------
At each dispatch event, the scheduler captures 3 live state features:
1. ready_queue_length: Total processes currently ready (including the one about to run).
2. average_cpu_burst : Mean remaining burst of all ready processes.
3. recent_workload_behavior: Fraction of the last N=10 processes with burst <= 5 (I/O indicator).

The feature vector is passed to a Random Forest Classifier (100 estimators, max_depth=8).
The model predicts the optimal quantum Q in [1, 20]. The quantum is applied to that specific 
dispatch and re-evaluated at the next dispatch event.

7. MODEL EVALUATION & HOLD-OUT RESULTS
--------------------------------------
Evaluated on unseen test workloads (split strictly by workload):
- Held-Out Classification Accuracy: 69.77%
- Held-Out Mean Absolute Error (MAE): 2.09 time units
- Held-Out Mean Regret: 0.38 cost units (regret vs label reference)
- Label Conflict Rate: 4.00% across unique feature states
- Feature Importances:
  * average_cpu_burst         : ~55.0% (dominant predictor)
  * ready_queue_length        : ~23.3%
  * recent_workload_behavior  : ~21.8%

8. EMPIRICAL BENCHMARK FINDINGS & HONEST TRADE-OFF ANALYSIS
-----------------------------------------------------------
A. Where AI-Adaptive WINS:
   1. Context-Switch Overhead Reduction: On CPU-bound workloads under penalty = 1, AI-Adaptive 
      cuts switches from >1000 (under Q=1) to 86, matching Fixed Q=20. On Random Medium, it achieves 
      20 switches (matching the best fixed policy).
   2. Waiting Time on Long Tasks: On CPU-bound workloads, AI-Adaptive achieves 411.23 waiting time, 
      beating Fixed Medium Q=8 (417.80).
   3. Fairness and Generalization: Maintains high Jain's fairness index across diverse dynamic 
      workloads without requiring manual parameter tuning.

B. Where AI-Adaptive LOSES or Matches:
   1. Initial Response Time: Fixed Small (Q=2) achieves significantly faster initial response 
      (e.g., 4.80 vs 17.60 on Preset, 8.25 vs 16.52 on Mixed) because tiny slices guarantee 
      immediate CPU access for newly arrived tasks. Because the AI favors larger quanta (10..20) 
      to eliminate switch overhead, response time is higher.
   2. Zero-Penalty Environments: When context switches are free (penalty = 0), fixed Q=1 is 
      theoretically optimal; enlarging the quantum degrades response latency without any overhead benefit.
   3. Uniform Static Workloads: An engineer who knows in advance that a workload is 100% CPU-bound 
      can achieve marginally better results with a static large quantum or FCFS.

C. Conclusion:
   Adaptive Round Robin is not a universal replacement for specialized fixed quanta on known static 
   workloads. Its true practical value lies in general-purpose operating systems where workloads are 
   heterogeneous, unpredictable, and bursty, preventing preemption churn without manual tuning.