# AI-Adaptive Round Robin CPU Scheduler (Project 19)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

An interactive simulation and research evaluation platform investigating whether dynamically adapting the Round Robin CPU time quantum via machine learning improves system responsiveness, reduces context-switch overhead, and balances turnaround time and fairness.

---

## Overview

Traditional Round Robin (RR) schedulers rely on a fixed, static time quantum:
- **Small Quanta (e.g., Q=1 or Q=2)**: Minimize initial response time and maximize interactivity, but cause excessive context switching and high CPU overhead under context penalties.
- **Large Quanta (e.g., Q=15 or Q=20)**: Minimize context-switch overhead for CPU-heavy jobs, but degenerate towards First-Come-First-Served (FCFS), drastically increasing waiting and response times for shorter or interactive tasks.

**AI-Adaptive Round Robin** uses a trained Random Forest model to evaluate live ready-queue state and recent execution history at each dispatch, dynamically selecting an optimal quantum ($Q \in \{1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20\}$) without clairvoyant knowledge of future process arrivals.

---

## Key Features

1. **Phase 1 Baseline & Phase 2 Adaptive Policies**:
   - Compare fixed quantum configurations ($Q \in [1, 20]$) against dynamic machine-learning prediction across CPU-bound, I/O-bound, and mixed workloads.
2. **Interactive Streamlit Dashboard**:
   - **Tab 1: Interactive Gantt & Metrics**: Real-time event-by-event Gantt chart, step-by-step playback slider, per-process execution tables, queue-state dynamics, and grouped comparison charts.
   - **Tab 2: "Why This Quantum?" Decision Explainer**: Transparent, non-clairvoyant lookahead cost evaluation showing candidate quanta, costs, and regret metrics.
   - **Tab 3: Model Insights & Training Data**: Model feature importances, dataset class distributions, and live per-dispatch feature inspector.
   - **Tab 4: Phase 2 Empirical Evaluation**: Comprehensive benchmark across penalties $0, 1, 2, 3$ comparing AI-Adaptive against fixed policies on all 6 standard OS metrics.
3. **Independent Reference Verification**:
   - Built-in verification engine validating simulator calculations against an independent reference implementation.

---

## Project Structure

```text
├── app.py                      # Interactive Streamlit Web Application
├── process.py                  # Process & PCB data structures
├── round_robin.py              # Round Robin simulator (Fixed & Adaptive)
├── features.py                 # Feature extraction & validation
├── metrics.py                  # OS metrics & per-process table computation
├── create_training_data.py     # Non-clairvoyant training dataset generator
├── train_model.py              # Random Forest training with GroupKFold split
├── run_experiments.py          # Benchmark runner for Phase 1 & Phase 2
├── test_scheduler.py           # Comprehensive verification test suite
├── adaptive_quantum_model.pkl  # Trained ML model
├── training_data.csv           # Dispatch-state dataset
├── requirements.txt            # Pinned dependencies
├── README.txt                  # Full technical specification & notes
└── results/                    # Experimental outputs and benchmark data
```

---

## Quick Start

### 1. Installation
Clone the repository and install the pinned dependencies:

```bash
git clone https://github.com/aayushgupta-art/OS-P2.git
cd OS-P2
pip install -r requirements.txt
```

### 2. Run the Web Application
Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

### 3. Run Verification Tests
Run the 10-suite mathematical verification and regression test suite:

```bash
python test_scheduler.py
```
