import csv
import pandas as pd
from process import Process
from round_robin import RoundRobinSimulator, AdaptiveQuantumPolicy, FixedQuantumPolicy
from train_model import model
from metrics import compute_metrics, format_process_table

# Create test processes
processes = [
    Process("P1", 0, 8),
    Process("P2", 1, 14),
    Process("P3", 2, 2),
    Process("P4", 4, 19),
    Process("P5", 8, 6),
]

print("Running AI-Adaptive Round Robin Simulation...")
adaptive_policy = AdaptiveQuantumPolicy(model)
simulator = RoundRobinSimulator(quantum_policy=adaptive_policy, context_switch_penalty=1)

results = simulator.run([Process(p.pid, p.arrival_time, p.burst_time) for p in processes])
metrics = compute_metrics(results, simulator.timeline, context_switch_penalty=1)
df, warnings = format_process_table(results, simulator.timeline)

print("\n" + "=" * 90)
print("                       ROUND ROBIN SCHEDULING RESULTS (AI-ADAPTIVE)")
print("=" * 90)
print(df.to_string(index=False))
print("-" * 90)
print(f"Context Switches        : {metrics['context_switches']}")
print(f"Context Switch Overhead : {metrics['context_switch_overhead']} units")
print(f"Average Waiting Time    : {metrics['avg_waiting']:.2f} units")
print(f"Average Turnaround Time : {metrics['avg_turnaround']:.2f} units")
print(f"Average Response Time   : {metrics['avg_response']:.2f} units")
print(f"Throughput              : {metrics['throughput']:.4f} processes/unit")
print(f"CPU Utilization         : {metrics['cpu_utilization']:.2%}")
print(f"Jain's Fairness Index   : {metrics['fairness']:.4f}")
print("=" * 90)

# Save feature data
if simulator.feature_data:
    with open("feature_data.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ready_queue_length",
                "average_cpu_burst",
                "recent_workload_behavior",
            ],
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(simulator.feature_data)
    print("feature_data.csv updated successfully!")