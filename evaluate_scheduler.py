import joblib
import pandas as pd
import numpy as np
from collections import deque

# 1. The Process Control Block
class Process:
    def __init__(self, pid, arrival_time, burst_time):
        self.pid = pid
        self.arrival_time = arrival_time
        self.total_burst_time = burst_time
        self.remaining_burst_time = burst_time
        
        self.start_time = -1
        self.completion_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0
        self.response_time = -1

# 2. The AI-Enhanced Simulator
class AdaptiveRoundRobinSimulator:
    def __init__(self, initial_quantum, context_switch_penalty=1, model_path=None):
        self.time_quantum = initial_quantum
        self.context_switch_penalty = context_switch_penalty
        self.current_time = 0
        self.ready_queue = deque()
        self.completed_processes = []
        self.total_context_switches = 0
        
        # Load the ML model if a path is provided
        self.model = None
        if model_path:
            resolved_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path) if not os.path.isabs(model_path) else model_path
            self.model = joblib.load(resolved_path)
            print("Successfully loaded AI Model into the CPU Simulator.")

    def get_queue_features(self):
        """Extracts real-time metrics from the queue to feed the AI."""
        queue_length = len(self.ready_queue)
        if queue_length == 0:
            return 0, 0.0, 0.0
        
        # Calculate Average Burst Time
        total_burst = sum(p.remaining_burst_time for p in self.ready_queue)
        avg_burst = total_burst / queue_length
        
        # Calculate I/O Ratio (Processes with short total bursts are treated as I/O-bound)
        io_count = sum(1 for p in self.ready_queue if p.total_burst_time < 10)
        io_ratio = io_count / queue_length
        
        return queue_length, avg_burst, io_ratio

    def run(self, process_list):
        unarrived_processes = sorted(process_list, key=lambda p: p.arrival_time)
        
        while unarrived_processes or self.ready_queue:
            # Check for arrivals
            while unarrived_processes and unarrived_processes[0].arrival_time <= self.current_time:
                self.ready_queue.append(unarrived_processes.pop(0))
                
            if not self.ready_queue:
                self.current_time = unarrived_processes[0].arrival_time
                continue
                
            # ==========================================
            # AI INTEGRATION POINT: Update Quantum
            # ==========================================
            if self.model and len(self.ready_queue) > 0:
                q_len, avg_burst, io_ratio = self.get_queue_features()
                
                # Format features for the model
                features = pd.DataFrame([[q_len, avg_burst, io_ratio]], 
                                        columns=['ready_queue_length', 'average_cpu_burst', 'io_ratio'])
                
                # Predict and update the time quantum dynamically
                self.time_quantum = self.model.predict(features)[0]
            # ==========================================

            current_process = self.ready_queue.popleft()
            
            # Record response time
            if current_process.start_time == -1:
                current_process.start_time = self.current_time
                current_process.response_time = self.current_time - current_process.arrival_time
                
            # Execute
            execution_time = min(current_process.remaining_burst_time, self.time_quantum)
            self.current_time += execution_time
            current_process.remaining_burst_time -= execution_time
            
            # Check for mid-execution arrivals
            while unarrived_processes and unarrived_processes[0].arrival_time <= self.current_time:
                self.ready_queue.append(unarrived_processes.pop(0))
                
            # Evaluate completion
            if current_process.remaining_burst_time == 0:
                current_process.completion_time = self.current_time
                current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                current_process.waiting_time = current_process.turnaround_time - current_process.total_burst_time
                self.completed_processes.append(current_process)
            else:
                self.ready_queue.append(current_process)
                
            # Apply Context Switch Penalty
            if self.ready_queue:
                self.current_time += self.context_switch_penalty
                self.total_context_switches += 1
                
                # Check for mid-switch arrivals
                while unarrived_processes and unarrived_processes[0].arrival_time <= self.current_time:
                    self.ready_queue.append(unarrived_processes.pop(0))

        return self.completed_processes, self.total_context_switches

# 3. Workload Generation and Benchmarking
def generate_workload(num_processes=100):
    """Generates a mixed workload of CPU-bound and I/O-bound processes."""
    np.random.seed(42) # Ensures identical workloads for fair testing
    workload = []
    for pid in range(1, num_processes + 1):
        arrival_time = np.random.randint(0, 500)
        # 50% chance of being CPU-bound (heavy), 50% chance of being I/O-bound (light)
        burst_time = np.random.randint(20, 100) if np.random.rand() > 0.5 else np.random.randint(1, 10)
        workload.append(Process(pid, arrival_time, burst_time))
    return workload

def calculate_metrics(processes):
    avg_waiting = np.mean([p.waiting_time for p in processes])
    avg_turnaround = np.mean([p.turnaround_time for p in processes])
    avg_response = np.mean([p.response_time for p in processes])
    return avg_waiting, avg_turnaround, avg_response

if __name__ == "__main__":
    print("Generating synthetic Mixed Workload (100 processes)...")
    
    # We must generate TWO identical copies of the workload so it is a fair test
    workload_baseline = generate_workload(100)
    workload_ai = generate_workload(100)

    # --- TEST 1: The Baseline (Fixed Time Quantum = 5) ---
    print("\n--- Running Baseline Round Robin (Fixed Q=5) ---")
    baseline_sim = AdaptiveRoundRobinSimulator(initial_quantum=5, context_switch_penalty=1)
    baseline_procs, baseline_switches = baseline_sim.run(workload_baseline)
    b_wait, b_turnaround, b_response = calculate_metrics(baseline_procs)

    # --- TEST 2: The AI-Enhanced Scheduler ---
    print("\n--- Running AI-Adaptive Round Robin ---")
    ai_sim = AdaptiveRoundRobinSimulator(initial_quantum=5, context_switch_penalty=1, model_path="adaptive_quantum_model.pkl")
    ai_procs, ai_switches = ai_sim.run(workload_ai)
    ai_wait, ai_turnaround, ai_response = calculate_metrics(ai_procs)

    # --- PRINT THE VERDICT ---
    print("\n" + "="*50)
    print("              FINAL RESULTS")
    print("="*50)
    print(f"{'Metric':<25} | {'Baseline (Fixed Q=5)':<22} | {'AI-Adaptive'}")
    print("-" * 70)
    print(f"{'Context Switches':<25} | {baseline_switches:<22} | {ai_switches}")
    print(f"{'Avg Waiting Time (ms)':<25} | {b_wait:<22.2f} | {ai_wait:.2f}")
    print(f"{'Avg Turnaround Time (ms)':<25} | {b_turnaround:<22.2f} | {ai_turnaround:.2f}")
    print(f"{'Avg Response Time (ms)':<25} | {b_response:<22.2f} | {ai_response:.2f}")
    print("="*50)