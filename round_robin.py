from collections import deque
from process import Process
from features import extract_features, features_to_dataframe, verify_feature_names


class FixedQuantumPolicy:
    """Fixed quantum policy: always returns a predetermined time quantum."""

    def __init__(self, quantum):
        val = int(quantum)
        if val < 1:
            raise ValueError(f"Time quantum must be >= 1 (got {quantum}). Quantum 0 is invalid.")
        self.quantum = val

    def __call__(self, state):
        return self.quantum


class AdaptiveQuantumPolicy:
    """Adaptive quantum policy: predicts time quantum per dispatch from workload state."""

    def __init__(self, model):
        self.model = model
        verify_feature_names(model)

    def __call__(self, state):
        feats = extract_features(state["ready_queue"], state["recent_history"])
        df = features_to_dataframe(feats)
        pred = self.model.predict(df)[0]
        return int(max(1, min(20, round(float(pred)))))


class RoundRobinSimulator:
    def __init__(self, time_quantum=5, context_switch_penalty=1, quantum_policy=None):
        if quantum_policy is not None:
            self.quantum_policy = quantum_policy
        elif callable(time_quantum):
            self.quantum_policy = time_quantum
        else:
            val = int(time_quantum)
            if val < 1:
                raise ValueError(f"Time quantum must be >= 1 (got {time_quantum}). Quantum 0 is invalid.")
            self.quantum_policy = FixedQuantumPolicy(val)

        self._time_quantum = time_quantum if isinstance(time_quantum, int) else 5
        self.context_switch_penalty = int(context_switch_penalty)

        self.current_time = 0
        self.ready_queue = deque()
        self.context_switches = 0
        self.timeline = []

        # ML feature storage per dispatch
        self.feature_data = []

        # Recent workload history (window of last N=10 processes)
        self.recent_processes = []
        self.window_size = 10

    @property
    def time_quantum(self):
        if isinstance(self.quantum_policy, FixedQuantumPolicy):
            return self.quantum_policy.quantum
        return self._time_quantum

    @time_quantum.setter
    def time_quantum(self, value):
        val = int(value)
        if val < 1:
            raise ValueError(f"Time quantum must be >= 1 (got {value}). Quantum 0 is invalid.")
        self._time_quantum = val
        if isinstance(self.quantum_policy, FixedQuantumPolicy):
            self.quantum_policy = FixedQuantumPolicy(val)

    def run(self, processes):
        # Sort processes by arrival time, stable (ties keep original order)
        processes = sorted(processes, key=lambda p: p.arrival_time)

        self.current_time = 0
        self.ready_queue = deque()
        self.context_switches = 0
        self.timeline = []
        self.feature_data = []
        self.recent_processes = []

        next_process = 0
        completed = 0
        total_processes = len(processes)
        completed_set = set()   # PIDs that have finished

        while completed < total_processes:
            # Add all processes that have arrived up to current_time
            while (
                next_process < total_processes
                and processes[next_process].arrival_time <= self.current_time
            ):
                self.ready_queue.append(processes[next_process])
                next_process += 1

            # If no process is ready, move time forward to the next arrival
            if not self.ready_queue:
                if next_process < total_processes:
                    if self.current_time < processes[next_process].arrival_time:
                        # Build not_arrived list
                        not_arrived = [
                            (processes[i].pid, processes[i].arrival_time)
                            for i in range(next_process, total_processes)
                        ]
                        self.timeline.append({
                            "start": self.current_time,
                            "end": processes[next_process].arrival_time,
                            "label": "Idle",
                            "kind": "idle",
                            # F3 enriched fields
                            "ready_queue": [],          # queue at slice start (excl. running)
                            "arrivals_during": [],
                            "not_arrived": not_arrived,
                            "completed_pids": sorted(completed_set),
                        })
                        self.current_time = processes[next_process].arrival_time
                continue

            # ---- CAPTURE STATE AT DISPATCH TIME (before popleft) ----
            # ready_queue at slice START = waiting processes EXCLUDING the one about to run
            # The front of the deque is the process about to run.
            current_process_peek = self.ready_queue[0]
            queue_at_start_pids = [p.pid for p in list(self.ready_queue)[1:]]  # exclude front

            # Capture live dispatch state (ready_queue includes the process about to run)
            state = {
                "ready_queue": list(self.ready_queue),
                "recent_history": list(self.recent_processes),
                "current_time": self.current_time,
            }
            feats = extract_features(self.ready_queue, self.recent_processes)
            raw_quantum = self.quantum_policy(state)
            if raw_quantum is None or int(raw_quantum) < 1:
                raise ValueError(f"Time quantum must be >= 1 (got {raw_quantum}). Quantum 0 is invalid.")
            chosen_quantum = int(max(1, min(20, round(float(raw_quantum)))))

            dispatch_ready_snapshot = [
                (p.pid, p.arrival_time, p.remaining_burst, p.start_time, p.response_time)
                for p in self.ready_queue
            ]

            self.feature_data.append({
                "ready_queue_length": feats["ready_queue_length"],
                "average_cpu_burst": feats["average_cpu_burst"],
                "recent_workload_behavior": feats["recent_workload_behavior"],
                "ready_queue_pids": [p.pid for p in self.ready_queue],
                "ready_snapshot": dispatch_ready_snapshot,
                "predicted_quantum": chosen_quantum,
            })

            # Take the first process from the ready queue
            current_process = self.ready_queue.popleft()

            # Record remaining burst BEFORE execution
            remaining_before = current_process.remaining_burst

            # Record first start time and response time
            if current_process.start_time == -1:
                current_process.start_time = self.current_time
                current_process.response_time = (
                    self.current_time - current_process.arrival_time
                )

            # Execute the process for up to chosen_quantum
            execution_time = min(
                chosen_quantum,
                current_process.remaining_burst,
            )

            execution_start = self.current_time
            self.current_time += execution_time
            current_process.remaining_burst -= execution_time

            # Enqueue processes that arrived DURING this execution slice
            arrivals_during = []
            while (
                next_process < total_processes
                and processes[next_process].arrival_time <= self.current_time
            ):
                p = processes[next_process]
                arrivals_during.append((p.pid, p.arrival_time))
                self.ready_queue.append(p)
                next_process += 1

            # Not-yet-arrived (after slice end)
            not_arrived_now = [
                (processes[i].pid, processes[i].arrival_time)
                for i in range(next_process, total_processes)
            ]

            # Will this process finish in this slice?
            finishes_now = (current_process.remaining_burst == 0)

            self.timeline.append({
                "start": execution_start,
                "end": self.current_time,
                "label": current_process.pid,
                "kind": "execution",
                # F3 enriched fields
                "ready_queue": queue_at_start_pids,    # waiting at slice START, excl. runner
                "arrivals_during": arrivals_during,
                "not_arrived": not_arrived_now,
                "completed_pids": sorted(completed_set),
                "quantum": chosen_quantum,
                "remaining_before": remaining_before,
                "remaining_after": current_process.remaining_burst,
                "finishes": finishes_now,
                "ready_snapshot": dispatch_ready_snapshot,
            })

            # Update recent processes window
            self.recent_processes.append(current_process)
            if len(self.recent_processes) > self.window_size:
                self.recent_processes.pop(0)

            # Check whether the current process has finished
            if current_process.remaining_burst == 0:
                current_process.completion_time = self.current_time
                current_process.turnaround_time = (
                    current_process.completion_time
                    - current_process.arrival_time
                )
                current_process.waiting_time = (
                    current_process.turnaround_time
                    - current_process.burst_time
                )
                completed += 1
                completed_set.add(current_process.pid)
                next_process_to_run = (
                    self.ready_queue[0] if self.ready_queue else None
                )
            else:
                # Process is not finished; re-append it to the back of the queue.
                self.ready_queue.append(current_process)
                next_process_to_run = self.ready_queue[0]

            # Context switch only when the next process to run is a DIFFERENT process.
            if (
                self.ready_queue
                and next_process_to_run is not None
                and next_process_to_run is not current_process
            ):
                switch_start = self.current_time
                self.current_time += self.context_switch_penalty
                self.context_switches += 1

                # Enqueue processes that arrived during the context switch
                cs_arrivals = []
                while (
                    next_process < total_processes
                    and processes[next_process].arrival_time <= self.current_time
                ):
                    p = processes[next_process]
                    cs_arrivals.append((p.pid, p.arrival_time))
                    self.ready_queue.append(p)
                    next_process += 1

                # Not yet arrived after CS
                not_arrived_cs = [
                    (processes[i].pid, processes[i].arrival_time)
                    for i in range(next_process, total_processes)
                ]

                if self.context_switch_penalty > 0:
                    self.timeline.append({
                        "start": switch_start,
                        "end": self.current_time,
                        "label": "CS",
                        "kind": "switch",
                        # F3 enriched fields
                        "ready_queue": [p.pid for p in self.ready_queue],
                        "arrivals_during": cs_arrivals,
                        "not_arrived": not_arrived_cs,
                        "completed_pids": sorted(completed_set),
                    })

        return processes