import numpy as np
import random, math, csv, os
import matplotlib.pyplot as plt

if os.path.exists("results-JSSP"):
    pass
else:
    os.mkdir("results-JSSP")
results = open("results-JSSP" + os.sep + "results-JSSP-LA02bak-testconvergence-3000.csv", "a+")
results_writer = csv.writer(results, lineterminator="\n")

# process time and machines data
num_jobs = 10
num_machines = 5
processing_times = [
    [20, 31, 17, 87, 76],
    [24, 18, 32, 81, 25],
    [58, 72, 23, 99, 28],
    [45, 76, 86, 90, 97],
    [42, 46, 17, 48, 27],
    [98, 67, 62, 27, 48],
    [80, 12, 50, 19, 28],
    [94, 63, 98, 50, 80],
    [75, 41, 50, 55, 14],
    [61, 37, 18, 79, 72]
]
machine_assignments = [
    [1, 4, 2, 5, 3],
    [5, 3, 1, 2, 4],
    [2, 3, 5, 1, 4],
    [3, 2, 5, 1, 4],
    [5, 1, 4, 3, 2],
    [2, 1, 5, 4, 3],
    [5, 2, 4, 1, 3],
    [2, 1, 3, 4, 5],
    [5, 1, 3, 2, 4],
    [5, 3, 2, 4, 1]
]

# Adjust machine indexing to be 0-based
job_data = [
    [(machine_assignments[job][i] - 1, processing_times[job][i]) for i in range(num_machines)]
    for job in range(num_jobs)
]

tabu_tenure = 15
max_iterations = 3000
tabu_list = []



# Calculate total processing time for each job
def get_job_total_time(job_idx):
    return sum([processing_times[job_idx][i] for i in range(num_machines)])


# Improved initialization with insertion heuristic
def init_solution_improved():
    # Calculate total processing time for each job
    job_times = [(job_idx, get_job_total_time(job_idx)) for job_idx in range(num_jobs)]
    # Sort jobs by descending total processing time
    job_times.sort(key=lambda x: x[1], reverse=True)

    solution = []
    scheduled_jobs = set()

    # Schedule the longest job first
    longest_job = job_times[0][0]
    # Get operations sorted by descending processing time
    ops = [(i, job_data[longest_job][i][1]) for i in range(num_machines)]
    ops.sort(key=lambda x: x[1], reverse=True)

    for op_idx, _ in ops:
        solution.append(longest_job)

    scheduled_jobs.add(longest_job)

    # Schedule remaining jobs
    for job_idx, _ in job_times[1:]:
        # Get operations for this job sorted by descending processing time
        ops = [(i, job_data[job_idx][i][1]) for i in range(num_machines)]
        ops.sort(key=lambda x: x[1], reverse=True)

        for op_idx, proc_time in ops:
            machine, _ = job_data[job_idx][op_idx]

            # Try all feasible insertion positions
            best_position = len(solution)
            best_makespan = float('inf')

            for insert_pos in range(len(solution) + 1):
                # Create candidate solution
                candidate = solution.copy()
                candidate.insert(insert_pos, job_idx)

                # Check if insertion is feasible (maintains operation order for this job)
                if is_feasible_insertion(candidate, job_idx):
                    # Evaluate makespan
                    schedule = decode_schedule(candidate)
                    makespan = calculate_makespan(schedule)

                    if makespan < best_makespan:
                        best_makespan = makespan
                        best_position = insert_pos

            # Insert at best position
            solution.insert(best_position, job_idx)

        scheduled_jobs.add(job_idx)

    return solution


# Check if a solution maintains the correct operation order for each job
def is_feasible_insertion(solution, job_idx):
    job_count = [0] * num_jobs

    for job in solution:
        if job == job_idx:
            job_count[job] += 1
            # Check if operations appear in order (no more than num_machines operations)
            if job_count[job] > num_machines:
                return False

    return True


# Decode function
def decode_schedule(solution):
    job_operation_count = [0] * num_jobs
    machine_available_time = [0] * num_machines
    job_completion_time = [0] * num_jobs

    operation_schedule = []

    for job in solution:
        op_idx = job_operation_count[job]
        if op_idx >= num_machines:
            continue

        machine, processing_time = job_data[job][op_idx]

        start_time = max(machine_available_time[machine], job_completion_time[job])
        completion_time = start_time + processing_time

        machine_available_time[machine] = completion_time
        job_completion_time[job] = completion_time

        job_operation_count[job] += 1

        operation_schedule.append((job, op_idx, machine, start_time, completion_time))

    return [op[4] for op in operation_schedule]


def calculate_makespan(schedule):
    if not schedule:
        return float('inf')
    return max(schedule)


def is_tabu(solution):
    return solution in tabu_list


def update_tabu_list(solution):
    tabu_list.append(solution)
    if len(tabu_list) > tabu_tenure:
        tabu_list.pop(0)


# Levy flight
def levy_flight(Lambda=1.5, max_step=5):
    sigma1 = np.power((math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)) /
                      (math.gamma((1 + Lambda) / 2) * Lambda * np.power(2, (Lambda - 1) / 2)), 1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, 1)
    v = np.random.normal(0, sigma2, 1)
    step = u / np.power(np.fabs(v), 1 / Lambda)
    step = int(abs(step[0]))
    return min(step, max_step)


def levy_perturbation(solution):
    neighbor = solution.copy()
    length = len(solution)
    i = random.randint(0, length - 1)
    j = (i + levy_flight()) % length
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def decode_disjunctive_graph(solution):
    job_ops = [0] * num_jobs
    machine_time = [0] * num_machines
    job_time = [0] * num_jobs
    op_schedules = []

    machine_queues = [[] for _ in range(num_machines)]

    valid_operations = []
    job_counts = [0] * num_jobs

    for job in solution:
        if job_counts[job] < num_machines:
            valid_operations.append(job)
            job_counts[job] += 1

    for job in valid_operations:
        op_idx = job_ops[job]
        machine, proc_time = job_data[job][op_idx]

        start = max(machine_time[machine], job_time[job])
        finish = start + proc_time

        machine_time[machine] = finish
        job_time[job] = finish
        job_ops[job] += 1

        machine_queues[machine].append((job, op_idx))
        op_schedules.append((job, op_idx, machine, start, finish))

    return op_schedules, machine_queues


def get_critical_path(solution):
    op_schedules, machine_queues = decode_disjunctive_graph(solution)

    graph = {}
    start_time = {}
    finish_time = {}

    for job, op_idx, machine, st, ft in op_schedules:
        node = (job, op_idx)
        graph.setdefault(node, [])
        start_time[node] = st
        finish_time[node] = ft

    for job in range(num_jobs):
        for op in range(num_machines - 1):
            u = (job, op)
            v = (job, op + 1)
            if u in graph and v in graph:
                graph[u].append(v)

    for machine in range(num_machines):
        queue = sorted(machine_queues[machine], key=lambda x: start_time.get((x[0], x[1]), 0))
        for i in range(len(queue) - 1):
            u = queue[i]
            v = queue[i + 1]
            if u in graph and v in graph:
                graph[u].append(v)

    longest_path = {}
    pred = {}

    valid_nodes = [node for node in graph if node in finish_time]
    sorted_nodes = sorted(valid_nodes, key=lambda x: finish_time.get(x, 0))

    for node in sorted_nodes:
        longest_path[node] = finish_time[node] - start_time[node]
        for prev in graph:
            if node in graph[prev] and prev in longest_path:
                alt = longest_path[prev] + (finish_time[node] - start_time[node])
                if alt > longest_path[node]:
                    longest_path[node] = alt
                    pred[node] = prev

    if not longest_path:
        return []

    end_node = max(longest_path.items(), key=lambda x: finish_time.get(x[0], 0) + x[1])[0]

    path = []
    current = end_node
    while current in pred:
        path.insert(0, current)
        current = pred[current]
    path.insert(0, current)

    return path


def generate_n5_neighbors(solution):
    neighbors = []
    critical_path = get_critical_path(solution)

    def op_position(job, op_idx):
        count = 0
        for i, val in enumerate(solution):
            if val == job:
                if count == op_idx:
                    return i
                count += 1
        return -1

    for i in range(len(critical_path) - 1):
        job1, op1 = critical_path[i]
        job2, op2 = critical_path[i + 1]

        if job1 != job2:
            idx1 = op_position(job1, op1)
            idx2 = op_position(job2, op2)
            if idx1 != -1 and idx2 != -1 and idx1 != idx2:
                neighbor = solution.copy()
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                neighbors.append(neighbor)

    if not neighbors:
        for _ in range(5):
            neighbor = solution.copy()
            i = random.randint(0, len(solution) - 1)
            j = random.randint(0, len(solution) - 1)
            if i != j:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)

    return neighbors


def tabu_search(use_improved_init=True):
    # Use improved initialization
    if use_improved_init:
        best_solution = init_solution_improved()

    current_solution = best_solution.copy()

    schedule = decode_schedule(best_solution)
    best_makespan = calculate_makespan(schedule)

    fitness_progress = []

    no_improve_count = 0
    max_no_improve = 100

    for iteration in range(max_iterations):
        neighbors = generate_n5_neighbors(current_solution)
        if not neighbors:
            current_solution = levy_perturbation(current_solution)
            continue

        best_neighbor = None
        best_neighbor_makespan = float("inf")

        for neighbor in neighbors:
            schedule = decode_schedule(neighbor)
            makespan = calculate_makespan(schedule)

            if not is_tabu(neighbor) or makespan < best_makespan:
                if makespan < best_neighbor_makespan:
                    best_neighbor = neighbor
                    best_neighbor_makespan = makespan

        if best_neighbor is None:
            current_solution = levy_perturbation(current_solution)
            continue

        current_solution = best_neighbor
        update_tabu_list(current_solution)

        if best_neighbor_makespan < best_makespan:
            best_solution = best_neighbor
            best_makespan = best_neighbor_makespan
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= max_no_improve:
            current_solution = levy_perturbation(current_solution)
            no_improve_count = 0

        fitness_progress.append(best_makespan)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Current Best: {best_makespan}")

    return best_solution, best_makespan, fitness_progress


# Run the main project

for z in range(30):

    best_solution, best_makespan, fitness_progress = tabu_search(use_improved_init=True)

    print("best_solution:", best_solution)
    print("Makespan:", best_makespan)

    # Write results
    results_writer.writerow([z + 1, best_makespan])

results.close()
print("\nAll runs completed!")