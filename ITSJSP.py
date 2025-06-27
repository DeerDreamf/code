import numpy as np
import random, math, csv, os
import matplotlib.pyplot as plt

if os.path.exists("results-JSSP"):
    pass
else:
    os.mkdir("results-JSSP")
results = open("results-JSSP" + os.sep + "results-JSSP-LA01(Levy)-convergence-3000.csv", "a+")
results_writer = csv.writer(results, lineterminator="\n")

# process time and machines data
num_jobs = 10
num_machines = 5
processing_times = [
    [53, 21, 34, 55, 95],
    [21, 71, 26, 52, 16],
    [12, 42, 31, 39, 98],
    [55, 77, 66, 77, 79],
    [83, 19, 64, 34, 37],
    [92, 54, 43, 62, 79],
    [93, 87, 87, 69, 77],
    [60, 41, 38, 24, 83],
    [44, 49, 98, 17, 25],
    [96, 75, 43, 79, 77]
]
machine_assignments = [
    [2, 1, 5, 4, 3],
    [1, 4, 5, 3, 2],
    [4, 5, 2, 3, 1],
    [2, 1, 5, 3, 4],
    [1, 4, 3, 2, 5],
    [2, 3, 5, 1, 4],
    [4, 5, 2, 3, 1],
    [3, 1, 2, 4, 5],
    [4, 2, 5, 1, 3],
    [5, 4, 3, 2, 1]
]

# Adjust machine indexing to be 0-based (machine numbering starts at 0 instead of 1)--
job_data = [
    [(machine_assignments[job][i] - 1, processing_times[job][i]) for i in range(num_machines)]
    for job in range(num_jobs)
]

tabu_tenure = 50
max_iterations = 3000
tabu_list = []
aspiration_criteria = 666


#  Initilization
def init_solution():
    solution = []
    for job_idx in range(num_jobs):
        solution += [job_idx] * num_machines
    random.shuffle(solution)
    return solution


#decode function
def decode_schedule(solution):
    #  Track the current operation index for each job
    job_operation_count = [0] * num_jobs
    # Track the available time for each machine
    machine_available_time = [0] * num_machines
    # Track the completion time for each job
    job_completion_time = [0] * num_jobs

    operation_schedule = []  #  Store completion times for all operations

    for job in solution:
        #  Get the next operation for the current job
        op_idx = job_operation_count[job]
        if op_idx >= num_machines:
            continue  # All operations for this job have been scheduled


        #  Get the machine and processing time for the operation
        machine, processing_time = job_data[job][op_idx]

        # Calculate the start time and completion time for the operation
        start_time = max(machine_available_time[machine], job_completion_time[job])
        completion_time = start_time + processing_time

        # Update the machine available time and job completion time
        machine_available_time[machine] = completion_time
        job_completion_time[job] = completion_time

        # Increment operation count
        job_operation_count[job] += 1

        # Store operation information
        operation_schedule.append((job, op_idx, machine, start_time, completion_time))

    #Return the completion times for all operations
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


# returns more detailed scheduling information
def decode_disjunctive_graph(solution):
    job_ops = [0] * num_jobs
    machine_time = [0] * num_machines
    job_time = [0] * num_jobs
    op_schedules = []  # [(job, op_idx, machine, start, finish)]

    machine_queues = [[] for _ in range(num_machines)]

    # Create a valid operation list (avoid duplicate operations)
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


# Find operations (job, op_idx) on the critical path
def get_critical_path(solution):
    op_schedules, machine_queues = decode_disjunctive_graph(solution)

    # Build directed graph G: each operation is a node, edges are dependencies
    graph = {}
    start_time = {}
    finish_time = {}

    for job, op_idx, machine, st, ft in op_schedules:
        node = (job, op_idx)
        graph.setdefault(node, [])
        start_time[node] = st
        finish_time[node] = ft

    # Add sequence edges (within the same job)
    for job in range(num_jobs):
        for op in range(num_machines - 1):
            u = (job, op)
            v = (job, op + 1)
            if u in graph and v in graph:
                graph[u].append(v)

    #  Add non-overlapping edges on machines
    for machine in range(num_machines):
        queue = sorted(machine_queues[machine], key=lambda x: start_time.get((x[0], x[1]), 0))
        for i in range(len(queue) - 1):
            u = queue[i]
            v = queue[i + 1]
            if u in graph and v in graph:
                graph[u].append(v)

    #  Backtrack the longest path (dynamic programming)
    longest_path = {}
    pred = {}

    #  Ensure all nodes have valid completion times
    valid_nodes = [node for node in graph if node in finish_time]
    sorted_nodes = sorted(valid_nodes, key=lambda x: finish_time.get(x, 0))

    for node in sorted_nodes:
        longest_path[node] = finish_time[node] - start_time[node]  # Initialize as the processing time for operating itself
        for prev in graph:
            if node in graph[prev] and prev in longest_path:
                alt = longest_path[prev] + (finish_time[node] - start_time[node])
                if alt > longest_path[node]:
                    longest_path[node] = alt
                    pred[node] = prev

    #  Find the node with the latest finish time
    if not longest_path:
        return []

    end_node = max(longest_path.items(), key=lambda x: finish_time.get(x[0], 0) + x[1])[0]

    # Backtrack to find critical path nodes
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

    #  Map (job, op_idx) back to index positions in the solution
    def op_position(job, op_idx):
        count = 0
        for i, val in enumerate(solution):
            if val == job:
                if count == op_idx:
                    return i
                count += 1
        return -1

    # Generate swaps for adjacent operations on the critical path
    for i in range(len(critical_path) - 1):
        job1, op1 = critical_path[i]
        job2, op2 = critical_path[i + 1]

        # Only swap when the two operations are in different jobs
        if job1 != job2:
            idx1 = op_position(job1, op1)
            idx2 = op_position(job2, op2)
            if idx1 != -1 and idx2 != -1 and idx1 != idx2:
                neighbor = solution.copy()
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                neighbors.append(neighbor)

    # If no neighbors are found, generate some random neighbors
    if not neighbors:
        for _ in range(5):
            neighbor = solution.copy()
            i = random.randint(0, len(solution) - 1)
            j = random.randint(0, len(solution) - 1)
            if i != j:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)

    return neighbors


def tabu_search():
    best_solution = init_solution()
    current_solution = best_solution.copy()

    schedule = decode_schedule(best_solution)
    best_makespan = calculate_makespan(schedule)

    fitness_progress = []

    no_improve_count = 0
    max_no_improve = 50  # The upper limit of the number of times the fitness value has not been improved continuously

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

            # If the neighbor solution is not taboo, or meets the amnesty criteria
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

        # Using the  Levy Flight  to jump the local
        if no_improve_count >= max_no_improve:
            current_solution = levy_perturbation(current_solution)
            no_improve_count = 0

        fitness_progress.append(best_makespan)

        # Print current progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Current Best: {best_makespan}")


    return best_solution, best_makespan, fitness_progress


# run the main project
for z in range(30):
    print(f"Run {z + 1}/30")
    best_solution, best_makespan, fitness_progress = tabu_search()

    print("best_solution：", best_solution)
    print("Makespan：", best_makespan)



results.close()