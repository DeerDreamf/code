import numpy as np
import random,math,csv,os
import matplotlib.pyplot as plt

if os.path.exists("results-JSSP"):
    pass
else:
    os.mkdir("results-JSSP")
results = open("results-JSSP" + os.sep + "results-JSSP-LA01(Levy)-convergence-3000.csv", "a+")
results_writer = csv.writer(results, lineterminator="\n")

# 工件数据：加工时间和分配机器
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


job_data = [
    [(machine_assignments[job][i] - 1, processing_times[job][i]) for i in range(num_machines)]
    for job in range(num_jobs)
]

tabu_tenure = 5
max_iterations = 3000
tabu_list = []
aspiration_criteria = 666

def init_solution():
    solution = []
    for job_idx in range(num_jobs):
        solution += [job_idx] * num_machines
    random.shuffle(solution)
    return solution


def decode_schedule(solution):
    job_count = [0] * num_jobs
    machine_time = [0] * num_machines
    job_end_time = [0] * num_jobs
    schedule = []

    for job in solution:
        operation_idx = job_count[job]
        machine, time = job_data[job][operation_idx]
        start_time = max(machine_time[machine], job_end_time[job])
        finish_time = start_time + time
        schedule.append(finish_time)
        machine_time[machine] = finish_time
        job_end_time[job] = finish_time
        job_count[job] += 1

    return schedule


def calculate_makespan(schedule):
    return max(schedule)


def is_tabu(solution):
    return solution in tabu_list


def update_tabu_list(solution):
    tabu_list.append(solution)
    if len(tabu_list) > tabu_tenure:
        tabu_list.pop(0)


def levy_flight(Lambda=1.5, max_step=5):
    sigma1 = np.power((math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)) /
                      (math.gamma((1 + Lambda) / 2) * Lambda * np.power(2, (Lambda - 1) / 2)), 1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, 1)
    v = np.random.normal(0, sigma2, 1)
    step = u / np.power(np.fabs(v), 1 / Lambda)
    step = int(abs(step[0]))
    return min(step, max_step)


def generate_neighbors(solution):
    neighbors = []
    length = len(solution)

    for i in range(length):
        for j in range(i + 1, length):
            neighbor = solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)


    for i in range(length):
        for j in range(length):
            if i != j:
                neighbor = solution.copy()
                job = neighbor.pop(i)
                neighbor.insert(j, job)
                neighbors.append(neighbor)


    for _ in range(length):
        i = random.randint(0, length - 1)
        j = (i + levy_flight()) % length
        neighbor = solution.copy()
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbors.append(neighbor)

    return neighbors

def tabu_search():
    best_solution = init_solution()
    best_makespan = calculate_makespan(decode_schedule(best_solution))
    current_solution = best_solution
    fitness_progress = []

    for iteration in range(max_iterations):
        neighbors = generate_neighbors(current_solution)
        best_neighbor = None
        best_neighbor_makespan = float("inf")

        for neighbor in neighbors:
            if not is_tabu(neighbor):
                schedule = decode_schedule(neighbor)
                makespan = calculate_makespan(schedule)
                if makespan < best_neighbor_makespan or (makespan < aspiration_criteria):
                    best_neighbor = neighbor
                    best_neighbor_makespan = makespan

        if best_neighbor and best_neighbor_makespan < best_makespan:
            best_solution = best_neighbor
            best_makespan = best_neighbor_makespan

        current_solution = best_neighbor
        update_tabu_list(current_solution)
        fitness_progress.append(best_makespan)
    results_writer.writerow(fitness_progress)

    return best_solution, best_makespan, fitness_progress


# 运行禁忌搜索
for z in range(30):
    best_solution, best_makespan, fitness_progress = tabu_search()

    print("best_solution：", best_solution)
    print("Makespan：", best_makespan)

