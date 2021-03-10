from random import *
from collections import defaultdict
import math
import re
import itertools
from copy import deepcopy
import time

import numpy as np
import pandas as pd


def process(*, batch, n_people, max_batch_time, max_total_time, max_wo_count, running_time):
    data = pd.read_excel("ASDEP.DATA3.xlsm", sheet_name="Data")
    old_work_orders = []
    for i in range(0, len(data)):
        old_work_orders.append(list(data.iloc[i]))

    work_orders = old_work_orders[:600]
    for i in range(0, len(work_orders)):
        work_orders[i][0] = int(work_orders[i][0])
        work_orders[i][2] = int(work_orders[i][2])
        work_orders[i][3] = int(work_orders[i][3])
        work_orders[i][4] = int(work_orders[i][4])
        work_orders[i][5] = int(work_orders[i][5])
        work_orders[i][1] = str(work_orders[i][1])

    data = pd.read_excel("ASDEP.DATA3.xlsm", sheet_name="Zbarf")
    d = defaultdict(dict)
    # d = defaultdict(list)

    D_ambar = pd.read_excel("AMBAR PLAN.xlsx", header=None)
    liste1 = list(D_ambar.iloc[1])

    for no, addr in zip(data["Sipariþ No"], data["Depo Adres"]):
        if "wo_addresses" not in d[no]:
            d[no]["wo_addresses"] = []
        d[no]["wo_addresses"].append(addr)
        # d[no].append(addr)

    """ Previous version """
    [(val.update({"wo_addresses": []})) for val in d.values() if "wo_addresses" not in val]
    [(val.update({"commonality": {}})) for val in d.values() if "commonality" not in val]

    def are_neighbors(addr1, addr2, row_distance=5.0):
        """
        Check if two addresses are neighbors
        :param row_distance: Max distance between rows, inclusive
        :param addr1:
        :param addr2:
        :rtype: bool
        """
        no1 = [int(val) for val in re.findall('[0-9]+', addr1)]
        no2 = [int(val) for val in re.findall('[0-9]+', addr2)]
        if any([not addr1.startswith('R'), not addr2.startswith('R'), len(no2) != 3, len(no1) != 3]):
            return False
        if no1[0] / 2 != no2[0] / 2:
            return False
        min_row, max_row = min(no1[2] / 100, no2[2] / 100), max(no1[2] / 100, no2[2] / 100)
        if min_row + row_distance / 2 >= max_row + row_distance / 2:
            return True
        return False

    def work_order_proximity(w1, w2, row_distance=5.0):
        """
        Number of materials that are in close proximity in each other
        :param row_distance:
        :param w1:
        :param w2:
        :return:
        """
        score = 0.0
        for m1, m2 in itertools.product(d[w1], d[w2]):
            score += are_neighbors(m1, m2, row_distance=row_distance)
        return score

    rack_01 = []
    rack_23 = []
    rack_45 = []
    rack_67 = []
    rack_89 = []

    for i in range(len(D_ambar)):
        if i < 20:
            rack_01.append(list(D_ambar.iloc[i]))
        elif 20 <= i < 40:
            rack_23.append(list(D_ambar.iloc[i]))
        elif 40 <= i < 60:
            rack_45.append(list(D_ambar.iloc[i]))
        elif 60 <= i < 76:
            rack_67.append(list(D_ambar.iloc[i]))
        elif 76 <= i < 96:
            rack_89.append(list(D_ambar.iloc[i]))

    solution = []

    for i in range(0, batch):
        solution.append([])

    ind1 = 0

    for i in range(len(work_orders)):
        solution[ind1].append(work_orders[i][0])
        order_index = solution[ind1][-1]
        order = work_orders[order_index - 1][1]
        work_orders[i].append(",".join(d[order]))

        if len(solution[ind1]) >= 10 and ind1 != batch - 1:
            ind1 += 1

    work_order_time = []

    def get_time_for_addr(addr):
        """

        :param addr:
        :return:
        """
        time_K = time_R = 0.0
        if addr[0] == "R" and addr[1] != 'G':
            kolon = int(addr[3:5])
            raf = int(addr[6:8])
            distance = math.sqrt(kolon ** 2 + raf ** 2)
            time_R = ((distance * 100) / 49.4773)
        else:
            time_K = (42 * 2) / n_people
        return (max(time_R, time_K)) / 3600.0

    def gradeSolution(solution, work_orders, grade=0, total_time=0, row_distance=5.0):
        total_materials = 0
        for i in solution[:-1]:

            material = 0
            batch_addrs = set()
            if len(i) != 0:
                batch_t = []
                batch_time = 0
                for j in i:

                    time_of_wo = 0
                    # if work_orders[j-1][2] !=1:
                    #     grade += -1e12
                    grade += work_orders[j - 1][2] * 1e8
                    grade += work_orders[j - 1][3] * 2000
                    grade += work_orders[j - 1][4] * 1000
                    material += work_orders[j - 1][7]

                    addrs = set(work_orders[j - 1][15].split(","))
                    for addr in addrs:
                        batch_addrs.add(addr)
                    addrs_point = 0

                    addrs = set(work_orders[j - 1][15].split(","))
                    addrs_point += len(batch_addrs.intersection(addrs))
                    grade += addrs_point / len(batch_addrs) * 100000

                    if "wo_addresses" not in d[work_orders[j - 1][1]]:
                        d[work_orders[j - 1][1]]["wo_addresses"] = []
                    wo_addresses = d[work_orders[j - 1][1]]["wo_addresses"]
                    # wo_addresses = d[work_orders[j-1][1]]

                    for wos in wo_addresses:
                        time_of_wo += get_time_for_addr(wos)
                    batch_t.append(time_of_wo)

                batch_time = sum(batch_t)

                if batch_time > max_batch_time:
                    return -1001, None

            """ Proximity """
            for j, k in itertools.combinations(i, 2):
                if "commonality" not in d[work_orders[k - 1][1]]:
                    d[work_orders[k - 1][1]]["commonality"] = {}
                if "commonality" not in d[work_orders[j - 1][1]]:
                    d[work_orders[j - 1][1]]["commonality"] = {}
                mem_result = d[work_orders[k - 1][1]]["commonality"].get(
                    work_orders[j - 1][1], d[work_orders[j - 1][1]]["commonality"].get(work_orders[k - 1][1]))
                if mem_result is not None:
                    grade += mem_result
                    continue
                new_result = work_order_proximity(
                    work_orders[j - 1][1], work_orders[k - 1][1], row_distance=row_distance) * 1000
                grade += new_result
                d[work_orders[k - 1][1]]["commonality"][work_orders[j - 1][1]] \
                    = d[work_orders[j - 1][1]]["commonality"][work_orders[k - 1][1]] \
                    = new_result

            total_time += batch_time

            if total_time > max_total_time:
                return -1000, None

            if len(i) > max_wo_count:
                return -2000, None
            """   
            if material > 120:
                grade += -1000000000
    
            if material > 0:
                if material < 20:
                    grade += -1000000000
            total_materials += material
    
    
        if total_materials > 500:
            grade += -1000000000000
            """
        return grade, total_time, batch_time

    tabu_tenure = 100
    tabu_list = []

    for i in range(len(work_orders)):
        tabu_list.append(0)

    def TabuSearch(solution):
        nonlocal work_orders
        failed = 0

        grade = gradeSolution(solution, work_orders)[0]
        start_time = np.inf

        count = 1
        while True:
            random = randint(0, 1)
            times_to_swap = np.arange(np.random.binomial(len(work_orders), 0.05))
            original = deepcopy(solution)

            if random == 0:

                for _ in times_to_swap:
                    first = randint(0, len(solution) - 1)
                    third = randint(0, len(solution[first]) - 1)

                    while tabu_list[solution[first][third] - 1] >= 1:
                        first = randint(0, len(solution) - 1)
                        third = randint(0, len(solution[first]) - 1)

                    second = randint(0, len(solution) - 1)
                    fourth = randint(0, len(solution[second]) - 1)

                    while tabu_list[solution[second][fourth] - 1] >= 1:
                        second = randint(0, len(solution) - 1)
                        fourth = randint(0, len(solution[second]) - 1)

                    solution[first][third], solution[second][fourth] = solution[second][fourth], solution[first][third]

                current_grade = gradeSolution(solution, work_orders)

                if current_grade[0] > grade:
                    failed = 0
                    grade = current_grade[0]
                    tabu_list[solution[first][third] - 1] = tabu_tenure
                    tabu_list[solution[second][fourth] - 1] = tabu_tenure
                    start_time = time.time() if start_time == np.inf and grade > 0 else start_time

                else:
                    solution = original
                    failed += 1

            else:
                for _ in times_to_swap:

                    fifth = randint(0, len(solution) - 1)
                    sixth = randint(0, len(solution[fifth]) - 1)

                    while tabu_list[solution[fifth][sixth] - 1] >= 1 or len(solution[fifth]) == 1:
                        fifth = randint(0, len(solution) - 1)
                        sixth = randint(0, len(solution[fifth]) - 1)

                    seventh = randint(0, len(solution) - 1)

                    solution[seventh].append(solution[fifth][sixth])

                    del solution[fifth][sixth]

                current_grade = gradeSolution(solution, work_orders)

                if current_grade[0] > grade:
                    failed = 0
                    grade = current_grade[0]
                    tabu_list[solution[seventh][len(solution[seventh]) - 1] - 1] = tabu_tenure
                    start_time = time.time() if start_time == np.inf and grade > 0 else start_time

                else:
                    solution = original
                    failed += 1

            count += 1
            for i in tabu_list:
                if i != 0:
                    i += -1

            if count % 1000 == 0:
                print(f"Iteration {count} Grade: {grade}. Solution time: {time.time() - start_time}")

            if time.time() - start_time >= running_time * 60:
                return solution, grade

    grade = gradeSolution(solution, work_orders)
    # z= gradeSolution(yasinSol,work_orders)
    # print(z)
    print(f"Initial Grade: {grade}")
    solution, grade = TabuSearch(solution)
    print(f"Improved Grade: {grade}")
    print(solution[:-1])
    print(f"Batch sizes: {[len(batch) for batch in solution[:-1]]}")

    batches = []
    for i in solution[:-1]:
        wos = []
        for j in i:
            wos.append(work_orders[j - 1][1])
        batches.append(wos)

    return grade, batches


def heuristic(*, min_batch, max_batch, n_people, max_batch_time, max_total_time, max_wo_count, running_time):
    best_grade = -10e6
    best_batches = []
    best_batch = -1
    for batch in range(min_batch, max_batch + 1):
        print(f'Running Batch: {batch}')
        grade, batches = process(batch=batch, n_people=n_people, max_batch_time=max_batch_time,
                                 max_total_time=max_total_time,
                                 max_wo_count=max_wo_count, running_time=running_time)
        print(f'Batch: {batch} Grade: {grade}')
        if grade > best_grade:
            best_batches = batches
            best_batch = batch
    print(f'Best Batch: {best_batch:0.2f} Best Grade: {best_grade:0.2f}\nBest Batches: {best_batches}')
    return best_batches


if __name__ == '__main__':
    heuristic(min_batch=10, max_batch=15, n_people=20, max_batch_time=0.8, max_total_time=8, max_wo_count=40,
              running_time=0.2)
