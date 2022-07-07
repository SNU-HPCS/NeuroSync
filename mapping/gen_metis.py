import os
import numpy as np
import time
import sys

start = time.time()

base_path = sys.argv[1]
workload = sys.argv[2]
n_sim = int(sys.argv[3])

total_arr = np.load(base_path + workload + "/dataset/connection.npy")
total_conn = len(total_arr)

# faster generation for sparse matrix
is_sparse = True
if is_sparse:
    matrix_conn = [[] for _ in range(n_sim)]
    matrix_wgt = [[] for _ in range(n_sim)]
    input_conn = [0 for _ in range(n_sim)]

    n_syn = 0
    for syn in total_arr:
        (src, dst, weight, delay, _) = syn
        input_conn[dst] += 1
        if src < dst:
            # if my index is smaller than dst 
            # ==> dst iteration is yet to be performed
            n_syn += 1
            matrix_conn[src].append(dst)
            matrix_conn[dst].append(src)
            matrix_wgt[src].append(1)
            matrix_wgt[dst].append(1)
        else:
            # if my index is smaller just check if there is dst -> src connection
            if not dst in matrix_conn[src]:
                n_syn += 1
                matrix_conn[src].append(dst)
                matrix_conn[dst].append(src)
                matrix_wgt[src].append(1)
                matrix_wgt[dst].append(1)
            else:
                dst_index = matrix_conn[src].index(dst)
                src_index = matrix_conn[dst].index(src)
                matrix_wgt[src][dst_index] += 1
                matrix_wgt[dst][src_index] += 1

    graph_path = workload
    if not os.path.isdir(graph_path):
        os.makedirs(graph_path)

    print(len(total_arr), n_sim)

    f = open(graph_path + "/chip_simulation.graph", "w")
    f.write(str(n_sim) + " " + str(n_syn) + " 011 2\n")

    for x in range(n_sim):
        f.write("1 ")
        f.write(str(input_conn[x]) + " ")
        for ind in range(len(matrix_conn[x])):
            dst = matrix_conn[x][ind]; wgt = matrix_wgt[x][ind]
            f.write(str(dst + 1) + " " + str(wgt) +" ")
        f.write("\n")

    f.close()

    end = time.time()
    print(end - start)
else:
    matrix_conn = [[False for _ in range(n_sim)] for _ in range(n_sim)]
    matrix_wgt = [[0 for _ in range(n_sim)] for _ in range(n_sim)]
    input_conn = [0 for _ in range(n_sim)]

    n_syn = 0
    for syn in total_arr:
        (src, dst, weight, delay) = syn
        input_conn[dst] += 1
        if matrix_conn[src][dst]:
            matrix_wgt[src][dst] = 2
        else:
            matrix_wgt[src][dst] = 1
            matrix_conn[src][dst] = True

        if matrix_conn[dst][src]:
            matrix_wgt[dst][src] = 2
        else:
            matrix_wgt[dst][src] = 1
            matrix_conn[dst][src] = True

    graph_path = workload
    if not os.path.isdir(graph_path):
        os.makedirs(graph_path)


    f = open(graph_path + "/chip_simulation.graph", "w")
    f.write(str(n_sim) + " " + str(n_syn) + " 011 2\n")

    for x in range(n_sim):
        f.write("1 ")
        f.write(str(input_conn[x]) + " ")
        for y in range(n_sim):
            if(matrix_conn[x][y]):
                wgt = matrix_wgt[x][y]
                f.write(str(y + 1) + " " + str(wgt) +" ")
        f.write("\n")

    f.close()

    end = time.time()
    print(end - start)
