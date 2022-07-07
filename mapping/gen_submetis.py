import os
import numpy as np
import time
import sys

start = time.time()

base_path = sys.argv[1]
workload = sys.argv[2]
num_chip = int(sys.argv[3])
n_sim = int(sys.argv[4])


is_sparse = True

connection_dat = base_path + workload + "/dataset/connection.npy"
# chip mapping result
chip_mapping_dat = workload + "/chip_simulation.graph.part." + str(num_chip)


total_arr = np.load(connection_dat)

total_conn = len(total_arr)

# chip mapping result (gid to core mapping dat)
mapping_result = open(chip_mapping_dat)
mapping_result = mapping_result.readlines()
mapping_result = list(map(int, mapping_result))

# gid list for each chip
gid_list_per_chip = [[] for _ in range(num_chip)]
# gid to chip id
gid_to_chipid = [0 for _ in range(len(mapping_result))]

for gid in range(len(mapping_result)):
    # convert gid to chip id
    gid_to_chipid[gid] = len(gid_list_per_chip[mapping_result[gid]])    
    gid_list_per_chip[mapping_result[gid]].append(gid)

np.save(workload + "/gids_per_chip.npy", gid_list_per_chip)

neu_per_core = [len(gid_list_per_chip[chip]) for chip in range(num_chip)]
#
matrix_conn = [[[] for _ in range(n_sim)] for _ in range(num_chip)]
matrix_wgt = [[[] for _ in range(n_sim)] for _ in range(num_chip)]
input_conn = [[0 for _ in range(n_sim)] for _ in range(num_chip)]

n_syn = [0 for _ in range(num_chip)]
for syn in total_arr:
    (src, dst, weight, delay, _) = syn

    # get src chip id and dst chip id
    src_chip = mapping_result[src]; dst_chip = mapping_result[dst]

    input_conn[dst_chip][dst] += 1
    # generate connection only if src and dst are in the same chip
    if src_chip == dst_chip:

        # increase connection of the chip
        if src < dst:
            n_syn[src_chip] += 1
            matrix_conn[src_chip][src].append(dst)
            matrix_conn[src_chip][dst].append(src)
            matrix_wgt[src_chip][src].append(1)
            matrix_wgt[src_chip][dst].append(1)
        else:
            # if my index is smaller just check if there is dst -> src connection
            if not dst in matrix_conn[src_chip][src]:
                n_syn[src_chip] += 1
                matrix_conn[src_chip][src].append(dst)
                matrix_conn[src_chip][dst].append(src)
                matrix_wgt[src_chip][src].append(1)
                matrix_wgt[src_chip][dst].append(1)
            else:
                dst_index = matrix_conn[src_chip][src].index(dst)
                src_index = matrix_conn[src_chip][dst].index(src)
                matrix_wgt[src_chip][src][dst_index] += 1
                matrix_wgt[src_chip][dst][src_index] += 1

for chip in range(num_chip):
    
    f = open(workload + "/chip_sub_simulation" + str(chip) + ".graph", "w")
    f.write(str(len(gid_list_per_chip[chip])) + " " + str(n_syn[chip]) + " 011 2\n")
    
    for x in gid_list_per_chip[chip]:
        f.write("1 ")
        f.write(str(input_conn[chip][x]) + " ")
        for ind in range(len(matrix_conn[chip][x])):
            dst = matrix_conn[chip][x][ind]; wgt = matrix_wgt[chip][x][ind]
            f.write(str(gid_to_chipid[dst] + 1) + " " + str(wgt) +" ")
        f.write("\n")
    f.close()

end = time.time()
