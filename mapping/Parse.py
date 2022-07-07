import numpy as np
import sys


workload_name = sys.argv[1]
chip_x = int(sys.argv[2])
chip_y = int(sys.argv[3])
core_x = int(sys.argv[4])
core_y = int(sys.argv[5])

chip_num = chip_x * chip_y
core_per_chip = core_x * core_y
used_core_num = chip_num * core_per_chip

core_ind = np.zeros((chip_num, core_per_chip), dtype = int)
for chip in range(chip_num):
    chip_x_ind = int(chip % chip_x)
    chip_y_ind = int(chip / chip_x)
    offset = int(chip_y_ind * (chip_x * core_per_chip) + chip_x_ind * core_x)
    for core in range(core_per_chip):
        core_x_ind = int(core % core_x)
        core_y_ind = int(core / core_x)
        core_ind[chip][core] = int(offset + core_y_ind * chip_x * core_x + core_x_ind)

neu_ind = np.load(workload_name + "/gids_per_chip.npy", allow_pickle=True)

node_list = [[] for _ in range(used_core_num)]
for chip in range(chip_num):
    f = open(workload_name + "/chip_sub_simulation" + str(chip) + ".graph.part." + str(core_per_chip), "r")
    neu_per_chip = f.readlines()

    for neu_lid in range(len(neu_per_chip)):
        core_lid = int(neu_per_chip[neu_lid])
        core_gid = core_ind[chip][core_lid]
        neu_gid = neu_ind[chip][neu_lid]
        node_list[core_gid].append(neu_gid)

for core in range(used_core_num):
    node_list[core].sort()

num_neu = 0
for ind in neu_ind:
    num_neu += len(ind)

np.savez(workload_name + "/mapping_" + str(num_neu) + "_" + str(used_core_num) + ".npz", node_list = node_list)
