import subprocess
import sys
import os

workload_name = sys.argv[1]
benchmark_path = sys.argv[2]

chip_x = int(sys.argv[3])
chip_y = int(sys.argv[4])
max_core_x_in_chip = int(sys.argv[5])
max_core_y_in_chip = int(sys.argv[6])

ufactor = int(sys.argv[7])

used_chip_num = chip_x * chip_y
used_core_num = chip_x * chip_y * max_core_x_in_chip * max_core_y_in_chip

if not os.path.isdir(workload_name):
    os.mkdir(workload_name)

proc = subprocess.Popen(['python3 gen_metis.py {}'\
    .format(workload_name)]\
    , close_fds = True, shell=True, cwd=".")
proc.communicate()

proc = subprocess.Popen(['gpmetis chip_simulation.graph -ufactor {} {}'\
    .format(ufactor, used_chip_num)]\
    , close_fds = True, shell=True, cwd=workload_name)
proc.communicate()

print("Gpmetis mapping start")

proc = subprocess.Popen(['gpmetis chip_simulation.graph -ufactor {} {}'\
    .format(ufactor, used_chip_num)]\
    , close_fds = True, shell=True, cwd=workload_name)
proc.communicate()

proc = subprocess.Popen(['python3 gen_submetis.py {} {}'\
    .format(workload_name, used_chip_num)]\
    , close_fds = True, shell=True, cwd=".")
proc.communicate()

used_local_core = max_core_x_in_chip * max_core_y_in_chip
for i in range(used_chip_num):
    proc = subprocess.Popen(['gpmetis chip_sub_simulation{}.graph -ufactor {} {}'\
        .format(i, ufactor, used_local_core)]\
        , close_fds = True, shell=True, cwd=workload_name)
    proc.communicate()

proc = subprocess.Popen(['python3 Parse.py {} {} {} {} {}'\
    .format(workload_name, chip_x, chip_y, max_core_x_in_chip, max_core_y_in_chip)]\
    , close_fds = True, shell=True, cwd=".")
proc.communicate()
