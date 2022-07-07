import subprocess
import sys
import time
import itertools
import configparser
import ast
import os
import shutil

import numpy as np


def get_list(param):
    return ast.literal_eval(config.get('simulation_parameter', param))


config = configparser.ConfigParser()
config.read(sys.argv[1])

working_directory = config.get('path', 'runspace_path') + config.get('run', 'result_folder_name') + '/'

if not os.path.exists(working_directory):
    os.makedirs(working_directory)

shutil.copy(sys.argv[1], working_directory + "used.cfg")

origin = config.get('path', 'simulator_path') + 'neurosync/'
workload_path = config.get('path', 'workload_path')
mapper_path = config.get('path', 'mapper_path')

#### mapping parameters ####
ufactor = int(config.get('mapping_parameter', 'ufactor'))

list_workload_name = [config.get('run', 'workload_name')]  # argv[1]
list_chip_xy = get_list('chip_xy')  # argv[3], argv[4]
list_max_core_xy_in_chip = get_list('max_core_xy_in_chip')  # argv[5], argv[6]
list_max_syn_delay = get_list('max_syn_delay')  # argv[7]
list_sync_period = get_list('sync_period')  # argv[8]
list_accum_width = get_list('accum_width')  # argv[10]
list_max_timestep = [config.get('run', 'max_timestep')]  # argv[11]
list_pretrace_engine = get_list('pretrace_engine')

setup_timestep = config.get('simulation_parameter', 'setup_timestep')

param_set = list(itertools.product(list_workload_name,
                                   list_chip_xy,
                                   list_max_core_xy_in_chip,
                                   list_max_syn_delay,
                                   list_sync_period,
                                   list_accum_width, 
                                   list_max_timestep, 
                                   list_pretrace_engine))
####################


for param in param_set:
    workload_name = param[0]
    (chip_x, chip_y) = param[1]
    (max_core_x_in_chip, max_core_y_in_chip) = param[2]
    max_syn_delay = param[3]
    sync_period = param[4]
    accum_width = param[5]
    max_timestep = param[6]
    pretrace_engine = param[7]

    folder_name = workload_name + "_" + \
                  "peri" + str(sync_period) + "_" + \
                  str(pretrace_engine) + "pretrace_eng"

    network_dict = np.load(workload_path + workload_name + "/dataset" + "/network_parameter.npy", allow_pickle=True).item()
    stimulus_dict = np.load(workload_path + workload_name + "/dataset" + "/stimulus.npy", allow_pickle=True).item()

    num_neu = network_dict["ne_sim"] + network_dict["ni_sim"]
    if stimulus_dict["valid_list"]["poisson_neurons"]:
        num_neu += (stimulus_dict["poisson_neurons"]["npe_sim"] + stimulus_dict["poisson_neurons"]["npi_sim"])

    used_chip_num = chip_x * chip_y
    used_core_num = chip_x * chip_y * max_core_x_in_chip * max_core_y_in_chip

    mapping_name = "mapping_" + str(num_neu) + "_" + str(used_core_num)
    hw_mapping_name = "hw_mapping_" + str(num_neu) + "_" + str(used_core_num)

    mapping_name += ".npz"
    hw_mapping_name += ".npz"
    mapping_folder_name = mapper_path + workload_name
    mapping_file_name = mapping_folder_name + "/" + mapping_name

    if not os.path.isdir(mapping_folder_name):
        os.mkdir(mapping_folder_name)

    if not os.path.isfile(mapping_file_name):
    #if True:
        print("Gpmetis mapping start")

        proc = subprocess.Popen(['python3 gen_metis.py {} {} {}' \
                                .format(workload_path, workload_name, num_neu)]
                                , close_fds=True, shell=True, cwd=mapper_path)
        proc.communicate()

        proc = subprocess.Popen(['gpmetis chip_simulation.graph -ufactor {} {}'.format(ufactor, used_chip_num)]
                                , close_fds=True, shell=True, cwd=mapper_path + workload_name)
        proc.communicate()

        proc = subprocess.Popen(['python gen_submetis.py {} {} {} {}' \
                                .format(workload_path, workload_name, used_chip_num, num_neu)]
                                , close_fds=True, shell=True, cwd=mapper_path)
        proc.communicate()

        used_local_core = max_core_x_in_chip * max_core_y_in_chip
        for i in range(used_chip_num):
            proc = subprocess.Popen(['gpmetis chip_sub_simulation{}.graph -ufactor {} -ptype rb {}'
                                    .format(i, ufactor, used_local_core)]
                                    , close_fds=True, shell=True, cwd=mapper_path + workload_name)
            proc.communicate()

        proc = subprocess.Popen(['python Parse.py {} {} {} {} {}'
                                .format(workload_name, chip_x, chip_y, max_core_x_in_chip, max_core_y_in_chip)]
                                , close_fds=True, shell=True, cwd=mapper_path)
        proc.communicate()

    argument = workload_path + workload_name + "/dataset" + " " \
               + mapper_path + workload_name + " " \
               + mapping_name + " " \
               + hw_mapping_name + " " \
               + str(chip_x) + " " + str(chip_y) + " " \
               + str(max_core_x_in_chip) + " " + str(max_core_y_in_chip) + " " \
               + str(max_syn_delay) + " " \
               + str(sync_period) + " " \
               + str(accum_width) + " " \
               + str(max_timestep) + " " \
               + str(pretrace_engine) + " " \
               + str(setup_timestep)

    # print(folder_name)

    subprocess.Popen(['rm', '-rf', folder_name], cwd=working_directory).communicate()

    time.sleep(1)

    subprocess.Popen(['cp', '-rf', origin, folder_name],
                     cwd=working_directory).communicate()

    time.sleep(1)

    # compile
    proc = subprocess.Popen(['python3 setup.py build_ext --inplace'],
                            close_fds=True, shell=True, cwd=working_directory + folder_name)
    out, err = proc.communicate()

    time.sleep(1)

    # execute
    # proc = subprocess.Popen(['python3 Main.py ' + argument]\
    command = 'python3 Main.py ' + argument + ' > log 2> err&'
    f = open(working_directory + folder_name + "/command.sh", "w")
    f.write(command)
    f.close()
    proc = subprocess.Popen([command],
                            close_fds=True, shell=True, cwd=working_directory + folder_name)
    out, err = proc.communicate()

    time.sleep(1)
