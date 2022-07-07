import sys
import Core
import NoC
import GlobalVars as GV

import EnumList as EL
import numpy as np

import Init
import copy


from timeit import default_timer as timer


def init():
    workload_path = sys.argv[1] + "/"
    mapper_path = sys.argv[2]
    mapping_name = sys.argv[3]
    hw_mapping_name = sys.argv[4]
    GV.sim_params["chip_x"] = int(sys.argv[5])
    GV.sim_params["chip_y"] = int(sys.argv[6])
    GV.sim_params["max_core_x_in_chip"] = int(sys.argv[7])
    GV.sim_params["max_core_y_in_chip"] = int(sys.argv[8])
    GV.sim_params["max_syn_delay"] = int(sys.argv[9])
    GV.sim_params["max_sync_period"] = int(sys.argv[10])
    GV.sim_params["accum_width"] = int(sys.argv[11])
    GV.sim_params["max_timestep"] = int(sys.argv[12])
    GV.sim_params["num_pretrace_engine"] = int(sys.argv[13])
    
    GV.sim_params["setup_timestep"] = int(sys.argv[14])
    GV.sim_params["cur_sync_period"] = 1
    GV.sim_params["prev_sync_period"] = 1
    

    GV.sim_params["workload_path"] = workload_path
    GV.sim_params["history_width"] = GV.sim_params["max_syn_delay"] + GV.sim_params["max_sync_period"] + 2

    GV.sim_params["chip_num"] = GV.sim_params["chip_x"] * GV.sim_params["chip_y"]
    GV.sim_params["used_core_num"] = GV.sim_params["chip_num"] * GV.sim_params["max_core_x_in_chip"] * GV.sim_params["max_core_y_in_chip"]

    GV.sim_params["max_core_x_in_total"] = GV.sim_params["chip_x"] * GV.sim_params["max_core_x_in_chip"]
    GV.sim_params["max_core_y_in_total"] = GV.sim_params["chip_y"] * GV.sim_params["max_core_y_in_chip"]

    GV.neu_consts = [{} for _ in range(EL.NeutypeIndex.neutype_max.value)]
    GV.syn_consts = {}

    Init.init_network(workload_path + "network_parameter.npy")
    Init.load_stimulus(workload_path+"stimulus.npy")
    Init.load_constant(workload_path+"neuron_parameter.npy", workload_path+"synapse_parameter.npy")

    print("Start Connection Initialization\n")
    sys.stdout.flush()

    conn_list, neu_states = Init.load_connection(
        workload_path + "connection.npy",
        workload_path + "initial_states.npy",
        mapper_path + "/" + mapping_name,
        mapper_path + "/" + hw_mapping_name)

    print("End Connection Initialization\n")
    sys.stdout.flush()

    GV.NoC = NoC.NoC()
    GV.cyc = [0 for _ in range(GV.sim_params['used_core_num'])]
    GV.timestep = [0 for _ in range(GV.sim_params['used_core_num'])]
    GV.chkpt_timestep = [0 for _ in range(GV.sim_params['used_core_num'])]
    GV.cores = [Core.Core(ind, conn_list[ind], neu_states[ind]) for ind in range(GV.sim_params['used_core_num'])]
    Init.init_sync_topology()

    GV.spike_out = [[] for _ in range(GV.sim_params["total_neu_num"] + GV.sim_params["total_poisson_num"])]

    state_file = open("state.dat", "w")
    spike_file = open("multicore_spike_out_raw.dat", "w")
    GV.debug_list = {"spike" : spike_file, "state" : state_file}


def simulate():
    GV.sync_root.prev_time = timer()

    simulation_end = False
    counter = 0
    while not simulation_end:
        for core in GV.cores:
            core.core_advance()

        GV.NoC.noc_advance()

        for core in GV.cores:
            if GV.timestep[core.ind] >= GV.sim_params["max_timestep"]:
                simulation_end = True


def stat():
    if GV.debug_list["spike"]:
        for neu in range(GV.sim_params['total_neu_num'] + GV.sim_params['total_poisson_num']):
            GV.debug_list["spike"].write(str(neu) + ": " + str(GV.spike_out[neu]) + "\n")

    spike_out_clean = copy.deepcopy(GV.spike_out)
    for spike_out_per_neu in spike_out_clean:
        remove_spikes = []
        for spike_timestep in spike_out_per_neu:
            if spike_timestep < 0:
                assert (-spike_timestep in spike_out_per_neu)
                remove_spikes.append(spike_timestep)

        for remove_spike in remove_spikes:
            spike_out_per_neu.remove(remove_spike)
            spike_out_per_neu.remove(-remove_spike)
    f = open("multicore_spike_out_clean.dat", "w")
    for neu in range(GV.sim_params['total_neu_num'] + GV.sim_params['total_poisson_num']):
        f.write(str(neu) + ": " + str(spike_out_clean[neu]) + "\n")


def run():
    init()

    print("Initialized Simulation States\n")
    sys.stdout.flush()
    
    simulate()

    print("Simulation Done\n")
    sys.stdout.flush()
    
    stat()

run()
