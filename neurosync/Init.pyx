cimport cython
cimport numpy as np
import numpy as np
import sys, os

import GlobalVars as GV
from collections import deque


cpdef int coord_to_ind(int x, int y):
    return y * GV.sim_params["max_core_x_in_total"] + x


# We assume a tree-based synchronization
# Set the root node (core) and parent / child for each node (core)
cpdef init_sync_topology():
    cdef int root_x = GV.sim_params["max_core_x_in_total"] // 2
    cdef int root_y = GV.sim_params["max_core_y_in_total"] // 2
    cdef object queue

    root_ind = coord_to_ind(root_x, root_y)
    root_core = GV.cores[root_ind]
    root_core.router.sync_role = EL.SyncroleIndex.root
    GV.sync_root = root_core

    queue = deque()
    queue.append( (root_x, root_y) )

    while queue:
        x, y = queue.popleft()
        core_ind = coord_to_ind(x, y)
        core = GV.cores[core_ind]

        if x > 0:
            left_core = GV.cores[coord_to_ind(x - 1, y)]
            if left_core.router.sync_role == EL.SyncroleIndex.syncrole_max:
                left_core.router.parent_x = x
                left_core.router.parent_y = y
                left_core.router.sync_role = EL.SyncroleIndex.non_root
                core.router.num_children += 1
                core.router.children.append((x-1, y))
                queue.append( (x-1, y) )
        if x < GV.sim_params["max_core_x_in_total"] - 1:
            right_core = GV.cores[coord_to_ind(x + 1, y)]
            if right_core.router.sync_role == EL.SyncroleIndex.syncrole_max:
                right_core.router.parent_x = x
                right_core.router.parent_y = y
                right_core.router.sync_role = EL.SyncroleIndex.non_root
                core.router.num_children += 1
                core.router.children.append((x+1, y))
                queue.append( (x+1, y) )
        if y > 0:
            bot_core = GV.cores[coord_to_ind(x, y - 1)]
            if bot_core.router.sync_role == EL.SyncroleIndex.syncrole_max:
                bot_core.router.parent_x = x
                bot_core.router.parent_y = y
                bot_core.router.sync_role = EL.SyncroleIndex.non_root
                core.router.num_children += 1
                core.router.children.append((x, y-1))
                queue.append( (x, y-1) )
        if y < GV.sim_params["max_core_y_in_total"] - 1:
            top_core = GV.cores[coord_to_ind(x, y + 1)]
            if top_core.router.sync_role == EL.SyncroleIndex.syncrole_max:
                top_core.router.parent_x = x
                top_core.router.parent_y = y
                top_core.router.sync_role = EL.SyncroleIndex.non_root
                core.router.num_children += 1
                core.router.children.append((x, y+1))
                queue.append( (x, y+1) )

        if core.router.num_children == 0:
            core.router.sync_role = EL.SyncroleIndex.non_root


# Load constant parameters (neuron / synapse model)
cpdef load_constant(neu_const_filename, syn_const_filename):
    neu_const_dict = np.load(neu_const_filename, allow_pickle=True)
    syn_const_dict = np.load(syn_const_filename, allow_pickle=True).item()

    GV.neu_consts = neu_const_dict
    for typ in range(len(GV.neu_consts)):
        # if the reset or E_L is not explicitly set => use E_0 instead
        if not "reset" in GV.neu_consts[typ]:
            GV.neu_consts[typ]["reset"] = GV.neu_consts[typ]["E_0"]
        if not "E_L" in GV.neu_consts[typ]:
            GV.neu_consts[typ]["E_L"] = GV.neu_consts[typ]["E_0"]

    GV.syn_consts = syn_const_dict


# Map neurons to cores
cpdef mapping(list gid_to_core, list lid_to_gid, mapping_filename):
    cdef int core_num
    cdef int neu
    cdef int neu_index
    cdef int core_index
    cdef int core

    core_num = GV.sim_params["used_core_num"]

    num_neu_per_core = [0 for _ in range(core_num)]
    num_poisson_per_core = [0 for _ in range(core_num)]

    if core_num == 1:
        node_list = [[i for i in range(GV.sim_params["total_neu_num"] + GV.sim_params["total_poisson_num"])]]
    else:
        node_list = np.load(mapping_filename, allow_pickle=True, encoding="bytes")["node_list"]

    for core in range(core_num):
        for ind in range(len(node_list[core])):            
            gid_to_core[node_list[core][ind]] = (core, ind)
            if node_list[core][ind] >= GV.sim_params["total_neu_num"]:
                num_poisson_per_core[core] += 1
            else:
                num_neu_per_core[core] += 1
    
    GV.sim_params["neu_per_core"] = num_neu_per_core
    GV.sim_params["poisson_per_core"] = num_poisson_per_core

    # map local neuron to global neuron (for debugging purpose)
    for core in range(core_num):
        lid_to_gid[core] = [0 for _ in range(GV.sim_params["neu_per_core"][core] + GV.sim_params["poisson_per_core"][core])]

    for neu in range(GV.sim_params['total_neu_num'] + GV.sim_params["total_poisson_num"]):
        (core_index, neu_index) = gid_to_core[neu]
        lid_to_gid[core_index][neu_index] = neu


# initialize the states using the predefined initial states
cpdef init_state(list neu_states, list gid_to_core, state_filename):
    cdef int core_num
    cdef int core

    core_num = GV.sim_params['used_core_num']
    for core in range(core_num):
        neu_states[core] = [{} for _ in range(GV.sim_params["neu_per_core"][core])]

    # initialize lid to gid dat array
    loaded_states = np.load(state_filename, allow_pickle=True)

    for n in range(GV.sim_params['total_neu_num']):
        (core_index, neu_index) = gid_to_core[n]
        neu_states[core_index][neu_index] = loaded_states[n]


# load required connections
cpdef tuple load_connection (conn_filename, state_filename, mapping_filename, hw_mapping_filename):
    cdef int core_num
    cdef int chip_num
    cdef int core
    cdef int num_neu
    cdef int x
    cdef int arr_ind = 0

    cdef list gid_to_core
    cdef list lid_to_gid
    cdef list neu_states_dat

    cdef list syn_mem_weight
    cdef list syn_mem_others
    cdef list syn_mem_debug

    cdef list inverse_table

    cdef list pid_to_gid
    cdef list gid_to_pid

    cdef list core_ind
    cdef list core_dat
    cdef list core_mem_ind

    core_num = GV.sim_params['used_core_num']
    chip_num = GV.sim_params["chip_num"]
    num_neu = GV.sim_params['total_neu_num'] + GV.sim_params["total_poisson_num"]

    gid_to_core = [(-1, -1) for _ in range(num_neu)]
    lid_to_gid = [[] for _ in range(core_num)]
    # load the metis mapping file
    mapping(gid_to_core, lid_to_gid, mapping_filename)

    if os.path.exists(hw_mapping_filename):
        hw_mapping = np.load(hw_mapping_filename, allow_pickle=True)
        conn_dat = hw_mapping["conn_dat"].tolist()
    else:

        # The routing procedure of NeuroSync
        # 1. When firing a spike => the source core uses the core_ind table which provides the start & end address of the core_dat memory
        # 2. The core iterates over the start to end address of the core_dat memory where each entry provides
        #    1) the destination core id
        #    2) the start & end address to use when indexing the synaptic memory @ the destination core
        #    3) the pid (pre-synaptic neuron id) @ the destination core
        #    and uses the data to generate a spike packet
        # 3. The router sends the packet to the destination core using information (2-1)
        # 4. After receiving the spike, the destination core uses (2-2) and (2-3) to perform weight accumulation & learning

        # Neuron Identification
        # 1. lid: local neuron id (dedicated to each core)
        # 2. gid: global neuron id (dedicated to a target network) (Not actually used except for debugging purpose)
        # 3. pid: pre-synaptic neuron id (dedicated to each core)

        # Synaptic Memory (connection information memory)
        # 1. syn_mem_weight:
        #    1) weight of a synapse
        # 2. syn_mem_others:
        #    1) destination neuron's lid
        #    2) synaptic delay
        #    3) learning enable (plastic for static)
        #    4) last spike time (last time the source or destination neuron fired a spike)
        #    5) checkpoint address (address of the checkpoint which keeps the dirty copy of the entry)

        # dst / weight / delay
        syn_mem_weight = [[] for core in range(core_num)]
        syn_mem_others = [[] for core in range(core_num)]
        syn_mem_debug  = [[] for core in range(core_num)]

        # inverse_table (this is required for post-learning)
        # -> can search for the pre synaptic neurons for a destination neuron
        inverse_table = [[[] for _ in range(GV.sim_params["neu_per_core"][core] + GV.sim_params["poisson_per_core"][core])] \
            for core in range(core_num)]

        # pid: the source neuron id for a core
        pid_to_gid = [[] for _ in range(core_num)]
        gid_to_pid = [{} for _ in range(core_num)]

        # core_ind: indirection table for a source neuron
        # -> for a source neuron, an indirection table provides start & end address of core_dat memory
        core_ind = [[0 for _ in range(GV.sim_params["neu_per_core"][core] + GV.sim_params["poisson_per_core"][core] + 1)] \
            for core in range(core_num)]
        # returns the list of destination cores
        core_dat = [[] \
            for core in range(core_num)]
        # temporary data structure to generate core_dat
        core_mem_ind = [0 for _ in range(core_num)]


        print("\nestablishing connection ...")
        total_list = np.load(conn_filename)

        # src_mem_ind / prev =>
        # src_mem_ind_prev & src_mem_ind
        # indicate the start & end address of the synaptic memory (for a given source neuron)
        src_mem_ind = [0 for _ in range(core_num)]
        src_mem_ind_prev = [0 for _ in range(core_num)]

        for x in range(num_neu):
            # x is the source and y is the destination

            (src_core_index, x_ind) = gid_to_core[x]
            core_ind[src_core_index][x_ind] = core_mem_ind[src_core_index]

            dst_core_set = []
            for core in range(core_num):
                src_mem_ind_prev[core] = src_mem_ind[core]

            # make an array of dictionary
            src_to_addr = [{} for _ in range(core_num)]

            while not arr_ind == len(total_list):
                (s,y,w,d, en) = total_list[arr_ind]
                if not s == x: break
                # core_index => core index of the destination neuron
                (core_index, y_ind) = gid_to_core[y]
                # dst_lid, weight, delay, enable (plastic?), lastspike, weight_chkpt_addr

                entry = [None for _ in range(<int>EL.SynmemIndex.synmem_max)]
                entry[<int>EL.SynmemIndex.dst_lid] = y_ind
                entry[<int>EL.SynmemIndex.delay] = d if d < GV.sim_params["max_syn_delay"] else GV.sim_params["max_syn_delay"] - 1
                entry[<int>EL.SynmemIndex.learning_en] = en
                entry[<int>EL.SynmemIndex.lastspike_t] = 0
                entry[<int>EL.SynmemIndex.chkpt_addr] = -1
                syn_mem_weight[core_index].append(w)
                syn_mem_others[core_index].append(entry)
                syn_mem_debug[core_index].append((s,y))

                if s not in pid_to_gid[core_index]:
                    gid_to_pid[core_index][s] = len(pid_to_gid[core_index])
                    pid_to_gid[core_index].append(s)

                inverse_table[core_index][y_ind].append((src_mem_ind[core_index], gid_to_pid[core_index][s]))
                src_mem_ind[core_index] = src_mem_ind[core_index] + 1

                if not core_index in dst_core_set:
                    dst_core_set.append(core_index)

                    # inverse table (input: src core index => output: memory addr)
                    src_to_addr[src_core_index][core_index] = \
                        len(core_dat[src_core_index])

                    core_dat[src_core_index].append((core_index, 0, 0, 0))
                    core_mem_ind[src_core_index] += 1
                arr_ind += 1

            for core in dst_core_set:
                # start addr / dst addr
                core_dat_addr = src_to_addr[src_core_index][core]
                # addresses of the starting and finishing address

                # append the pid 
                core_dat[src_core_index][core_dat_addr] = \
                    (core, src_mem_ind_prev[core], src_mem_ind[core], len(pid_to_gid[core]) - 1)

        for core in range(core_num):
            core_ind[core][GV.sim_params["neu_per_core"][core] + GV.sim_params["poisson_per_core"][core]] = core_mem_ind[core]

        conn_dat = [{"syn_mem_weight" : syn_mem_weight[core]
            , "syn_mem_others" : syn_mem_others[core]
            , "syn_mem_debug" : syn_mem_debug[core]
            , "core_ind" : core_ind[core]
            , "core_dat" : core_dat[core]
            , "inverse_table" : inverse_table[core]
            , "pid_to_gid" : pid_to_gid[core]
            , "lid_to_gid" : lid_to_gid[core]}
                      for core in range(GV.sim_params['used_core_num'])]

        np.savez(hw_mapping_filename, conn_dat = conn_dat)
        
    for core in range(GV.sim_params['used_core_num']):
        assert(len(conn_dat[core]["lid_to_gid"]) > 0 and "The number of neurons mapped to a core should be at least 1. Use smaller number of chips")
        assert(len(conn_dat[core]["syn_mem_weight"]) > 0 and "The number of connections mapped to a core should be at least 1. Use smaller number of chips")
        assert(len(conn_dat[core]["syn_mem_others"]) > 0 and "The number of connections mapped to a core should be at least 1. Use smaller number of chips")

    neu_states_dat = [None for _ in range(core_num)]
    init_state(neu_states_dat, gid_to_core, state_filename)

    return conn_dat, neu_states_dat


cpdef init_network (network_filename):
    network_dict = np.load(network_filename, allow_pickle=True).item()
    assert((network_dict["simtime"] >= GV.sim_params["max_timestep"]) and
            "The defined simulation time for a network is shorter than the target simulation time.\nPlease regenerate the benchmark or run the simulation for shorter period")
    GV.sim_params['timestep_per_sec'] = network_dict["timestep_per_sec"]
    GV.sim_params['total_nE_num'] = network_dict["ne_sim"]
    GV.sim_params['total_nI_num'] = network_dict["ni_sim"]
    GV.sim_params['total_neu_num'] = GV.sim_params['total_nE_num'] + GV.sim_params['total_nI_num']


# load the external stimulus-related parameters
cpdef load_stimulus (stimulus_filename):
    # This work considers only two types of external stimulus
    # 1) poisson neuron
    # 2) input current

    stimulus_dict = np.load(stimulus_filename, allow_pickle=True).item()
    GV.sim_params["stimulus_dict"] = stimulus_dict

    if stimulus_dict["valid_list"]["poisson_neurons"]:
        GV.sim_params["total_nPE_num"] = stimulus_dict["poisson_neurons"]["npe_sim"]
        GV.sim_params["total_nPI_num"] = stimulus_dict["poisson_neurons"]["npi_sim"]
        GV.sim_params["total_poisson_num"] = GV.sim_params['total_nPE_num'] + GV.sim_params['total_nPI_num']
    else:
        GV.sim_params["total_nPE_num"] = 0
        GV.sim_params["total_nPI_num"] = 0
        GV.sim_params["total_poisson_num"] = 0
