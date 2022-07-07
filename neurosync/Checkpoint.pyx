cimport cython
import numpy as np
cimport numpy as np

import GlobalVars as GV
import KeyDict
import heapq
import copy
import sys

cdef class Checkpoint:
    def __init__(self, int ind, int neu_num, int pre_neu_num, RRManager.RRManager rr_manager):
        self.ind = ind
        self.state_chkpt = np.asarray([[0 for _ in range(<int>EL.StateIndex.state_max)] for _ in range(neu_num)], dtype=np.dtype("f"))
        # In the actual hardware module,
        # NeuroSync only keeps lastspike_t and address here
        # (but we store all the metadata for debugging purpose
        self.syn_mem_chkpt_weight = []
        self.syn_mem_chkpt_others = []

        self.pre_neu_num = pre_neu_num
        self.src_spike_history = [[False for _ in range(GV.sim_params["max_sync_period"])]
                                  for _ in range(self.pre_neu_num)]
        self.src_spiked_in_period = [False for _ in range(self.pre_neu_num)]
        self.rr_manager = rr_manager


    # Check if the weight should be replayed
    # as the learning should be processed in the time order
    # ex) Spike arrival @ t5 (Learning) => Spike arrival @ t4 (Incorrect Learning)
    # => Replay from t0 -> t4 -> t5
    cpdef int check_replay(self, int src_pid, int pos):
        if sum(self.src_spike_history[src_pid][pos+1:]) > 0:
            return True
        else:
            return False


    # flush the deferred post learnings
    cpdef int flush_learning(self, Neuron.NeuronModule neuron_module, Learning.LearningModule learning_module):
        # profile the total number of deferred learnings
        cdef int ts
        cdef int lid
        cdef int learning_cyc = 0

        for ts in range(GV.sim_params["prev_sync_period"]):
        
            # on-demand post trace update
            for lid in range(len(self.rr_manager.out_spike_history[ts])):
                spiked = self.rr_manager.out_spike_history[ts][lid]
                neuron_module.post_trace_update(lid, spiked, temp=0)
                learning_cyc += 1
                if spiked:
                    learning_cyc += learning_module.post_learn(lid, GV.chkpt_timestep[self.ind] + ts + 1, ts, self)

        return learning_cyc


    # collectively update the pre-traces
    cpdef int collective_pretrace_update(self, Learning.LearningModule learning_module):
        cdef int ts
        cdef int src_pid
        cdef int trace_update_cyc = 0

        for ts in range(GV.sim_params["prev_sync_period"]):
            # on-demand pre trace update
            for src_pid in range(len(self.src_spike_history)):
                learning_module.pre_trace_update(src_pid, 0, self.src_spike_history[src_pid][ts])
            trace_update_cyc += (len(self.src_spike_history) +
                                 GV.timing_params["trace_update_pipeline"]) // \
                                GV.sim_params["num_pretrace_engine"]

        return trace_update_cyc
        

    # reset the local / remote history
    cpdef int reset(self):
        self.src_spike_history = [[False for _ in range(GV.sim_params["max_sync_period"])] for _ in range(self.pre_neu_num)]
        self.src_spiked_in_period = [False for _ in range(self.pre_neu_num)]

        return self.pre_neu_num


    # allocate a dirty copy entry
    cpdef tuple allocate_weight_chkpt(self, int addr, float [:] syn_mem_weight, int [:,:] syn_mem_others):

        cdef float weight_dat
        cdef int [:] others_dat = np.array([0. for _ in range(<int>EL.SynmemIndex.synmem_max)], dtype=np.dtype("i"))
        cdef int index = 0
        cdef int chkpt_addr


        ### check if the entry has been allocated
        if syn_mem_others[addr][<int>EL.SynmemIndex.chkpt_addr] == -1:
            # if the entry has not been allocated => allocate a new entry
            
            weight_dat = syn_mem_weight[addr]
            for index in range(<int>EL.SynmemIndex.synmem_max):
                others_dat[index] = syn_mem_others[addr][index]
            
            others_dat[<int>EL.SynmemIndex.chkpt_addr] = addr

            chkpt_addr = len(self.syn_mem_chkpt_weight)
            syn_mem_others[addr][<int>EL.SynmemIndex.chkpt_addr] = chkpt_addr
        
            self.syn_mem_chkpt_weight.append(weight_dat)
            self.syn_mem_chkpt_others.append(others_dat)
        else:
            # if the entry has been allocated => retrieve the allocated entry
            chkpt_addr = syn_mem_others[addr][<int>EL.SynmemIndex.chkpt_addr]
            weight_dat = self.syn_mem_chkpt_weight[chkpt_addr]
            for index in range(<int>EL.SynmemIndex.synmem_max):
                others_dat[index] = self.syn_mem_chkpt_others[chkpt_addr][index]

        return weight_dat, others_dat, chkpt_addr


    # change the checkpoint entry
    cpdef set_weight_chkpt(self, int chkpt_addr, float new_weight, int [:] new_others_dat):
        cdef int index
        assert(self.syn_mem_chkpt_others[chkpt_addr][<int>EL.SynmemIndex.chkpt_addr] == new_others_dat[<int>EL.SynmemIndex.chkpt_addr])
        assert(self.syn_mem_chkpt_others[chkpt_addr][<int>EL.SynmemIndex.learning_en])
        self.syn_mem_chkpt_weight[chkpt_addr] = new_weight
        for index in range(<int>EL.SynmemIndex.synmem_max):
            self.syn_mem_chkpt_others[chkpt_addr][index] = new_others_dat[index]


    # write back the dirty copies to the synaptic memory
    # then, reset the dirty copies 
    cpdef int reset_weight_chkpt(self, float [:] syn_mem_weight, int [:,:] syn_mem_others):
        cdef int addr
        cdef int index
        cdef int chkpt_addr
        cdef int reset_cyc

        reset_cyc = len(self.syn_mem_chkpt_weight)
        for chkpt_addr in range(reset_cyc):
            entry = self.syn_mem_chkpt_others[chkpt_addr]
            addr = entry[<int>EL.SynmemIndex.chkpt_addr]
            assert(syn_mem_others[addr][<int>EL.SynmemIndex.dst_lid] == entry[<int>EL.SynmemIndex.dst_lid])

            syn_mem_weight[addr] = self.syn_mem_chkpt_weight[chkpt_addr]
            for index in range(<int>EL.SynmemIndex.synmem_max):
                syn_mem_others[addr][index] = entry[index]
            syn_mem_others[addr][<int>EL.SynmemIndex.chkpt_addr] = -1

        self.syn_mem_chkpt_weight = []
        self.syn_mem_chkpt_others = []

        return reset_cyc
        

    cpdef float [:] get_state_chkpt(self, int neu_id):
        cdef int index
        cdef float [:] state_dat = np.asarray([0 for _ in range(<int>EL.StateIndex.state_max)], dtype=np.dtype("f"))

        for index in range(<int>EL.StateIndex.state_max):
            state_dat[index] = self.state_chkpt[neu_id][index]
        return state_dat


    cpdef set_state_chkpt(self, int neu_id, float [:] state):
        cdef int index

        for index in range(<int>EL.StateIndex.state_max):        
            self.state_chkpt[neu_id][index] = state[index]
