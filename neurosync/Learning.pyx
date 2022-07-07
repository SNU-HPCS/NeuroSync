cimport cython
import numpy as np
cimport numpy as np

import GlobalVars as GV
import copy


cdef class LearningModule:

    # initialize the states
    def __init__(self, core):
        cdef int pid

        self.core = core

        self.pre_traces = np.array([[0 for _ in range(<int>EL.PretIndex.pret_max)] for _ in range(core.pid_length)], dtype=np.dtype("f"))
        for pid in range(core.pid_length):
            self.pre_traces[pid][<int>EL.PretIndex.x_t] = 1.
            self.pre_traces[pid][<int>EL.PretIndex.u_t] = 1.

        self.pre_traces_temp = np.array([0 for _ in range(<int>EL.PretIndex.pret_max)], dtype=np.dtype("f"))
        self.pre_traces_temp[<int>EL.PretIndex.x_t] = 1.
        self.pre_traces_temp[<int>EL.PretIndex.u_t] = 1.

        # TODO: Extend the parameters for each type (Define type even for synapses)
        self.trace_params = [0. for _ in range(<int>EL.PreTraceParamIndex.prett_param_max)]
        if "decay_stdp" in GV.syn_consts:
            self.trace_params[<int>EL.PreTraceParamIndex.prett_param_decay_stdp] = GV.syn_consts["decay_stdp"]
        if "decay_triplet" in GV.syn_consts:
            self.trace_params[<int> EL.PreTraceParamIndex.prett_param_decay_triplet] = GV.syn_consts["decay_triplet"]
        if "decay_suppress" in GV.syn_consts:
            self.trace_params[<int> EL.PreTraceParamIndex.prett_param_decay_suppress] = GV.syn_consts["decay_suppress"]
        if "decay_eps" in GV.syn_consts:
            self.trace_params[<int> EL.PreTraceParamIndex.prett_param_decay_eps] = GV.syn_consts["decay_eps"]
        if "decay_x" in GV.syn_consts:
            self.trace_params[<int> EL.PreTraceParamIndex.prett_param_decay_x] = GV.syn_consts["decay_x"]
        if "decay_u" in GV.syn_consts:
            self.trace_params[<int> EL.PreTraceParamIndex.prett_param_decay_u] = GV.syn_consts["decay_u"]

        self.learn_params = [0. for _ in range(<int>EL.LearnParamIndex.learn_param_max)]
        if "U" in GV.syn_consts:
            self.learn_params[<int>EL.LearnParamIndex.learn_param_U] = GV.syn_consts["U"]
        if "eta_stdp" in GV.syn_consts:
            self.learn_params[<int>EL.LearnParamIndex.learn_param_eta_stdp] = GV.syn_consts["eta_stdp"]
        if "eta_triplet" in GV.syn_consts:
            self.learn_params[<int>EL.LearnParamIndex.learn_param_eta_triplet] = GV.syn_consts["eta_triplet"]
        if "gmax" in GV.syn_consts:
            self.learn_params[<int>EL.LearnParamIndex.learn_param_gmax] = GV.syn_consts["gmax"]
        if "alpha" in GV.syn_consts:
            self.learn_params[<int>EL.LearnParamIndex.learn_param_alpha] = GV.syn_consts["alpha"]
        if "slope" in GV.syn_consts:
            self.learn_params[<int>EL.LearnParamIndex.learn_param_slope] = GV.syn_consts["slope"]
        if "th_low_ca" in GV.syn_consts:
            self.learn_params[<int>EL.LearnParamIndex.learn_param_th_low_ca] = GV.syn_consts["th_low_ca"]
        if "th_high_ca" in GV.syn_consts:
            self.learn_params[<int>EL.LearnParamIndex.learn_param_th_high_ca] = GV.syn_consts["th_high_ca"]
        if "th_v_ca" in GV.syn_consts:
            self.learn_params[<int>EL.LearnParamIndex.learn_param_th_v] = GV.syn_consts["th_v"]
        if "up_dn" in GV.syn_consts:
            self.learn_params[<int>EL.LearnParamIndex.learn_param_up_dn] = GV.syn_consts["up_dn"]
        

    # perform post learning (this is called when flushing the deferred post learning)
    cpdef int post_learn(self, int lid, int time, int ts, Checkpoint.Checkpoint checkpoint):
        #cdef int spiked
        cdef int mem_addr
        cdef int src_pid
        cdef int dst_lid
        cdef float weight
        cdef float new_weight
        cdef int syn_delay
        cdef int enable
        cdef int lastspike_t

        cdef int learning_cyc

        synapse_model = GV.syn_consts["rule"]
        learning_cyc = 0

        # iterate over the pre-synaptic neurons
        for [mem_addr, src_pid] in self.core.inverse_table[lid]:
            # if the spike is replayed => the weight has already been updated
            if checkpoint.check_replay(src_pid, ts): continue

            weight = self.core.syn_mem_weight[mem_addr]
            dst_lid = self.core.syn_mem_others[mem_addr][<int>EL.SynmemIndex.dst_lid]
            syn_delay = self.core.syn_mem_others[mem_addr][<int>EL.SynmemIndex.delay]
            enable = self.core.syn_mem_others[mem_addr][<int>EL.SynmemIndex.learning_en]
            lastspike_t = self.core.syn_mem_others[mem_addr][<int>EL.SynmemIndex.lastspike_t]

            assert(dst_lid == lid)
            if enable:

                # Update the pre-traces in the form of temporary traces
                # On-demand trace updates are not permanently stored (for easier collective trace update)
                self.reset_temp_pre_trace(src_pid)
                for t in range(ts + 1):
                    self.pre_trace_update(src_pid, 1, checkpoint.src_spike_history[src_pid][t])
                    # we assume that on-demand trace update is not parallelized
                    learning_cyc += 1

                # perform post-learning using the temporarily updated traces
                new_weight = self.post_learning_rule(
                    weight, lastspike_t, time, synapse_model, lid = lid)
            else:
                new_weight = weight

            self.core.syn_mem_weight[mem_addr] = new_weight
            self.core.syn_mem_others[mem_addr][<int>EL.SynmemIndex.lastspike_t] = time
            self.core.syn_mem_others[mem_addr][<int>EL.SynmemIndex.chkpt_addr] = -1

            learning_cyc += 1
        
        learning_cyc += GV.timing_params["learning_pipeline"]

        return learning_cyc


    # perform temporary post learn (This is used for weight replay
    cpdef float post_learn_temp(self, int lid, float weight, int lastspike_t, int time):
        synapse_model = GV.syn_consts["rule"]

        return self.post_learning_rule(weight, lastspike_t, time, synapse_model, lid = lid)


    cpdef float post_learning_rule(self, float weight, int lastspike_t, int time, synapse_model, int lid):
                        
        cdef float new_weight

        if "TRIPLET" in synapse_model:
            new_weight = weight + self.pre_traces_temp[<int>EL.PretIndex.k_plus_t] * \
                         (self.learn_params[<int>EL.LearnParamIndex.learn_param_eta_stdp] +
                          self.pre_traces_temp[<int>EL.PretIndex.k_plus_triplet_t] *
                          self.learn_params[<int>EL.LearnParamIndex.learn_param_eta_triplet])

        elif "SUPPRESS" in synapse_model:
            new_weight = weight + self.learn_params[<int>EL.LearnParamIndex.learn_param_eta_stdp] * \
                         (self.learn_params[<int>EL.LearnParamIndex.learn_param_gmax] - weight) * \
                         self.pre_traces_temp[<int>EL.PretIndex.k_plus_t] * \
                         (1. - self.core.neuron_module.post_traces[lid][<int>EL.PosttIndex.post_eps_t])

        elif "STP" in synapse_model:
            new_weight = weight + self.learn_params[<int>EL.LearnParamIndex.learn_param_eta_stdp] * \
                         self.pre_traces_temp[<int>EL.PretIndex.k_plus_t]

        elif "VOLTAGE" in synapse_model:
            new_weight = weight + (time - lastspike_t) * self.learn_params[<int>EL.LearnParamIndex.learn_param_slope]

        else:
            raise "NOT IMPLEMENTED"

        if new_weight < 0:
            new_weight = 0.
        if new_weight > self.learn_params[<int> EL.LearnParamIndex.learn_param_gmax]:
            new_weight = self.learn_params[<int> EL.LearnParamIndex.learn_param_gmax]

        return new_weight


    cpdef reset_temp_pre_trace(self, int pid):
        cdef int index
        for index in range(<int>EL.PretIndex.pret_max):
            self.pre_traces_temp[index] = self.pre_traces[pid][index]


    cpdef pre_trace_update(self, int src_pid, int temp, int is_spiked):

        cdef float [:] traces = self.pre_traces_temp if temp else self.pre_traces[src_pid]

        traces[<int>EL.PretIndex.k_plus_t] *= \
            self.trace_params[<int>EL.PreTraceParamIndex.prett_param_decay_stdp]
        traces[<int>EL.PretIndex.k_plus_triplet_t] *= \
            self.trace_params[<int>EL.PreTraceParamIndex.prett_param_decay_triplet]

        traces[<int>EL.PretIndex.pre_eps_t] = \
            traces[<int>EL.PretIndex.pre_eps_t_temp]
        traces[<int>EL.PretIndex.pre_eps_t] *= \
            self.trace_params[<int>EL.PreTraceParamIndex.prett_param_decay_eps]
        traces[<int>EL.PretIndex.pre_eps_t_temp] = \
            traces[<int>EL.PretIndex.pre_eps_t]

        traces[<int>EL.PretIndex.decayx_t] = \
            traces[<int>EL.PretIndex.decayx_t_temp]
        traces[<int>EL.PretIndex.decayx_t] *= \
            self.trace_params[<int>EL.PreTraceParamIndex.prett_param_decay_x]
        traces[<int>EL.PretIndex.decayx_t_temp] = \
            traces[<int>EL.PretIndex.decayx_t]

        traces[<int>EL.PretIndex.decayu_t] = \
            traces[<int>EL.PretIndex.decayu_t_temp]
        traces[<int>EL.PretIndex.decayu_t] *= \
            self.trace_params[<int>EL.PreTraceParamIndex.prett_param_decay_u]
        traces[<int>EL.PretIndex.decayu_t_temp] = \
            traces[<int>EL.PretIndex.decayu_t]

        if is_spiked:
            if "SUPPRESS" in GV.syn_consts["rule"]:
                traces[<int>EL.PretIndex.k_plus_t] += (1. - traces[<int>EL.PretIndex.pre_eps_t])
            else:
                traces[<int>EL.PretIndex.k_plus_t] += 1

            traces[<int>EL.PretIndex.k_plus_triplet_t] += 1
            traces[<int>EL.PretIndex.pre_eps_t_temp] = 1
            traces[<int>EL.PretIndex.decayx_t_temp] = 1
            traces[<int>EL.PretIndex.decayu_t_temp] = 1


    # perform pre-learning for the target synapse using the temporary trace
    cpdef tuple pre_learning_rule(self, int lid, float weight, int lastspike_t, int time):
        cdef float delta = 1.
        cdef float new_weight
        cdef float effective_weight

        synapse_model = GV.syn_consts["rule"]
        neuron_module = self.core.neuron_module

        if "TRIPLET" in synapse_model:
            new_weight = weight + \
                         (neuron_module.post_traces_temp[lid][<int>EL.PosttIndex.k_minus_t] -
                          self.learn_params[<int>EL.LearnParamIndex.learn_param_alpha]) * \
                         (self.learn_params[<int>EL.LearnParamIndex.learn_param_eta_stdp] +
                          (neuron_module.post_traces_temp[lid][<int>EL.PosttIndex.k_minus_triplet_t] -
                           self.learn_params[<int>EL.LearnParamIndex.learn_param_alpha]) *
                          self.learn_params[<int>EL.LearnParamIndex.learn_param_eta_triplet])

        elif "SUPPRESS" in synapse_model:
            new_weight = weight - self.learn_params[<int>EL.LearnParamIndex.learn_param_eta_stdp] * \
                         weight * neuron_module.post_traces_temp[lid][<int>EL.PosttIndex.k_minus_t] * \
                         (1. - self.pre_traces_temp[<int>EL.PretIndex.pre_eps_t])
            
        elif "STP" in synapse_model:
            new_weight = weight + (neuron_module.post_traces_temp[lid][<int>EL.PosttIndex.k_minus_t] -
                                   self.learn_params[<int>EL.LearnParamIndex.learn_param_alpha]) * \
                         self.learn_params[<int>EL.LearnParamIndex.learn_param_eta_stdp]

            self.pre_traces_temp[<int>EL.PretIndex.x_t] = 1.0 + \
                                                          (self.pre_traces_temp[<int>EL.PretIndex.x_t] -
                                                           self.pre_traces_temp[<int>EL.PretIndex.x_t] *
                                                           self.pre_traces_temp[<int>EL.PretIndex.u_t] - 1.0) * \
                                                          self.pre_traces_temp[<int>EL.PretIndex.decayx_t]
            self.pre_traces_temp[<int>EL.PretIndex.u_t] = self.learn_params[<int>EL.LearnParamIndex.learn_param_U] + \
                                                          self.pre_traces_temp[<int>EL.PretIndex.u_t] * \
                                                          (1.0 - self.learn_params[<int>EL.LearnParamIndex.learn_param_U]) * \
                                                          self.pre_traces_temp[<int>EL.PretIndex.decayu_t]
            delta = self.pre_traces_temp[<int>EL.PretIndex.x_t] * \
                    self.pre_traces_temp[<int>EL.PretIndex.u_t]

        elif "VOLTAGE" in synapse_model:
            new_weight = weight + (time - lastspike_t) * \
                         self.learn_params[<int> EL.LearnParamIndex.learn_param_slope]
            
            if self.learn_params[<int>EL.LearnParamIndex.learn_param_th_low_ca] > \
                    neuron_module.post_traces_temp[lid][<int>EL.PosttIndex.ca_t] > \
                    self.learn_params[<int>EL.LearnParamIndex.learn_param_th_high_ca]:

                if neuron_module.states[lid][<int>EL.StateIndex.v_t] > \
                        self.learn_params[<int>EL.LearnParamIndex.learn_param_th_v]:
                    new_weight += self.learn_params[<int>EL.LearnParamIndex.learn_param_up_dn]
                else:
                    new_weight -= self.learn_params[<int>EL.LearnParamIndex.learn_param_up_dn]

        else:
            raise "NOT IMPLEMENTED"

        if new_weight < 0:
            new_weight = 0.
        if new_weight > self.learn_params[<int>EL.LearnParamIndex.learn_param_gmax]:
            new_weight = self.learn_params[<int>EL.LearnParamIndex.learn_param_gmax]

        effective_weight = new_weight * delta

        return new_weight, effective_weight
