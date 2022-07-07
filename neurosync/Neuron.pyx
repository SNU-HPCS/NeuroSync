cimport cython
import numpy as np
cimport numpy as np

import GlobalVars as GV
import copy
from libc.math cimport exp
import sys
        

cdef class NeuronModule:

    def __init__(self,
                 Core.Core core,
                 dict conn_list,
                 list initial_states,
                 Checkpoint.Checkpoint checkpoint):

        #
        self.core = core
        self.ind = core.ind
        self.checkpoint = checkpoint

        # set the lid_to_gid connection / neu_num
        self.lid_to_gid = conn_list["lid_to_gid"]
        self.neu_num = GV.sim_params["neu_per_core"][core.ind]
        self.poisson_num = GV.sim_params["poisson_per_core"][core.ind]

        # set the global and local id
        self.timestep = [GV.leading_timestep - 1 for _ in range(self.neu_num + self.poisson_num)]

        #
        self.acc_weights = [[[0, 0] for _ in range(GV.sim_params["history_width"])] for _ in range(self.neu_num)]

        self.poisson_spike = [False for _ in range(self.poisson_num)]

        self.states = np.asarray([[0 for _ in range(<int>EL.StateIndex.state_max)] for _ in range(self.neu_num)], dtype=np.dtype("f"))
        self.neu_type = [None for _ in range(self.neu_num + self.poisson_num)]
        # set the required states using the initial state
        for lid in range(self.neu_num):
            if "v_t" in initial_states[lid]:
                self.states[lid][<int>EL.StateIndex.v_t] = initial_states[lid]["v_t"]
            if "gE_t" in initial_states[lid]:
                self.states[lid][<int>EL.StateIndex.gE_t] = initial_states[lid]["gE_t"]
            if "gI_t" in initial_states[lid]:
                self.states[lid][<int>EL.StateIndex.gI_t] = initial_states[lid]["gI_t"]
            if "yE_t" in initial_states[lid]:
                self.states[lid][<int>EL.StateIndex.yE_t] = initial_states[lid]["yE_t"]
            if "yI_t" in initial_states[lid]:
                self.states[lid][<int>EL.StateIndex.yI_t] = initial_states[lid]["yI_t"]
            if "w_t" in initial_states[lid]:
                self.states[lid][<int>EL.StateIndex.w_t] = initial_states[lid]["u_t"]
            self.neu_type[lid] = initial_states[lid]["neu_type"]
        
        for lid in range(self.neu_num, self.neu_num + self.poisson_num):
            gid = self.lid_to_gid[lid]
            self.neu_type[lid] = <int>EL.NeutypeIndex.poisson_exc if (gid - GV.sim_params["total_neu_num"]) < GV.sim_params["stimulus_dict"]["poisson_neurons"]["npe_sim"] else <int>EL.NeutypeIndex.poisson_inh

        self.neu_params = [[0. for _ in range(<int>EL.NeuParamIndex.neu_param_max)] for _ in range(len(GV.neu_consts))]
        for neu_type in range(len(self.neu_params)):
            # set the parameters using
            if "v_a" in GV.neu_consts[neu_type]:
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_v_a] = GV.neu_consts[neu_type]['v_a']
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_v_b] = GV.neu_consts[neu_type]['v_b']
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_v_c] = GV.neu_consts[neu_type]['v_c']

                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_u_a] = GV.neu_consts[neu_type]['u_a']
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_u_b] = GV.neu_consts[neu_type]['u_b']
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_u_c] = GV.neu_consts[neu_type]['u_c']
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_u_d] = GV.neu_consts[neu_type]['u_d']

            if "decay_v" in GV.neu_consts[neu_type]:
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_decay_gE] = GV.neu_consts[neu_type]['decay_gE']
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_decay_gI] = GV.neu_consts[neu_type]['decay_gI']
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_decay_v] = GV.neu_consts[neu_type]['decay_v']

            if "decay_ad" in GV.neu_consts[neu_type]:
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_decay_ad] = GV.neu_consts[neu_type]['decay_ad']
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_a] = GV.neu_consts[neu_type]['a']
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_b] = GV.neu_consts[neu_type]['b']
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_d] = GV.neu_consts[neu_type]['d']
                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_v_w] = GV.neu_consts[neu_type]['v_w']

                self.neu_params[neu_type][<int>EL.NeuParamIndex.neu_param_Delta_T] = GV.neu_consts[neu_type]['Delta_T']

        # TODO: Extend the parameters for each type (Define type even for synapses)
        self.trace_params = [0. for _ in range(<int>EL.PostTraceParamIndex.postt_param_max)]
        if "decay_stdp" in GV.syn_consts:
            self.trace_params[<int>EL.PostTraceParamIndex.postt_param_decay_stdp] = GV.syn_consts["decay_stdp"]
        if "decay_triplet" in GV.syn_consts:
            self.trace_params[<int> EL.PostTraceParamIndex.postt_param_decay_triplet] = GV.syn_consts["decay_triplet"]
        if "decay_eps" in GV.syn_consts:
            self.trace_params[<int> EL.PostTraceParamIndex.postt_param_decay_eps] = GV.syn_consts["decay_eps"]
        if "decay_ca" in GV.syn_consts:
            self.trace_params[<int> EL.PostTraceParamIndex.postt_param_decay_ca] = GV.syn_consts["decay_ca"]

        self.I_t = [0 for _ in range(self.neu_num)]
        if GV.sim_params["stimulus_dict"]["valid_list"]["current_stimulus"]:
            for lid in range(self.neu_num):
                self.I_t[lid] = GV.sim_params["stimulus_dict"]["current_stimulus"]["I_list"][self.lid_to_gid[lid]]

        self.threshold = [0. for _ in range(self.neu_num)]
        for lid in range(self.neu_num):
            self.threshold[lid] = initial_states[lid]["threshold"]
            self.checkpoint.set_state_chkpt(lid, self.states[lid])

        self.post_traces = np.asarray([[0. for _ in range(<int>EL.PosttIndex.postt_max)] for _ in range(self.neu_num)], dtype=np.dtype("f"))
        self.post_traces_temp = np.asarray([[0. for _ in range(<int>EL.PosttIndex.postt_max)] for _ in range(self.neu_num)], dtype=np.dtype("f"))


    cpdef tuple state_update(self, int lid, int is_rollback):
        cdef int read_pos = 0
        cdef int history_pos = 0

        cdef float delta_g_exc = 0.
        cdef float delta_g_inh = 0.

        cdef float v_t = 0.
        cdef float gE_t = 0.
        cdef float gI_t = 0.
        cdef float yE_t = 0.
        cdef float yI_t = 0.
        cdef float refr = 0.
        cdef float w_t = 0.

        cdef float threshold = 0.

        cdef int typ = self.neu_type[lid]
        cdef int spiked = False

        cdef str neuron_model

        cdef float v_a = 0.
        cdef float v_b = 0.
        cdef float v_c = 0.

        cdef float u_a = 0.
        cdef float u_b = 0.
        cdef float u_c = 0.
        cdef float u_d = 0.

        cdef float decay_gE = 0.
        cdef float decay_gI = 0.
        cdef float decay_v = 0.

        cdef float decay_u = 0.
        cdef float a = 0.
        cdef float b = 0.
        cdef float d = 0.
        cdef float v_w = 0.
        cdef float Delta_T = 0.

        cdef int poisson_id

        self.timestep[lid] += 1

        if GV.leading_timestep < self.timestep[lid]:
            GV.leading_timestep = self.timestep[lid]

        if typ == <int>EL.NeutypeIndex.normal_exc or typ == <int>EL.NeutypeIndex.normal_inh:
            v_t =  self.states[lid][<int>EL.StateIndex.v_t]
            gE_t = self.states[lid][<int>EL.StateIndex.gE_t]
            gI_t = self.states[lid][<int>EL.StateIndex.gI_t]
            yE_t = self.states[lid][<int>EL.StateIndex.yE_t]
            yI_t = self.states[lid][<int>EL.StateIndex.yI_t]
            refr = self.states[lid][<int>EL.StateIndex.refr]
            w_t =  self.states[lid][<int>EL.StateIndex.w_t]

            threshold = self.threshold[lid]

            neuron_model = GV.neu_consts[typ]['model']

            v_a = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_v_a]
            v_b = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_v_b]
            v_c = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_v_c]

            u_a = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_u_a]
            u_b = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_u_b]
            u_c = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_u_c]
            u_d = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_u_d]

            decay_gE = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_decay_gE]
            decay_gI = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_decay_gI]
            decay_v  = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_decay_v]

            decay_u = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_decay_u]
            a       = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_a]
            b       = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_b]
            d       = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_d]
            v_w     = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_v_w]
            Delta_T = self.neu_params[typ][<int>EL.NeuParamIndex.neu_param_Delta_T]

            # checkpointing
            if not is_rollback and ((self.timestep[lid] - (GV.chkpt_timestep[self.ind] + 1)) % GV.sim_params["cur_sync_period"] == 0):
                self.checkpoint.set_state_chkpt(lid, self.states[lid])

            # in-spike buf garbage collecting
            if not is_rollback:
                reset_pos = (self.timestep[lid] + GV.sim_params["max_syn_delay"] + 1) % GV.sim_params["history_width"]
                self.acc_weights[lid][reset_pos][0] = 0
                self.acc_weights[lid][reset_pos][1] = 0

            read_pos = self.timestep[lid] % GV.sim_params["history_width"]
            history_pos = (self.timestep[lid] - (GV.chkpt_timestep[self.ind] + 1)) % GV.sim_params["cur_sync_period"]
            delta_g_exc = self.acc_weights[lid][read_pos][0]
            delta_g_inh = self.acc_weights[lid][read_pos][1]

        # set default
        spiked = False
        if typ == <int>EL.NeutypeIndex.poisson_exc or typ == <int>EL.NeutypeIndex.poisson_inh:
            poisson_id = lid - self.neu_num
            assert(0 <= poisson_id < self.poisson_num)

            spiked = self.poisson_spike[poisson_id]
            self.poisson_spike[poisson_id] = False
        else:
            if "DSRM0" in neuron_model:
                if refr == 0:
                    gE_t += delta_g_exc
                    gI_t += delta_g_inh

                if refr == 0:
                    v_t = decay_v * \
                        (v_t - GV.neu_consts[typ]['E_L']) + \
                        GV.neu_consts[typ]['E_L']
                    v_t += (gE_t - gI_t)
                gE_t *= decay_gE
                gI_t *= decay_gI


                if refr > 0:
                    refr -= 1
                # if the neuron fired a spike
                elif refr == 0 and v_t > threshold:
                    spiked = True
                    v_t = GV.neu_consts[typ]['reset']
                    refr = GV.neu_consts[typ]['r_ar_max'] - 1

            if "Alpha" in neuron_model:
                if refr == 0:
                    yE_t += delta_g_exc
                    yI_t += delta_g_inh

                if refr == 0:
                    #v_t *= decay_v
                    v_t = decay_v * \
                        (v_t - GV.neu_consts[typ]['E_L']) + \
                        GV.neu_consts[typ]['E_L']
                    v_t += (gE_t - gI_t)
                gE_t = decay_gE * gE_t + (1. - decay_gE) * yE_t
                gI_t = decay_gI * gI_t + (1. - decay_gI) * yI_t
                yE_t *= decay_gE
                yI_t *= decay_gI

                if refr > 0:
                    refr -= 1
                elif refr == 0 and v_t > threshold:
                    spiked = True
                    v_t = GV.neu_consts[typ]['reset']
                    refr = GV.neu_consts[typ]['r_ar_max'] - 1

            if "DLIF" in neuron_model:
                gE_t += delta_g_exc
                # gI
                gI_t += delta_g_inh
                #v_t *= decay_v
                v_t = decay_v * \
                    (v_t - GV.neu_consts[typ]['E_L']) + \
                    GV.neu_consts[typ]['E_L']
                v_t += gE_t * (GV.neu_consts[typ]['reversal_vE'] - v_t) \
                    + gI_t * (GV.neu_consts[typ]['reversal_vI'] - v_t)                    

                gE_t *= decay_gE
                gI_t *= decay_gI

                if refr > 0:
                    refr -= 1
                elif refr == 0 and v_t > threshold:
                    spiked = True
                    v_t = GV.neu_consts[typ]['reset']
                    refr = GV.neu_consts[typ]['r_ar_max'] - 1

            if "AdEx" in neuron_model:
                if refr == 0:
                    gE_t += delta_g_exc
                    # gI
                    gI_t += delta_g_inh


                if refr == 0:
                    v_t_temp = v_t
                    v_t = decay_v * \
                        (v_t_temp - GV.neu_consts[typ]['E_L']) + \
                        GV.neu_consts[typ]['E_L']
                    v_t += Delta_T * exp((v_t_temp-threshold)/Delta_T) * b - w_t
                    v_t += gE_t * (GV.neu_consts[typ]['reversal_vE'] - v_t_temp) \
                        + gI_t * (GV.neu_consts[typ]['reversal_vI'] - v_t_temp)                    
                w_t *= decay_u
                w_t += a * b * (v_t - v_w)
                gE_t *= decay_gE
                gI_t *= decay_gI

                if refr > 0:
                    refr -= 1
                    #v_t = GV.neu_consts[typ]['reset']
                # if the neuron fired a spike
                elif refr == 0 and v_t > threshold:
                    spiked = True
                    v_t = GV.neu_consts[typ]['reset']
                    refr = GV.neu_consts[typ]['r_ar_max'] - 1
                    w_t += d

            if "SLIF" in neuron_model:
                v_t *= decay_v
                if refr == 0:
                    v_t += (delta_g_exc - delta_g_inh)

                if refr > 0:
                    refr -= 1
                elif refr == 0 and v_t > threshold:
                    spiked = True
                    v_t -= threshold
                    refr = GV.neu_consts[typ]['r_ar_max'] - 1

            if "Izhikevich" in neuron_model:

                v_t += v_a * v_t * v_t + v_b * v_t + v_c - w_t
                w_t += u_a * (u_b * v_t - w_t)

                if refr == 0:
                    v_t += delta_g_exc
                    v_t -= delta_g_inh

                if refr > 0:
                    refr -= 1
                elif refr == 0 and v_t > threshold:
                    spiked = True
                    v_t = GV.neu_consts[typ]['reset']
                    refr = GV.neu_consts[typ]['r_ar_max'] - 1
                    w_t += u_d


            self.states[lid][<int>EL.StateIndex.v_t] = v_t
            self.states[lid][<int>EL.StateIndex.gE_t] = gE_t
            self.states[lid][<int>EL.StateIndex.gI_t] = gI_t
            self.states[lid][<int>EL.StateIndex.yE_t] = yE_t
            self.states[lid][<int>EL.StateIndex.yI_t] = yI_t
            self.states[lid][<int>EL.StateIndex.w_t] = w_t
            self.states[lid][<int>EL.StateIndex.refr] = refr

            if GV.sim_params["stimulus_dict"]["valid_list"]["current_stimulus"]:
                self.current_stimulus(lid)

        required_action = EL.ActionIndex.no_action

        if typ == <int>EL.NeutypeIndex.poisson_exc or typ == <int>EL.NeutypeIndex.poisson_inh:
            if spiked: required_action = EL.ActionIndex.gen_spike
        else:
            if not is_rollback and spiked:
                required_action = EL.ActionIndex.gen_spike
            else:
                if not self.core.rr_manager.out_spike_history[history_pos][lid] and spiked:
                    required_action = EL.ActionIndex.gen_spike
                elif self.core.rr_manager.out_spike_history[history_pos][lid] and not spiked:
                    required_action = EL.ActionIndex.gen_antispike

            self.core.rr_manager.out_spike_history[history_pos][lid] = spiked

        return required_action, self.timestep[lid]


    cpdef post_trace_update(self, int lid, int is_spiked, int temp):
        cdef float decay_stdp       = self.trace_params[<int>EL.PostTraceParamIndex.postt_param_decay_stdp]
        cdef float decay_triplet    = self.trace_params[<int>EL.PostTraceParamIndex.postt_param_decay_triplet]
        cdef float decay_eps        = self.trace_params[<int>EL.PostTraceParamIndex.postt_param_decay_eps]
        cdef float decay_ca         = self.trace_params[<int>EL.PostTraceParamIndex.postt_param_decay_ca]

        cdef float [:,:] traces
        traces = self.post_traces_temp if temp else self.post_traces

        traces[lid][<int>EL.PosttIndex.k_minus_t] *= decay_stdp
        traces[lid][<int>EL.PosttIndex.k_minus_triplet_t] *= decay_triplet
        traces[lid][<int>EL.PosttIndex.ca_t] *= decay_ca
        # get reset after learning
        traces[lid][<int>EL.PosttIndex.post_eps_t] = traces[lid][<int>EL.PosttIndex.post_eps_t_temp]
        traces[lid][<int>EL.PosttIndex.post_eps_t] *= decay_eps
        traces[lid][<int>EL.PosttIndex.post_eps_t_temp] = traces[lid][<int>EL.PosttIndex.post_eps_t]

        if is_spiked:
            if "SUPPRESS" in GV.syn_consts["rule"]:
                traces[lid][<int>EL.PosttIndex.k_minus_t] += \
                    (1 - traces[lid][<int>EL.PosttIndex.post_eps_t])
            else:
                traces[lid][<int>EL.PosttIndex.k_minus_t] += 1

            traces[lid][<int>EL.PosttIndex.k_minus_triplet_t] += 1
            traces[lid][<int>EL.PosttIndex.ca_t] += 1
            traces[lid][<int>EL.PosttIndex.post_eps_t_temp] = 1.
        if (self.core.neuron_module.lid_to_gid[lid] in GV.debug_target):
            print("trace_update_temp", self.post_traces_temp[lid][<int>EL.PosttIndex.k_minus_t], self.post_traces[lid][<int>EL.PosttIndex.k_minus_t])


    cpdef reset_temp_post_trace(self, int lid):
        cdef int index
        for index in range(<int>EL.PosttIndex.postt_max):
            self.post_traces_temp[lid][index] = self.post_traces[lid][index]


    cpdef poisson_stimulus(self, int lid):
        cdef dict poisson_stimulus = GV.sim_params["stimulus_dict"]["poisson_neurons"]
        cdef float rv
        cdef int poisson_id = lid - self.neu_num
        cdef int gid = self.lid_to_gid[lid]
        cdef int g_poisson_id = gid - GV.sim_params['total_neu_num']

        assert(0 <= poisson_id < self.poisson_num)

        if poisson_stimulus["external"]:
            if len(poisson_stimulus["spikes"][g_poisson_id]) != 0:
                while (self.timestep[lid] + 1) == poisson_stimulus["spikes"][g_poisson_id][0]:
                    self.poisson_spike[poisson_id] = True
                    temp = np.delete(poisson_stimulus["spikes"][g_poisson_id], 0)
                    if len(temp) == 0: break
                    else: poisson_stimulus["spikes"][g_poisson_id] = temp
        else:
            rv = GV.get_random() * GV.sim_params["timestep_per_sec"]
            if g_poisson_id < poisson_stimulus["poisson_neurons"]["npe_sim"]:
                if rv < poisson_stimulus["poisson_stimulus"]["e_rates"]:
                    self.poisson_spike[poisson_id] = True
            else:
                if rv < poisson_stimulus["poisson_stimulus"]["i_rates"]:
                    self.poisson_spike[poisson_id] = True


    cpdef current_stimulus(self, int lid):
        self.states[lid][<int>EL.StateIndex.v_t] += self.I_t[lid]
