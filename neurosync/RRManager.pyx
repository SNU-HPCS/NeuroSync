cimport cython
import numpy as np
cimport numpy as np

import GlobalVars as GV
import KeyDict
import heapq
import copy

import sys

cdef class RRManager:
    def __init__(self,
                 int ind,
                 int neu_num,
                 int pre_neu_num,
                 Core.Core core):

        self.ind = ind
        self.neu_num = neu_num
        self.pre_neu_num = pre_neu_num

        self.out_spike_history = np.array([[False for _ in range(self.neu_num)]
                                           for _ in range(GV.sim_params["max_sync_period"])],
                                          dtype=np.dtype("bool"))

        GV.chkpt_timestep[self.ind] = GV.leading_timestep - 1
        self.rollback_consuming_cyc = 0
        self.checkpoint = Checkpoint.Checkpoint(ind, neu_num, pre_neu_num, self)
        self.recovery_stack = RecoveryStack(core)
        self.core = core

        self.rollback_trigger_id = []

        self.recovery_spike_history = []
        self.recovery_index = [0 for _ in range(self.pre_neu_num)]

    cpdef get_history(self,
                      int lid):

        cdef int i
        return [self.out_spike_history[i][lid] for i in range(GV.sim_params["cur_sync_period"])]

    cpdef reset(self,
                int timestep):

        reset_cyc1 = self.checkpoint.reset()
        reset_cyc2 = self.neu_num
        self.out_spike_history = np.array([[False for _ in range(self.neu_num)]
                                           for _ in range(GV.sim_params["max_sync_period"])],
                                           dtype=np.dtype("bool"))

        GV.chkpt_timestep[self.ind] = timestep
        return max(reset_cyc1, reset_cyc2)

    cpdef list get_recovery_spike_history(self,
                                          list spike_history,
                                          int wrong_side,
                                          int wrong_pos):

        recovery_spike_history = []
        for i in range(GV.sim_params["cur_sync_period"]):
            if i == wrong_pos and wrong_side == 1:
                if spike_history[i] == 1:
                    recovery_spike_history.append(0)
                else:
                    recovery_spike_history.append(1)
            else:
                recovery_spike_history.append(spike_history[i])
        return recovery_spike_history

    cpdef int rollback(self):

        cdef int rollback_lid
        cdef int rollback_gid
        cdef int affecting_timestep
        cdef int neu_type
        cdef float weight
        cdef int cur_timestep
        cdef int read_pos
        cdef int i
        cdef int neu_timestep
        cdef int rs_ind
        cdef int is_last
        cdef int delta_t

        cdef int src_pid

        cdef int ts

        if not self.core.rollback_queue: return -1

        self.recovery_spike_history = []
        self.recovery_index = [0 for _ in range(self.pre_neu_num)]

        # 0 cyc
        # retrieve the target neuron
        self.rollback_consuming_cyc = 1
        src_pid, addr, rollback_lid, spiked_timestep, affecting_timestep, \
        neu_type, weight, lastspike_t, rs_ind, is_plastic, is_anti, replay_weight, is_last  = self.core.rollback_queue.popleft()
        neuron_module = self.core.neuron_module
        rollback_gid = neuron_module.lid_to_gid[rollback_lid]

        # 1 cyc
        # retrieve the checkpoint states
        self.rollback_consuming_cyc += 1
        delta_t = neuron_module.timestep[rollback_lid] - GV.chkpt_timestep[self.ind]
        neuron_module.timestep[rollback_lid] = GV.chkpt_timestep[self.ind]

        self.core.neuron_module.states[rollback_lid] = self.checkpoint.get_state_chkpt(rollback_lid)

        assert self.core.recovery_stack.new_msg_cnt == 0

        #          late_pos     now_pos
        # |  ROLLBACK  |  RECOVER  |
        late_pos = (spiked_timestep - GV.chkpt_timestep[self.ind] - 1) % GV.sim_params["cur_sync_period"]
        now_pos = (GV.timestep[self.ind] - GV.chkpt_timestep[self.ind] - 1) % GV.sim_params["cur_sync_period"]
            
        # Check if not removed
        if is_last: 
            if not is_anti:
                self.core.spike_in_history_buf[-1][EL.HistoryBufIndex.history_invalid] = True

        # set the rollback trigger id
        self.rollback_trigger_id = [src_pid]

        # Plasticity
        if is_plastic:
            if replay_weight:
                # This process is required because only the checkpoint for the first weight is stored
                # the checkpoint is only for the first 

                recovery_spike_history = self.get_recovery_spike_history(self.get_history(rollback_lid), 0, late_pos)
                self.recovery_spike_history.append(recovery_spike_history)

                self.wrong_learning_recovery(src_pid, neu_type, addr, rollback_lid, 0, late_pos, rs_ind, 0, spiked_timestep)

            else:

                # neuron rollback to position
                for ts in range(late_pos+1):
                    required_action, neu_timestep = neuron_module.state_update(rollback_lid, True)
                    assert required_action == EL.ActionIndex.no_action
                assert neuron_module.timestep[rollback_lid] == GV.chkpt_timestep[self.ind] + late_pos + 1

                self.rollback_consuming_cyc += late_pos+1

                # replay from T_late
                post_weight = weight
                post_lastspike_t = lastspike_t
                for ts in range(late_pos+1, now_pos+1):
                    required_action, neu_timestep = neuron_module.state_update(rollback_lid, True)

                    # behavior not changed
                    if required_action == EL.ActionIndex.no_action:
                        self.rollback_consuming_cyc += 1

                    # behavior changed: not-spiked => spiked
                        

                    elif required_action == EL.ActionIndex.gen_spike:
                        self.rollback_consuming_cyc += 1

                        GV.spike_out[rollback_gid].append(neu_timestep)

                        # Send
                        self.core.recovery_stack.new_msg_cnt += self.core.router.num_packets(rollback_lid)

                        self.core.router.spike_out_buf.append((GV.cyc[self.ind] + 2 + self.rollback_consuming_cyc, rollback_lid, neu_timestep, EL.SpiketypeIndex.norm_spike, rs_ind))

                        traces_save = self.save_temp_traces(rollback_lid)
                        
                        recovery_spike_history = self.get_recovery_spike_history(self.get_history(rollback_lid), 1, ts)
                        self.recovery_spike_history.append(recovery_spike_history)

                        self.wrong_behavior_recovery(src_pid, rollback_lid, ts, rs_ind, 0)
                        self.load_temp_traces(rollback_lid, traces_save)

                    # behavior changed: spiked => not-spiked
                    elif required_action == EL.ActionIndex.gen_antispike:
                        self.rollback_consuming_cyc += 1

                        GV.spike_out[rollback_gid].append(-1 * neu_timestep)
                        self.core.recovery_stack.new_msg_cnt += self.core.router.num_packets(rollback_lid)

                        self.core.router.spike_out_buf.append((GV.cyc[self.ind] + 2 + self.rollback_consuming_cyc, rollback_lid, neu_timestep, EL.SpiketypeIndex.anti_spike, rs_ind))

                        traces_save = self.save_temp_traces(rollback_lid)

                        recovery_spike_history = self.get_recovery_spike_history(self.get_history(rollback_lid), 1, ts)
                        self.recovery_spike_history.append(recovery_spike_history)

                        self.wrong_behavior_recovery(src_pid, rollback_lid, ts, rs_ind, 0)
                        self.load_temp_traces(rollback_lid, traces_save)


                    pre_spiked = self.checkpoint.src_spike_history[src_pid][ts]
                    post_spiked = self.out_spike_history[ts][rollback_lid]

                    # post-trace update
                    neuron_module.post_trace_update(rollback_lid, post_spiked, temp = 1)

                    # pre-trace update
                    self.core.learning_module.pre_trace_update(src_pid, 1,pre_spiked)

                    # pre-learning
                    if pre_spiked:
                        raise NotImplementedError

                    # post-learning
                    if post_spiked:
                        post_weight = self.core.learning_module.post_learn_temp(rollback_lid,
                                                                                weight,
                                                                                lastspike_t,
                                                                                GV.chkpt_timestep[self.ind] + ts + 1)

                        post_lastspike_t = GV.chkpt_timestep[self.ind] + ts + 1

        # No plasticity
        else:
            if is_anti:
                assert self.checkpoint.src_spike_history[src_pid][late_pos] == False
            else:
                assert self.checkpoint.src_spike_history[src_pid][late_pos] == True

            self.core.accumulate(rollback_lid, affecting_timestep, neu_type, is_anti, weight)


            # replay from T_chkpt
            for ts in range(delta_t):
                required_action, neu_timestep = neuron_module.state_update(rollback_lid, True)

                # behavior not changed
                if required_action == EL.ActionIndex.no_action:
                    self.rollback_consuming_cyc += 1

                # behavior changed: not-spiked => spiked
                elif required_action == EL.ActionIndex.gen_spike:
                    self.rollback_consuming_cyc += 1

                    GV.spike_out[rollback_gid].append(neu_timestep)
                    self.core.recovery_stack.new_msg_cnt += self.core.router.num_packets(rollback_lid)

                    self.core.router.spike_out_buf.append((GV.cyc[self.ind] + 2 + self.rollback_consuming_cyc, rollback_lid, neu_timestep, EL.SpiketypeIndex.norm_spike, rs_ind))

                    recovery_spike_history = self.get_recovery_spike_history(self.get_history(rollback_lid), 1, ts)
                    self.recovery_spike_history.append(recovery_spike_history)

                    self.wrong_behavior_recovery(src_pid, rollback_lid, ts, rs_ind, 0)

                # behavior changed: spiked => not-spiked
                elif required_action == EL.ActionIndex.gen_antispike:
                    self.rollback_consuming_cyc += 1

                    GV.spike_out[rollback_gid].append(-1 * neu_timestep)
                    self.core.recovery_stack.new_msg_cnt += self.core.router.num_packets(rollback_lid)

                    self.core.router.spike_out_buf.append((GV.cyc[self.ind] + 2 + self.rollback_consuming_cyc, rollback_lid, neu_timestep, EL.SpiketypeIndex.anti_spike, rs_ind))

                    recovery_spike_history = self.get_recovery_spike_history(self.get_history(rollback_lid), 1, ts)
                    self.recovery_spike_history.append(recovery_spike_history)

                    self.wrong_behavior_recovery(src_pid, rollback_lid, ts, rs_ind, 0)

        if is_last: 
            if not is_anti:
                self.core.spike_in_history_buf[-1][EL.HistoryBufIndex.history_invalid] = False

        return rs_ind


    # recover the weight to the late_pos using the checkpoint
    cpdef tuple weight_recovery(self,
                          int src_pid,
                          int addr,
                          int dst_lid,
                          int late_pos,
                          int hist_mode):

        cdef float weight
        cdef int lastspike_t
        cdef int is_pre_spiked
        cdef int is_post_spiked
        cdef int ts
        cdef float new_weight
        cdef float effective_weight
        cdef float post_weight
        cdef int post_lastspike_t

        # destination neuron
        neuron_module = self.core.neuron_module

        # checkpointed weight
        weight = self.core.syn_mem_weight[addr]
        lastspike_t = self.core.syn_mem_others[addr][<int>EL.SynmemIndex.lastspike_t]

        # reset temp pre/post trace
        neuron_module.reset_temp_post_trace(dst_lid)
        self.core.learning_module.reset_temp_pre_trace(src_pid)

        # weight recovery
        effective_weight = weight
        post_weight = weight
        post_lastspike_t = lastspike_t

        for ts in range(late_pos+1):
            pre_spiked = self.checkpoint.src_spike_history[src_pid][ts]
            if hist_mode == 0:
                post_spiked = self.out_spike_history[ts][dst_lid]
            else:
                # retrieve the out spike history (the latest one)
                post_spiked = self.recovery_spike_history[self.recovery_index[src_pid]][ts]

            # post-trace update
            neuron_module.post_trace_update(dst_lid, post_spiked, temp = 1)

            # pre-trace update
            self.core.learning_module.pre_trace_update(src_pid, 1, pre_spiked)

            # pre-learning
            if pre_spiked:
                # lazy post learning
                weight = post_weight
                lastspike_t = post_lastspike_t

                new_weight, effective_weight = self.core.learning_module.pre_learning_rule(dst_lid,
                                                                                           weight,
                                                                                           lastspike_t,
                                                                                           GV.chkpt_timestep[self.ind] + ts + 1)

                weight = new_weight
                post_weight = new_weight

            # post-learning
            if post_spiked:
                post_weight = self.core.learning_module.post_learn_temp(dst_lid,
                                                                        weight,
                                                                        lastspike_t,
                                                                        GV.chkpt_timestep[self.ind] + ts + 1)

                post_lastspike_t = GV.chkpt_timestep[self.ind] + ts + 1
        

        return effective_weight, weight, lastspike_t, post_weight, post_lastspike_t

    # This is called recursively
    cpdef wrong_behavior_recovery(self,
                                  int src_pid,
                                  int dst_lid,
                                  int wrong_pos,
                                  int rs_ind,
                                  int recover_depth):

        cdef float weight
        cdef int lastspike_t
        cdef int now_serving
        cdef int spiked_pos
        cdef int dst_start_addr
        cdef int dst_end_addr
        cdef int addr
        cdef int ts
        cdef int neu_timestep

        # destination neuron
        neuron_module = self.core.neuron_module
        
        recover_depth += 1

        # search the in-spike history buffer to find pre-neurons that spiked after T_wrong
        for spike in reversed(self.core.spike_in_history_buf):
            history_invalid = spike[EL.HistoryBufIndex.history_invalid]
            spiked_pos = (spike[EL.HistoryBufIndex.history_timestep] - GV.chkpt_timestep[self.ind] - 1) % GV.sim_params["cur_sync_period"]
            pre_pid = spike[EL.HistoryBufIndex.history_pid]

            # spiked before T_wrong
            #if spiked_pos < wrong_pos or history_invalid or pre_pid in self.rollback_trigger_id:
            if history_invalid or pre_pid in self.rollback_trigger_id or (spiked_pos < wrong_pos and sum(self.checkpoint.src_spike_history[pre_pid][wrong_pos:]) <= 0):
                self.rollback_consuming_cyc += len(self.rollback_trigger_id)

            # spiked after T_wrong
            else:
                neu_type = spike[EL.HistoryBufIndex.history_type]
                dst_start_addr = spike[EL.HistoryBufIndex.history_start_addr]
                dst_end_addr = spike[EL.HistoryBufIndex.history_end_addr]

                # search the synapse buffer
                # list of synapses to perform wrong_learning_recovery
                for addr in range(dst_start_addr, dst_end_addr):
                    self.rollback_consuming_cyc += 1

                    # retrieve the connection for the given spike (only a single)
                    if self.core.syn_mem_others[addr][<int>EL.SynmemIndex.dst_lid] == dst_lid:
                        # plastic
                        if self.core.syn_mem_others[addr][<int>EL.SynmemIndex.learning_en]:
                            spike[EL.HistoryBufIndex.history_invalid] = True
                            self.rollback_trigger_id.append(pre_pid)
                            # retrain the synapse
                            
                            self.wrong_learning_recovery(pre_pid, neu_type, addr, dst_lid, 1, wrong_pos, rs_ind, recover_depth, spike[EL.HistoryBufIndex.history_timestep])
                            
                            # set the rollback depth for the given lid
                            neuron_module.timestep[dst_lid] = GV.chkpt_timestep[self.ind]
                            
                            neuron_module.states[dst_lid] = self.checkpoint.get_state_chkpt(dst_lid)

                            for ts in range(wrong_pos + 1):
                                required_action, neu_timestep = neuron_module.state_update(dst_lid, True)
                                assert required_action == EL.ActionIndex.no_action
                            assert neuron_module.timestep[dst_lid] == GV.chkpt_timestep[self.ind] + wrong_pos + 1

                            spike[EL.HistoryBufIndex.history_invalid] = False
                            self.rollback_trigger_id.remove(pre_pid)
                        break


    cpdef wrong_learning_recovery(self,
                                  int src_pid,
                                  int neu_type,
                                  int addr,
                                  int dst_lid,
                                  int wrong_side,
                                  int wrong_pos,
                                  int rs_ind,
                                  int recover_depth,
                                  int spiked_timestep):

        cdef int now_pos
        cdef float weight
        cdef int syn_delay
        cdef int pre_spiked
        cdef int post_spiked
        cdef int ts
        cdef float new_weight
        cdef float effective_weight
        cdef float post_weight
        cdef int lastspike_t
        cdef int post_lastspike_t
        cdef int affecting_timestep
        cdef int dst_gid

        # destination neuron
        neuron_module = self.core.neuron_module
        dst_gid = neuron_module.lid_to_gid[dst_lid]

        recover_depth += 1

        # connection
        syn_delay = self.core.syn_mem_others[addr][<int>EL.SynmemIndex.delay]

        # current timestep (T_now)
        now_pos = (GV.timestep[self.ind] - GV.chkpt_timestep[self.ind] - 1) % GV.sim_params["cur_sync_period"]

        ##### first iteration (recover the accumulated weight) #####
        self.rollback_consuming_cyc += now_pos+1

        # revert to the mis-speculated state
        # pre-neuron wrong

        if wrong_side == 0:
            self.checkpoint.src_spike_history[src_pid][wrong_pos] ^= True

        # weight recovery to T_wrong-1
        spiked_pos = (spiked_timestep - GV.chkpt_timestep[self.ind] - 1) % GV.sim_params["cur_sync_period"]
        restart_pos = min(spiked_pos, wrong_pos)

        effective_weight, weight, lastspike_t, post_weight, post_lastspike_t = self.weight_recovery(src_pid, 
                                                                                                    addr,
                                                                                                    dst_lid,
                                                                                                    restart_pos-1,
                                                                                                    hist_mode = 1)

        # cancel bad weight accumulations from T_wrong
        #post_weight = weight
        #post_lastspike_t = lastspike_t

        # recover the state
        #for ts in range(wrong_pos, now_pos+1):
        for ts in range(restart_pos, now_pos+1):
            pre_spiked = self.checkpoint.src_spike_history[src_pid][ts]
            post_spiked = self.recovery_spike_history[self.recovery_index[src_pid]][ts]

            # post-trace update
            neuron_module.post_trace_update(dst_lid, post_spiked, temp = 1)

            # pre-trace update
            self.core.learning_module.pre_trace_update(src_pid, 1, pre_spiked)

            # pre-learning
            if pre_spiked:
                # lazy learning
                weight = post_weight
                lastspike_t = post_lastspike_t

                new_weight, effective_weight = self.core.learning_module.pre_learning_rule(dst_lid,
                                                                                           weight,
                                                                                           lastspike_t,
                                                                                           GV.chkpt_timestep[self.ind] + ts + 1)

                weight = new_weight
                post_weight = new_weight

            # post-learning
            if post_spiked:
                post_weight = self.core.learning_module.post_learn_temp(dst_lid,
                                                                        weight,
                                                                        lastspike_t,
                                                                        GV.chkpt_timestep[self.ind] + ts + 1)
                post_lastspike_t = GV.chkpt_timestep[self.ind] + ts + 1

            # cancel bad weight accumulation
            if pre_spiked:
                affecting_timestep = GV.chkpt_timestep[self.ind] + ts + 1 + syn_delay

                self.core.accumulate(dst_lid, affecting_timestep, neu_type, True, effective_weight)


        ##### second iteration #####
        # return to the correct state
        # pre-neuron wrong

        if wrong_side == 0:
            self.checkpoint.src_spike_history[src_pid][wrong_pos] ^= True

        # rollback and recover the neuron state to T_wrong-1
        neuron_module.timestep[dst_lid] = GV.chkpt_timestep[self.ind]

        neuron_module.states[dst_lid] = self.checkpoint.get_state_chkpt(dst_lid)

        for ts in range(restart_pos):
            required_action, neu_timestep = neuron_module.state_update(dst_lid, True)
            assert required_action == EL.ActionIndex.no_action
        assert neuron_module.timestep[dst_lid] == GV.chkpt_timestep[self.ind] + restart_pos

        self.rollback_consuming_cyc += restart_pos

        # weight recovery to T_wrong-1
        # recover the weight using spiking activity
        effective_weight, weight, lastspike_t, post_weight, post_lastspike_t = self.weight_recovery(src_pid,
                                                                                                    addr,
                                                                                                    dst_lid,
                                                                                                    restart_pos-1,
                                                                                                    hist_mode = 0)

        # accumulate newly learned weight from T_wrong
        for ts in range(restart_pos, now_pos+1):
            # replay from T_wrong
            required_action, neu_timestep = neuron_module.state_update(dst_lid, True)
            
            # behavior not changed
            self.rollback_consuming_cyc += 1

            # behavior changed: not-spiked => spiked
            if ts >= wrong_pos:
                if required_action == EL.ActionIndex.gen_spike:
                    GV.spike_out[dst_gid].append(neu_timestep)
                    self.core.recovery_stack.new_msg_cnt += self.core.router.num_packets(dst_lid)

                    self.core.router.spike_out_buf.append((GV.cyc[self.ind] + 1 + self.rollback_consuming_cyc, dst_lid, neu_timestep, EL.SpiketypeIndex.norm_spike, rs_ind))

                    new_wrong_pos = (neu_timestep - GV.chkpt_timestep[self.ind] - 1) % GV.sim_params["cur_sync_period"]

                    traces_save = self.save_temp_traces(dst_lid)

                    self.wrong_behavior_recovery(src_pid, dst_lid, new_wrong_pos, rs_ind, recover_depth)
                    self.load_temp_traces(dst_lid, traces_save)

                # behavior changed: spiked => not-spiked
                elif required_action == EL.ActionIndex.gen_antispike:
                    GV.spike_out[dst_gid].append(-1 * neu_timestep)
                    self.core.recovery_stack.new_msg_cnt += self.core.router.num_packets(dst_lid)

                    self.core.router.spike_out_buf.append((GV.cyc[self.ind] + 1 + self.rollback_consuming_cyc, dst_lid, neu_timestep, EL.SpiketypeIndex.anti_spike, rs_ind))

                    new_wrong_pos = (neu_timestep - GV.chkpt_timestep[self.ind] - 1) % GV.sim_params["cur_sync_period"]

                    traces_save = self.save_temp_traces(dst_lid)

                    self.wrong_behavior_recovery(src_pid, dst_lid, new_wrong_pos, rs_ind, recover_depth)
                    self.load_temp_traces(dst_lid, traces_save)

            pre_spiked = self.checkpoint.src_spike_history[src_pid][ts]
            post_spiked = self.out_spike_history[ts][dst_lid]

            # post-trace update
            neuron_module.post_trace_update(dst_lid, post_spiked, temp = 1)

            # pre-trace update
            self.core.learning_module.pre_trace_update(src_pid, 1, pre_spiked)

            # pre-learning
            if pre_spiked:
                # lazy post learning
                weight = post_weight
                lastspike_t = post_lastspike_t

                new_weight, effective_weight = self.core.learning_module.pre_learning_rule(dst_lid,
                                                                                           weight,
                                                                                           lastspike_t,
                                                                                           GV.chkpt_timestep[self.ind] + ts + 1)

                weight = new_weight
                post_weight = new_weight

            # post-learning
            if post_spiked:
                post_weight = self.core.learning_module.post_learn_temp(dst_lid,
                                                                        weight,
                                                                        lastspike_t,
                                                                        GV.chkpt_timestep[self.ind] + ts + 1)

                post_lastspike_t = GV.chkpt_timestep[self.ind] + ts + 1

            # accumulate newly learned weight
            if pre_spiked:
                affecting_timestep = GV.chkpt_timestep[self.ind] + ts + 1 + syn_delay
                self.core.accumulate(dst_lid, affecting_timestep, neu_type, False, effective_weight)


        # there should be a dirty copy! (unless there should be no weight recovery)
        chkpt_addr = self.core.syn_mem_others[addr][<int>EL.SynmemIndex.chkpt_addr]
        assert not chkpt_addr == -1
        
        # 
        assert self.core.syn_mem_others[addr][<int>EL.SynmemIndex.dst_lid] == self.core.checkpoint.syn_mem_chkpt_others[chkpt_addr][<int>EL.SynmemIndex.dst_lid] == dst_lid
        assert self.core.syn_mem_others[addr][<int>EL.SynmemIndex.learning_en] == self.core.checkpoint.syn_mem_chkpt_others[chkpt_addr][<int>EL.SynmemIndex.learning_en] == True

        # write back the newly learned weight to the checkpoint
        self.checkpoint.syn_mem_chkpt_weight[chkpt_addr] = weight
        self.checkpoint.syn_mem_chkpt_others[chkpt_addr] = [dst_lid, syn_delay, True, lastspike_t, addr]
        assert(self.core.syn_mem_others[addr][<int>EL.SynmemIndex.dst_lid] == dst_lid)

        recovery_spike_history = self.get_recovery_spike_history(self.get_history(dst_lid), 0, wrong_pos)

        self.recovery_index[src_pid] = len(self.recovery_spike_history)
        self.recovery_spike_history.append(recovery_spike_history)


    cpdef tuple save_temp_traces(self, int post_lid):

        cdef list post_traces_save = []
        cdef list pre_traces_save = []

        # post neuron
        for index in range(<int>EL.PosttIndex.postt_max):
            post_traces_save.append(self.core.neuron_module.post_traces_temp[post_lid][index])
        
        for index in range(<int>EL.PretIndex.pret_max):
            pre_traces_save.append(self.core.learning_module.pre_traces_temp[index])


        return post_traces_save, pre_traces_save


    cpdef load_temp_traces(self,
                           int post_lid,
                           tuple traces_save):

        # load saved temporal trace values
        for index in range(<int>EL.PosttIndex.postt_max):
            self.core.neuron_module.post_traces_temp[post_lid][index] = traces_save[0][index]

        for index in range(<int>EL.PretIndex.pret_max):
            self.core.learning_module.pre_traces_temp[index] = traces_save[1][index]

cdef class RecoveryStackEntry:

    def __init__(self, core):

        self.core = core
        self.ind = core.ind
        self.entry_status = EL.RecoveryStackIndex.stack_invalid

        self.barrier1 = 0
        self.barrier2 = 0

        self.rs_ack_left = 0
        self.src_x = -1
        self.src_y = -1
        self.src_rs_ind = -1
        self.src_timestep = -1

    cpdef init_entry(self,
                     int src_x,
                     int src_y,
                     int src_rs_ind,
                     int src_timestep):

        assert self.entry_status == EL.RecoveryStackIndex.stack_invalid

        self.rs_ack_left = 0
        self.src_x = src_x
        self.src_y = src_y
        self.src_rs_ind = src_rs_ind
        self.src_timestep = src_timestep
        self.entry_status = EL.RecoveryStackIndex.stack_valid
        
    cpdef process(self):

        assert self.rs_ack_left == 0
        self.send_ack()
        self.entry_status = EL.RecoveryStackIndex.stack_retired

    cpdef add_ack(self, int add_num):

        assert self.rs_ack_left >= 0
        self.rs_ack_left += add_num

    cpdef recv_ack(self):

        assert self.rs_ack_left > 0
        self.rs_ack_left -= 1

        if self.rs_ack_left == 0:
            self.process()

    cpdef send_ack(self):

        cdef dict ack_packet = {"pck_type" : EL.PacketIndex.ack,
            "dst_x" : self.src_x,
            "dst_y" : self.src_y,
            "rs_ind" : self.src_rs_ind,
            "gen_timestep" : self.src_timestep,
            "gen_cyc" : GV.cyc[self.ind] + 2
        }
        heapq.heappush(self.core.router.send_buf, KeyDict.KeyDict(GV.cyc[self.ind] + 2, ack_packet))


cdef class RecoveryStack:

    def __init__(self, Core.Core core):
        # initialize only a portion at once (for faster emulation)
        self.stack = [RecoveryStackEntry(core) for _ in range(GV.STACK_EXTENSION_LEN)]
        self.offset = 0
        self.rs_head = 0
        self.rs_tail = 0

        self.check_recovery_stack = 0
        self.temp_rb_rs_ind = -1
        self.temp_rb_new_msg_cnt = 0
        self.new_msg_cnt = 0

        self.core = core

    cpdef allocate_entry(self,
                         Core.Core core,
                         int src_x,
                         int src_y,
                         int src_rs_ind,
                         int spiked_timestep):

        self.stack[self.rs_tail - self.offset].init_entry(src_x, src_y, src_rs_ind, spiked_timestep)
        self.rs_tail += 1

        # initialize another portion
        if self.rs_tail % GV.STACK_EXTENSION_LEN == 0:
            self.stack += [RecoveryStackEntry(self.core) for _ in range(GV.STACK_EXTENSION_LEN)]

            # Profile the recovery stack length
            if GV.recovery_length < len(self.stack):
                GV.recovery_length = len(self.stack)


    cpdef garbage_collection(self):

        if self.stack[self.rs_head - self.offset].entry_status == EL.RecoveryStackIndex.stack_retired:
            self.stack[self.rs_head - self.offset].entry_status = EL.RecoveryStackIndex.stack_invalid
            self.rs_head += 1
            if self.rs_head % GV.STACK_EXTENSION_LEN == 0:
                del self.stack[:GV.STACK_EXTENSION_LEN]
                self.offset += GV.STACK_EXTENSION_LEN
	
    cpdef rollback(self,
                   Core.Core core):

        # perform rollback
        self.temp_rb_rs_ind = core.rr_manager.rollback()
        self.temp_rb_new_msg_cnt = self.new_msg_cnt
        self.new_msg_cnt = 0

        if self.temp_rb_rs_ind != -1:
            self.stack[self.temp_rb_rs_ind - self.offset].add_ack(self.temp_rb_new_msg_cnt)

    cpdef barrier1_begin(self,
                         int rs_ind):

        self.stack[rs_ind - self.offset].add_ack(1)
        self.stack[rs_ind - self.offset].barrier1 += 1

    cpdef barrier1_end(self,
                       int rs_ind):

        self.stack[rs_ind - self.offset].recv_ack()
        self.stack[rs_ind - self.offset].barrier1 -= 1
        self.check_recovery_stack = 1

    cpdef barrier2_begin(self,
                         int rs_ind):

        self.stack[rs_ind - self.offset].add_ack(1)
        self.stack[rs_ind - self.offset].barrier2 += 1

    cpdef barrier2_end(self):

        if self.temp_rb_rs_ind != -1:
            self.stack[self.temp_rb_rs_ind - self.offset].recv_ack() # RS barrier-2 end
            self.stack[self.temp_rb_rs_ind - self.offset].barrier2 -= 1
            self.check_recovery_stack = 1

    cpdef reset_stack(self):
        self.stack = [RecoveryStackEntry(self.core) for _ in range(GV.STACK_EXTENSION_LEN)]
        self.offset = 0
        self.rs_head = 0
        self.rs_tail = 0

    cpdef int empty(self):

        return self.rs_tail == self.rs_head

    cpdef int size(self):
        return self.rs_tail - self.rs_head
