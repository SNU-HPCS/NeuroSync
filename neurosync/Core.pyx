cimport cython
import numpy as np
cimport numpy as np

from timeit import default_timer as timer
import GlobalVars as GV
from collections import deque
import copy
import sys


cdef class Core:

    def __init__(self,
                 int ind,
                 dict conn_list,
                 list neu_states):

        self.ind = ind 
        GV.timestep[self.ind] = 0
        self.neu_num = GV.sim_params["neu_per_core"][ind]
        self.poisson_num = GV.sim_params["poisson_per_core"][ind]

        # Core states
        self.FSM_state = EL.FsmIndex.neu_comp_state
        self.FSM_prev_state = EL.FsmIndex.neu_comp_state
        self.next_sync_timestep = GV.sim_params["cur_sync_period"] - 1
        GV.cyc[self.ind] = 0
        self.cur_period = 0

        # Connection
        # (core-to-core)
        self.pid_to_gid = conn_list["pid_to_gid"]
        self.pid_length = len(self.pid_to_gid)
        # (neu-to-neu)
        self.syn_mem_weight = np.asarray(conn_list["syn_mem_weight"], dtype=np.dtype("f"))
        self.syn_mem_others = np.asarray(conn_list["syn_mem_others"], dtype=np.dtype("i"))
        self.syn_mem_debug  = conn_list["syn_mem_debug"]
        self.inverse_table = conn_list["inverse_table"]

        # Router Initialize
        self.router = Router.Router(ind, self, conn_list)

        # Checkpoints and RR Manager
        self.rr_manager = RRManager.RRManager(ind, self.neu_num, self.pid_length, self)
        self.checkpoint = self.rr_manager.checkpoint
        self.recovery_stack = self.rr_manager.recovery_stack
        self.router.recovery_stack = self.recovery_stack

        # Neurons
        self.neuron_module = Neuron.NeuronModule(self, conn_list, neu_states, self.checkpoint)
        self.next_comp_lid = 0

        # Learning
        self.post_syn_learning_cyc = 0
        self.spike_in_history_buf = deque()
        self.learning_module = Learning.LearningModule(self)
        self.weight_recovery_end_cyc = 0

        # Key module to handle rollback and recovery
        self.rollback_queue = deque()
        self.reset_cyc = 0

        # Profiling
        self.prev_time = 0


    # called at each tick
    cpdef core_advance(self):
        cdef int pos
        cdef int is_committed

        # Check if an entry can be removed in the recovery stack
        # Done in parallel with other operations
        self.recovery_stack.garbage_collection()

        # FSM Definition
        # Neuron Comp Stage
        if self.FSM_state == EL.FsmIndex.neu_comp_state:
            # iterate neuron computation
            if self.next_comp_lid < self.neu_num + self.poisson_num:
                self.neuron_computation()
                self.next_comp_lid += 1

            # start to propagate only when the neuron computation finish
            if self.next_comp_lid == self.neu_num + self.poisson_num:
                self.process_synapse_events()
                self.spike_to_synapse_events()

            # prepare to send / receive packet to / from NoC
            self.router.spike_packet_generation()
            self.router.packet_reception()

            # prioritize rollback state
            if self.rollback_queue:
                assert self.next_comp_lid == self.neu_num + self.poisson_num
                self.FSM_prev_state = EL.FsmIndex.neu_comp_state
                self.FSM_state = EL.FsmIndex.rollback_state

            # neuron computation end
            elif self.next_comp_lid == self.neu_num + self.poisson_num:
                # periodic sync
                if GV.timestep[self.ind] == self.next_sync_timestep:
                    if not self.router.spike_out_buf and not self.router.send_buf:
                        self.FSM_state = EL.FsmIndex.neu_comp_end_state

                    else: pass # wait for empty send buf

                # speculative advance
                else:
                    pos = GV.timestep[self.ind] % (1 + GV.sim_params["cur_sync_period"])
                    if not self.router.recv_buf[pos] and not self.router.spike_in_buf \
                    and not self.router.synapse_events:
                        self.next_comp_lid = 0
                        GV.timestep[self.ind] += 1

        # Neuron Comp End Stage
        elif self.FSM_state == EL.FsmIndex.neu_comp_end_state: # until make sync req
            # perform spike propagation and packet-related funcs in parallel
            self.process_synapse_events()
            self.spike_to_synapse_events()
            self.router.spike_packet_generation()
            self.router.packet_reception()

            # prioritize rollback state
            if self.rollback_queue:
                self.FSM_prev_state = EL.FsmIndex.neu_comp_end_state
                self.FSM_state = EL.FsmIndex.rollback_state
            # synch request when there is nothing left to do
            else:
                if self.router.send_sync_msg():
                    self.FSM_state = EL.FsmIndex.sync_wait_state

        # Sync Wait Stage
        elif self.FSM_state == EL.FsmIndex.sync_wait_state:

            # wait for commit message
            self.process_synapse_events()
            self.spike_to_synapse_events()
            self.router.spike_packet_generation()
            self.router.packet_reception()

            # prioritize rollback state
            if self.rollback_queue:
                self.FSM_prev_state = EL.FsmIndex.sync_wait_state
                self.FSM_state = EL.FsmIndex.rollback_state

            # wait until the recovery stack is empty
            else:
                if self.router.check_commit():
                    self.FSM_state = EL.FsmIndex.commit_state
                    if self.router.sync_role == EL.SyncroleIndex.root:
                        cur_time = timer()
                        print("%d / %d ... cyc %d, elapsed %f (s)" % (GV.timestep[self.ind], GV.sim_params["max_timestep"], GV.cyc[self.ind], cur_time - self.prev_time))
                        self.prev_time = cur_time
                        sys.stdout.flush()

                        if GV.timestep[self.ind] >= GV.sim_params["setup_timestep"]:
                            GV.sim_params["prev_sync_period"] = GV.sim_params["cur_sync_period"]
                            GV.sim_params["cur_sync_period"] = GV.sim_params["max_sync_period"]

                        if self.next_sync_timestep + GV.sim_params["cur_sync_period"] >= GV.sim_params["max_timestep"]:
                            GV.sim_params["cur_sync_period"] = 1

        # Rollback Stage
        elif self.FSM_state == EL.FsmIndex.rollback_state:
            if self.rr_manager.rollback_consuming_cyc == 0:
                self.recovery_stack.rollback(self)

            self.router.spike_packet_generation()
            self.router.packet_reception()

            self.rr_manager.rollback_consuming_cyc -= 1

            if self.rr_manager.rollback_consuming_cyc == 0:
                self.recovery_stack.barrier2_end()
                # rollback until the queue is empty
                if not self.rollback_queue:
                    self.FSM_state = self.FSM_prev_state

        # Commit Stage
        elif self.FSM_state == EL.FsmIndex.commit_state:

            self.process_synapse_events()
            self.spike_to_synapse_events()
            self.router.packet_reception()

            # propagate commit messages to the target cores
            if self.router.send_commit_msg():
                assert self.post_syn_learning_cyc == 0
                # reset the checkpoints (save the dirty copies to the original memory)
                self.post_syn_learning_cyc += self.checkpoint.reset_weight_chkpt(self.syn_mem_weight, self.syn_mem_others)
                # flush the deferred post learnings
                self.post_syn_learning_cyc += self.checkpoint.flush_learning(self.neuron_module, self.learning_module)
                # after flushing, perform collective trace update
                self.post_syn_learning_cyc += self.checkpoint.collective_pretrace_update(self.learning_module)

                self.FSM_state = EL.FsmIndex.sync_done_state
                
        # Sync Done Stage
        elif self.FSM_state == EL.FsmIndex.sync_done_state:

            self.process_synapse_events()
            self.spike_to_synapse_events()
            self.router.packet_reception()

            is_safe_to_advance = True

            pos = GV.timestep[self.ind] % (1 + GV.sim_params["cur_sync_period"])
            # advance to the next timestep when
            # 1) the receive & spike_in & accumulation buffer is empty
            # 2) post learning is finished
            if self.router.recv_buf[pos] or \
               self.router.spike_in_buf or \
               self.router.synapse_events or \
               self.post_syn_learning_cyc > 0:
                is_safe_to_advance = False
                self.post_syn_learning_cyc -= 1

            # prepare to advance to the next period
            if is_safe_to_advance:

                # reset the cycle
                if self.reset_cyc == 0:
                    # reset time
                    self.reset_cyc = self.rr_manager.reset(GV.timestep[self.ind])
                    self.spike_in_history_buf = deque()
                    self.post_syn_learning_cyc = 0

                else:
                    self.reset_cyc -= 1
                    # prepare to advance to the next period
                    if not self.reset_cyc:
                        self.FSM_state = EL.FsmIndex.neu_comp_state
                        GV.timestep[self.ind] += 1
                        self.next_sync_timestep += GV.sim_params["cur_sync_period"]
                        self.next_comp_lid = 0
                        self.cur_period += 1

        else: assert(0 and "Invalid FSM State")

        GV.cyc[self.ind] += 1

    
    # perform normal neuron computation (not in rollback stage)
    cpdef neuron_computation(self):
        cdef int timestep
        cdef EL.ActionIndex required_action

        if self.next_comp_lid < self.neu_num:
            required_action, timestep = self.neuron_module.state_update(self.next_comp_lid, False)
        else:
            assert(GV.sim_params["stimulus_dict"]["valid_list"]["poisson_neurons"])
            self.neuron_module.poisson_stimulus(self.next_comp_lid)
            required_action, timestep = self.neuron_module.state_update(self.next_comp_lid, False)

        assert required_action != EL.ActionIndex.gen_antispike
        if required_action == EL.ActionIndex.gen_spike:

            src_gid = self.neuron_module.lid_to_gid[self.next_comp_lid]
            GV.spike_out[src_gid].append(timestep)
            
            self.router.spike_out_buf.append((GV.cyc[self.ind] + 1,
                                              self.next_comp_lid,
                                              timestep,
                                              EL.SpiketypeIndex.norm_spike,
                                              -1))


    # process the synapse events (learning + weight accumulation for each synapse)
    cpdef process_synapse_events(self):
        cdef int addr
        cdef int neu_type
        cdef int spiked_timestep
        cdef int exc_inh
        cdef int rs_ind
        cdef int is_anti
        cdef int replay_weight
        cdef int is_last
        cdef int dst_lid
        cdef int syn_delay
        cdef int enable
        cdef int lastspike_t
        cdef int affecting_timestep
        cdef int accum_pos
        cdef int late_pos

        cdef int [:] others_dat
        cdef float weight
        cdef float effective_weight

        # check if the required computation for the previous weight accumulation is done
        # and the accum_event is valid
        if not self.router.synapse_events or \
                self.router.synapse_events[0][0] > GV.cyc[self.ind] or \
                GV.cyc[self.ind] < self.weight_recovery_end_cyc:
            return

        _, src_pid, addr, neu_type, spiked_timestep, \
        rs_ind, is_anti, replay_weight, is_last = self.router.synapse_events.popleft()

        # allocate a new checkpoint entry (as the weight accumlation should generate a dirty copy)
        weight, others_dat, chkpt_addr = self.checkpoint.allocate_weight_chkpt(addr, self.syn_mem_weight, self.syn_mem_others)

        dst_lid = others_dat[<int>EL.SynmemIndex.dst_lid]
        syn_delay = others_dat[<int>EL.SynmemIndex.delay]
        enable = others_dat[<int>EL.SynmemIndex.learning_en]
        lastspike_t = others_dat[<int>EL.SynmemIndex.lastspike_t]

        affecting_timestep = spiked_timestep + syn_delay
        late_pos = (spiked_timestep - GV.chkpt_timestep[self.ind] - 1) % GV.sim_params["cur_sync_period"]

        if enable:
            if replay_weight:
                # schedule a rollback procedure due to the weight accumulation
                if rs_ind != -1: self.recovery_stack.barrier2_begin(rs_ind)
                self.rollback_queue.append( (src_pid, addr, dst_lid, spiked_timestep,
                                             affecting_timestep, neu_type, weight,
                                             lastspike_t, rs_ind,
                                             enable, is_anti, replay_weight, is_last) )
            else:
                # perform pre_learning + weight accumulation in the normal accumulation procedure
                if is_anti:
                    assert self.checkpoint.src_spike_history[src_pid][late_pos] == False

                    self.accumulate(dst_lid, affecting_timestep, neu_type, is_anti, weight)

                    effective_weight, weight, lastspike_t, _, _ = self.rr_manager.weight_recovery(src_pid, addr, dst_lid, late_pos, hist_mode = 0)

                    others_dat[<int>EL.SynmemIndex.lastspike_t] = lastspike_t
                    others_dat[<int>EL.SynmemIndex.chkpt_addr] = addr

                    self.checkpoint.set_weight_chkpt(chkpt_addr, weight, others_dat)

                else:
                    assert self.checkpoint.src_spike_history[src_pid][late_pos] == True

                    effective_weight, weight, lastspike_t, _, _ = self.rr_manager.weight_recovery(src_pid, addr, dst_lid, late_pos, hist_mode = 0)
                    self.checkpoint.syn_mem_chkpt_weight[chkpt_addr] = weight
                    self.checkpoint.syn_mem_chkpt_others[chkpt_addr] = [dst_lid, syn_delay, enable, lastspike_t, addr]

                    self.accumulate(dst_lid, affecting_timestep, neu_type, is_anti, effective_weight)

                # schedule a rollback procedure only when the affecting timestep is smaller 
                if affecting_timestep <= GV.timestep[self.ind]:
                    if rs_ind != -1: self.recovery_stack.barrier2_begin(rs_ind)
                    self.rollback_queue.append( (src_pid, addr, dst_lid, spiked_timestep,
                                                 affecting_timestep, neu_type, weight,
                                                 lastspike_t, rs_ind,
                                                 enable, is_anti, replay_weight, is_last) )
            self.weight_recovery_end_cyc = GV.cyc[self.ind] + late_pos + 1
        # static synapse
        else:
            if affecting_timestep <= self.neuron_module.timestep[dst_lid]:
                if rs_ind != -1:
                    self.recovery_stack.barrier2_begin(rs_ind)
                self.rollback_queue.append( (src_pid, addr, dst_lid, spiked_timestep,
                affecting_timestep, neu_type, weight, lastspike_t, rs_ind, enable, is_anti, False, is_last) )
            else:
                self.accumulate(dst_lid, affecting_timestep, neu_type, is_anti, weight)

        # release the now_serving only after accumulating the last synaptic weight
        if is_last:
            if self.spike_in_history_buf:
                self.spike_in_history_buf[-1][<int>EL.HistoryBufIndex.history_invalid] = False

            if rs_ind != -1:
                self.recovery_stack.barrier1_end(rs_ind)
                
            

    # convert a spike to synapse events (iterate over the connections)
    cpdef spike_to_synapse_events(self):
        cdef int src_pid
        cdef int dst_start_addr
        cdef int dst_end_addr
        cdef int neu_type
        cdef int spiked_timestep
        cdef int is_anti
        cdef int rs_ind
        cdef int pos
        cdef int ts
        cdef int is_spiked
        cdef int replay_weight
        cdef int dst_lid
        cdef int syn_delay
        cdef float weight
        cdef int accum_pos
        cdef int is_last
        cdef int addr

        cdef list remote_spike
        cdef list spike
        
        if not self.router.spike_in_buf or self.router.synapse_events or self.rollback_queue:
            return
        
        src_pid, dst_start_addr, dst_end_addr, neu_type, \
        spiked_timestep, is_anti, rs_ind = self.router.spike_in_buf.popleft()

        if is_anti:
            # Search for the target spike and remove the spike
            # Assume that anti spike never arrives faster than normal spike
            delete_spike = None
            for spike in self.spike_in_history_buf:
                if spike[EL.HistoryBufIndex.history_pid] == src_pid and \
                        spike[EL.HistoryBufIndex.history_timestep] == spiked_timestep:
                    delete_spike = spike
            assert delete_spike
            self.spike_in_history_buf.remove(delete_spike)
        else:
            spike = [None for _ in range(EL.HistoryBufIndex.history_max)]
            # The spike history is invalidated until the all the synapse events have been processed
            spike[EL.HistoryBufIndex.history_invalid] = True
            spike[EL.HistoryBufIndex.history_timestep] = spiked_timestep
            spike[EL.HistoryBufIndex.history_pid] = src_pid
            spike[EL.HistoryBufIndex.history_type] = neu_type
            spike[EL.HistoryBufIndex.history_start_addr] = dst_start_addr
            spike[EL.HistoryBufIndex.history_end_addr] = dst_end_addr
            spike[EL.HistoryBufIndex.history_anti] = is_anti
            spike[EL.HistoryBufIndex.history_post_spikes] = None
            self.spike_in_history_buf.append(spike)

        if rs_ind != -1:
            self.recovery_stack.barrier1_begin(rs_ind)

        pos = (spiked_timestep - GV.chkpt_timestep[self.ind] - 1) % GV.sim_params["cur_sync_period"]

        if not is_anti:
            assert not self.checkpoint.src_spike_history[src_pid][pos]
            self.checkpoint.src_spike_history[src_pid][pos] = True
        else:
            assert self.checkpoint.src_spike_history[src_pid][pos] == True
            self.checkpoint.src_spike_history[src_pid][pos] = False

        # set whether to replay the weight
        # if there had benn any firing after me 
        # -> I should replay the weight from the checkpoint
        replay_weight = self.checkpoint.check_replay(src_pid, pos)

        # spike fan-out => only the last connection is first targeted
        for addr in range(dst_start_addr, dst_end_addr):
            if addr == dst_end_addr - 1:
                is_last = True
            else:
                is_last = False

            # The latency should increase by 1 for each iteration
            self.router.synapse_events.append((GV.cyc[self.ind] + (addr - dst_start_addr) + 1,
                                               src_pid,
                                               addr,
                                               neu_type,
                                               spiked_timestep,
                                               rs_ind,
                                               is_anti,
                                               replay_weight,
                                               is_last))


    cpdef accumulate(self,
                     int dst_lid,
                     int affecting_timestep,
                     int neu_type,
                     int is_anti,
                     float weight):

        cdef int accum_pos
        cdef int exc_inh = 0

        accum_pos = affecting_timestep % GV.sim_params["history_width"]
        assert(weight >= 0)

        if neu_type == <int>EL.NeutypeIndex.normal_exc:
            exc_inh = 0
        elif neu_type == <int>EL.NeutypeIndex.normal_inh:
            exc_inh = 1
        elif neu_type == <int>EL.NeutypeIndex.poisson_exc:
            exc_inh = 0
        elif neu_type == <int>EL.NeutypeIndex.poisson_inh:
            exc_inh = 1

        assert(neu_type < <int>EL.NeutypeIndex.neutype_max)

        if is_anti: weight *= -1
            
        # Assume that anti never arrives before normal spike
        if self.neuron_module.acc_weights[dst_lid][accum_pos][exc_inh] + weight < 0:
            assert(0)

        self.neuron_module.acc_weights[dst_lid][accum_pos][exc_inh] += weight
