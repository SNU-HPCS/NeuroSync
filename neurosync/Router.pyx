import GlobalVars as GV
import KeyDict
import heapq
from collections import deque
import sys

cpdef tuple ind_to_coord(int ind):
    return ind % GV.sim_params["max_core_x_in_total"], ind / GV.sim_params["max_core_x_in_total"]

cdef class Router:

    def __init__(self, int ind, Core.Core core, dict conn_list):
        # Set position
        self.ind = ind
        self.x, self.y = ind_to_coord(self.ind)
        self.x_in_chip = self.x % GV.sim_params["max_core_x_in_chip"]
        self.y_in_chip = self.y % GV.sim_params["max_core_y_in_chip"]
        self.chip_x = self.x // GV.sim_params["max_core_x_in_chip"]
        self.chip_y = self.y // GV.sim_params["max_core_y_in_chip"]

        # Routing
        #self.recv_buf_len = 1 + GV.sim_params["max_sync_period"]
        #self.recv_buf = [deque() for _ in range(self.recv_buf_len)]
        self.recv_buf = [deque() for _ in range(1 + GV.sim_params["max_sync_period"])]
        self.send_buf = []
        self.spike_in_buf = deque()
        self.spike_out_buf = deque()
        self.spike_generator_available_cyc = 0
        self.synapse_events = deque()
        self.sw = GV.NoC.sw[self.x][self.y]
        self.sw.router = self

        self.core_indir = conn_list["core_ind"]
        self.core_dat = conn_list["core_dat"]

        # Synchronization
        self.parent_x = -1
        self.parent_y = -1
        self.children = []
        self.ack_left = 0
        self.num_children = 0
        self.child_ready_count = 0
        self.child_commit_count = 0
        self.sync_role = EL.SyncroleIndex.syncrole_max
        self.recv_commit = False

        # Core
        self.core = core
        self.recovery_stack = None

    # Check if the core is in either state
    # BS: before the core has made synchronization request
    # AS: after the core has made synchronization request
    cpdef is_core_bs(self):
        return (self.core.FSM_state == EL.FsmIndex.neu_comp_state) \
            or (self.core.FSM_state == EL.FsmIndex.neu_comp_end_state) \
            or (self.core.FSM_state == EL.FsmIndex.rollback_state
                and self.core.FSM_prev_state == EL.FsmIndex.neu_comp_state) \
            or (self.core.FSM_state == EL.FsmIndex.rollback_state
                and self.core.FSM_prev_state == EL.FsmIndex.neu_comp_end_state)
        
    cpdef is_core_as(self):
        return (self.core.FSM_state == EL.FsmIndex.sync_wait_state) \
            or (self.core.FSM_state == EL.FsmIndex.rollback_state
                and self.core.FSM_prev_state == EL.FsmIndex.sync_wait_state)
    
    
    # generate spike packet
    cpdef spike_packet_generation(self):
        cdef int src_lid
        cdef int src_pid

        cdef int spiked_timestep

        cdef int core_start_addr
        cdef int core_end_addr
        cdef int addr
        
        cdef int dst_core
        cdef int dst_start_addr
        cdef int dst_end_addr

        cdef int dst_x
        cdef int dst_y
        cdef object pck_type

        # check valid entry exist
        if not self.spike_out_buf or self.spike_out_buf[0][<int>EL.LocalSpikeIndex.local_spike_cyc] > GV.cyc[self.ind]:
            return

        # check spike generator available
        if not GV.cyc[self.ind] >= self.spike_generator_available_cyc:
            return

        # retrieve proper metadata from the spike out buffer
        _, src_lid, spiked_timestep, spike_type, rs_ind = self.spike_out_buf.popleft()

        # set the packet type according to the spike type
        pck_type = EL.PacketIndex.spike
        if spike_type == EL.SpiketypeIndex.anti_spike:
            pck_type = EL.PacketIndex.antispike

        core_start_addr = self.core_indir[src_lid]
        core_end_addr = self.core_indir[src_lid+1]

        for addr in range(core_start_addr, core_end_addr):
            # address of the destination core
            dst_core = self.core_dat[addr][0]

            # retrieve the start and end address @ the destination core
            dst_start_addr = self.core_dat[addr][1]
            dst_end_addr = self.core_dat[addr][2]

            # pid @ destination
            src_pid = self.core_dat[addr][3]
            dst_x, dst_y = ind_to_coord(dst_core)
            
            new_spike_packet = {"pck_type": pck_type,
                                "dst_x": dst_x, "dst_y": dst_y,
                                "src_x": self.x, "src_y": self.y,
                                "src_pid": src_pid,
                                "dst_start_addr": dst_start_addr, "dst_end_addr": dst_end_addr,
                                "neu_type": self.core.neuron_module.neu_type[src_lid],
                                "gen_timestep": spiked_timestep, "rs_ind": rs_ind}  # pop, indir, read_dst

            # for returning ack

            if self.is_core_bs():
                self.ack_left += 1
                assert rs_ind == -1
            elif self.is_core_as():
                assert rs_ind != -1
            else: assert(0 and "Incorrect FSM State")

            
            heapq.heappush(self.send_buf, KeyDict.KeyDict(GV.cyc[self.ind] + (addr - core_start_addr) + 3, new_spike_packet))


        self.spike_generator_available_cyc = GV.cyc[self.ind] + (core_end_addr - core_start_addr) + 1

    cpdef num_packets(self, lid):
        return self.core_indir[lid+1] - self.core_indir[lid]

    cpdef packet_in(self, recv_packet):
        cdef int pos
        if recv_packet["gen_timestep"] < GV.timestep[self.ind]:
            assert recv_packet["pck_type"] == EL.PacketIndex.spike\
                or recv_packet["pck_type"] == EL.PacketIndex.ack\
                or recv_packet["pck_type"] == EL.PacketIndex.antispike
            pos = GV.timestep[self.ind] % (1 + GV.sim_params["cur_sync_period"])
        else:
            pos = recv_packet["gen_timestep"] % (1 + GV.sim_params["cur_sync_period"])

        self.recv_buf[pos].append(recv_packet)

    cpdef packet_reception(self):
        # recv_packet
        cdef int pos

        cdef dict recv_packet
        cdef object pck_type

        cdef int neu_type
        cdef int spiked_timestep
        cdef int src_x
        cdef int src_y
        cdef int src_pid
        cdef int dst_start_addr
        cdef int dst_end_addr
        cdef int src_rs_ind
        cdef int spike_rs_ind

        cdef int is_anti

        cdef int my_rs_ind

        pos = GV.timestep[self.ind] % (1 + GV.sim_params["cur_sync_period"])

        while self.recv_buf[pos]:
            recv_packet = self.recv_buf[pos].popleft()
            pck_type = recv_packet["pck_type"]

            if pck_type == EL.PacketIndex.spike or pck_type == EL.PacketIndex.antispike: # spike or anti-spike packet
                assert recv_packet["gen_timestep"] <= GV.timestep[self.ind]
                neu_type = recv_packet["neu_type"]
                spiked_timestep = recv_packet["gen_timestep"]
                src_x = recv_packet["src_x"]
                src_y = recv_packet["src_y"]
                src_pid = recv_packet["src_pid"]
                dst_start_addr = recv_packet["dst_start_addr"]
                dst_end_addr = recv_packet["dst_end_addr"]
                src_rs_ind = recv_packet["rs_ind"]
                spike_rs_ind = -1

                # BS: before synchronization request
                # the core should send the acknowledgement right away
                if self.is_core_bs():
                    spike_rs_ind = -1
                    new_ack_packet = {"pck_type" : EL.PacketIndex.ack,
                        "dst_x": src_x, "dst_y": src_y,
                        "rs_ind" : src_rs_ind,
                        "gen_timestep": spiked_timestep}
                    heapq.heappush(self.send_buf, KeyDict.KeyDict(GV.cyc[self.ind]+1, new_ack_packet))
         
                # AS: after synchronization request
                # the core should send the acknowledgement only after
                # the recovery stack entry is solved
                elif self.is_core_as():
                    spike_rs_ind = self.recovery_stack.rs_tail
                    self.recovery_stack.allocate_entry(self.core, src_x, src_y, src_rs_ind, spiked_timestep)
                else: 
                    assert(0 and "Incorrect FSM State")

                if recv_packet["pck_type"] == EL.PacketIndex.antispike: is_anti = True
                else: is_anti = False

                remote_spike = [src_pid, dst_start_addr, dst_end_addr, neu_type, spiked_timestep, is_anti, spike_rs_ind]
                self.spike_in_buf.append(remote_spike)

            # sync request
            elif pck_type == EL.PacketIndex.sync_req:
                assert recv_packet["gen_timestep"] == GV.timestep[self.ind]

                self.child_ready_count += 1

            # ack
            elif pck_type == EL.PacketIndex.ack:
                assert recv_packet["gen_timestep"] <= GV.timestep[self.ind]
                assert (not self.core.FSM_state == EL.FsmIndex.commit_state)\
                    and (not self.core.FSM_state == EL.FsmIndex.sync_done_state)

                my_rs_ind = recv_packet["rs_ind"]
                if my_rs_ind == -1:
                    self.ack_left -= 1
                    assert self.ack_left >= 0
                else:
                    self.recovery_stack.stack[my_rs_ind - self.recovery_stack.offset].recv_ack()
                    self.recovery_stack.check_recovery_stack = 1

            # commit
            elif pck_type == EL.PacketIndex.commit: #sync commit
                assert recv_packet["gen_timestep"] == GV.timestep[self.ind]
                self.recv_commit = True

    cpdef send_commit_msg(self):
        # Commit and prepare to advance for the next synchronization period
        if self.child_commit_count == self.num_children:
            self.child_commit_count = 0
            return True
        # Send commit message to the children
        else:
            new_commit_packet = {"pck_type" : EL.PacketIndex.commit,
                    "dst_x" : self.children[self.child_commit_count][0],
                    "dst_y" : self.children[self.child_commit_count][1],
                    "gen_timestep" : GV.timestep[self.ind]}
            heapq.heappush(self.send_buf, KeyDict.KeyDict(GV.cyc[self.ind]+1, new_commit_packet))

            self.child_commit_count += 1
            return False

    cpdef send_sync_msg(self):
        cdef int pos
        cdef dict new_req_packet

        pos = GV.timestep[self.ind] % (1 + GV.sim_params["cur_sync_period"])

        if self.ack_left == 0 \
            and self.child_ready_count == self.num_children \
            and not self.recv_buf[pos] \
            and not self.spike_in_buf \
            and not self.spike_out_buf \
            and not self.synapse_events \
            and self.recovery_stack.empty() \
            and not self.core.rollback_queue:
        
            if self.sync_role == EL.SyncroleIndex.root:
                self.child_ready_count = 0

            else:
                # send request packet to the parent node
                new_req_packet = {"pck_type" : EL.PacketIndex.sync_req,
                    "dst_x": self.parent_x, "dst_y": self.parent_y,
                    "gen_timestep": GV.timestep[self.ind]}
                heapq.heappush(self.send_buf, KeyDict.KeyDict(GV.cyc[self.ind]+1, new_req_packet))

                self.child_ready_count = 0
                assert self.ack_left == 0
                assert self.recovery_stack.empty()

            return True
        else:
            return False

    cpdef check_commit(self):

        # Check recovery stack
        if not self.recovery_stack.empty(): return False

        # If not root (wait for the commit message)
        if self.sync_role == EL.SyncroleIndex.non_root and not self.recv_commit: return False
        assert self.ack_left == 0

        self.recovery_stack.reset_stack()
        self.recv_commit = False

        return True
