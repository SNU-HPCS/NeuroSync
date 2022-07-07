cimport cython
import numpy as np
cimport numpy as np

cimport EnumList as EL
cimport Checkpoint
cimport Core
cimport Neuron
cimport Learning
ctypedef np.uint8_t uint8

cdef class RRManager:
    cdef public int ind

    cdef public int neu_num
    cdef public int pre_neu_num

    cdef public uint8 [:,:] out_spike_history
    cdef public list recovery_spike_history
    cdef public list recovery_index
    cdef public list recovery_spike_history_temp
    cdef public list recovery_index_temp
    
    cdef public list rollback_trigger_id
    cdef public int rollback_consuming_cyc

    cdef public Checkpoint.Checkpoint checkpoint

    cdef public RecoveryStack recovery_stack

    cdef public Core.Core core

    cpdef get_history(self,
                      int lid)

    cpdef reset(self,
                int timestep)

    cpdef list get_recovery_spike_history(self,
                                          list spike_history,
                                          int wrong_side,
                                          int wrong_pos)

    cpdef int rollback(self)

    cpdef tuple weight_recovery(self,
                          int src_pid,
                          int addr,
                          int dst_lid,
                          int late_pos,
                          int hist_mode)

    cpdef wrong_behavior_recovery(self,
                                  int src_pid,
                                  int dst_lid,
                                  int wrong_pos,
                                  int rs_ind,
                                  int recover_depth)

    cpdef wrong_learning_recovery(self,
                                  int src_pid,
                                  int neu_type,
                                  int addr,
                                  int dst_lid,
                                  int wrong_side,
                                  int wrong_pos,
                                  int rs_ind,
                                  int recover_depth,
                                  int spiked_timestep)

    cpdef tuple save_temp_traces(self,
                                 int post_lid)

    cpdef load_temp_traces(self,
                           int post_lid,
                           tuple traces_save)

cdef class RecoveryStackEntry:
    cdef public Core.Core core
    cdef public int ind
    cdef public int src_x
    cdef public int src_y
    cdef public int src_rs_ind
    cdef public int src_timestep
    cdef public int rs_ack_left
    cdef public int entry_status

    cdef public int barrier1
    cdef public int barrier2

    cpdef init_entry(self,
                     int src_x,
                     int src_y,
                     int src_rs_ind,
                     int src_timestep)
    cpdef process(self)
    cpdef add_ack(self, int add_num)
    cpdef recv_ack(self)
    cpdef send_ack(self)

cdef class RecoveryStack:
    cdef public Core.Core core
    cdef public list stack
    cdef public int offset
    cdef public int rs_head
    cdef public int rs_tail
    cdef public int check_recovery_stack
    cdef public int temp_rb_rs_ind
    cdef public int temp_rb_new_msg_cnt
    cdef public int new_msg_cnt

    cpdef allocate_entry(self,
                         Core.Core core,
                         int src_x,
                         int src_y,
                         int src_rs_ind,
                         int spiked_timestep)

    cpdef garbage_collection(self)

    cpdef rollback(self,
                   Core.Core core)

    cpdef barrier1_begin(self,
                         int rs_ind)

    cpdef barrier1_end(self,
                       int rs_ind)

    cpdef barrier2_begin(self,
                         int rs_ind)

    cpdef barrier2_end(self)

    cpdef reset_stack(self)

    cpdef int empty(self)

    cpdef int size(self)
