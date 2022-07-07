cimport cython
import numpy as np
cimport numpy as np

cimport EnumList as EL
cimport RRManager
cimport Neuron
cimport Learning
ctypedef np.uint8_t uint8

cdef class Checkpoint:
    cdef public int ind
    cdef public float [:,:] state_chkpt
    cdef public list syn_mem_chkpt_weight
    cdef public list syn_mem_chkpt_others
    cdef public int pre_neu_num
    cdef public list src_spike_history
    cdef public list src_spiked_in_period
    cdef public RRManager.RRManager rr_manager

    cpdef int check_replay(self, int src_pid, int pos)
    cpdef int flush_learning(self, Neuron.NeuronModule neuron_module, Learning.LearningModule learning_module)
    cpdef int collective_pretrace_update(self, Learning.LearningModule learning_module)
    cpdef int reset(self)
    cpdef tuple allocate_weight_chkpt(self, int addr, float [:] syn_mem_weight, int [:,:] syn_mem_others)
    cpdef set_weight_chkpt(self, int chkpt_addr, float new_weight, int [:] new_others_dat)
    cpdef int reset_weight_chkpt(self, float [:] syn_mem_weight, int [:,:] syn_mem_others)
    cpdef float [:] get_state_chkpt(self, int neu_id)
    cpdef set_state_chkpt(self, int neu_id, float [:] state)
