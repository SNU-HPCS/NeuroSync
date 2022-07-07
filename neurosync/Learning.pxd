cimport cython
import numpy as np
cimport numpy as np
ctypedef np.uint8_t uint8

cimport EnumList as EL
cimport Checkpoint
cimport Core

cdef class LearningModule:
    cdef public Core.Core core

    cdef public int pid_length

    cdef public float [:,:] pre_traces
    cdef public float [:] pre_traces_temp
    # Params
    cdef public list trace_params

    cdef public list learn_params

    cpdef int post_learn(self, int lid, int time, int ts, Checkpoint.Checkpoint checkpoint)
    cpdef float post_learn_temp(self, int lid, float weight, int lastspike_t, int time)
    cpdef tuple pre_learning_rule(self, int lid, float weight, int lastspike_t, int time)

    cpdef reset_temp_pre_trace(self, int pid)
    cpdef pre_trace_update(self, int src_pid, int temp, int is_spiked)
    cpdef float post_learning_rule(self, float weight, int lastspike_t, int time, synapse_model, int lid)
