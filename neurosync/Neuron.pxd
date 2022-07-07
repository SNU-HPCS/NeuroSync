cimport EnumList as EL
cimport Core
cimport Checkpoint

cdef class NeuronModule:
    cdef public Core.Core core
    cdef public int ind

    cdef public int neu_num
    cdef public int poisson_num

    cdef public list lid_to_gid

    cdef public list timestep
    cdef public list acc_weights
    cdef public list threshold
    cdef public list I_t
    cdef public list poisson_rate
    cdef public list external_weight
    cdef public list neu_type

    cdef public list poisson_spike

    cdef public float [:,:] states

    # Params
    cdef public list neu_params
    cdef public list trace_params


    cdef public Checkpoint.Checkpoint checkpoint

    cdef public float [:,:] post_traces
    cdef public float [:,:] post_traces_temp

    cpdef tuple state_update(self, int lid, int is_rollback)
    cpdef post_trace_update(self, int lid, int is_spiked, int temp)
    cpdef reset_temp_post_trace(self, int lid)
    cpdef poisson_stimulus(self, int lid)
    cpdef current_stimulus(self, int lid)
