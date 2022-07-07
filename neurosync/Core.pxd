cimport EnumList as EL
cimport Checkpoint
cimport Learning
cimport Neuron
cimport Router
cimport RRManager

cdef class Core:
    cdef public int ind

    # Core states
    cdef public EL.FsmIndex FSM_state
    cdef public EL.FsmIndex FSM_prev_state
    cdef public int next_sync_timestep
    cdef public int cyc
    cdef public int cur_period

    # Connection
    # (core-to-core)
    cdef public list pid_to_gid
    cdef public int  pid_length
    # (neu-to-neu)
    cdef public float [:] syn_mem_weight
    cdef public int [:,:] syn_mem_others
    cdef public list syn_mem_debug
    cdef public list inverse_table

    # Router
    cdef public Router.Router router

    # Neurons
    cdef public int neu_num
    cdef public int poisson_num
    cdef public Neuron.NeuronModule neuron_module
    cdef public int next_comp_lid

    # Learning 
    cdef public int post_syn_learning_cyc
    cdef public object spike_in_history_buf
    cdef public Learning.LearningModule learning_module
    cdef public int weight_recovery_end_cyc

    # Checkpoints to handle RR
    cdef public Checkpoint.Checkpoint checkpoint
    cdef public RRManager.RRManager rr_manager

    # Key module to handle rollback and recovery
    cdef public object rollback_queue
    cdef public RRManager.RecoveryStack recovery_stack
    cdef public int reset_cyc

    # To profile the elapsed time
    cdef public float prev_time

    cpdef core_advance(self)
    cpdef neuron_computation(self)
    cpdef process_synapse_events(self)
    cpdef spike_to_synapse_events(self)
    cpdef accumulate(self, int dst_lid, int affecting_timestep, int neu_type, int is_anti, float weight)
