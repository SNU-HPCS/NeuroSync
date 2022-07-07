cimport EnumList as EL
cimport Core

cpdef tuple ind_to_coord(int ind)

cdef class Router:
    # position
    cdef public int ind
    cdef public int x
    cdef public int y
    cdef public int x_in_chip
    cdef public int y_in_chip
    cdef public int chip_x
    cdef public int chip_y

    # Routing
    cdef public list recv_buf
    cdef public list send_buf
    cdef public object spike_in_buf
    cdef public object spike_out_buf
    cdef public int spike_generator_available_cyc
    cdef public object synapse_events
    cdef public object sw
    cdef public list core_indir
    cdef public list core_dat

    # Core
    cdef public object core

    # Synchronization
    cdef public int parent_x
    cdef public int parent_y
    cdef public int num_children
    cdef public list children
    cdef public int child_commit_count
    cdef public int child_ready_count
    cdef public int ack_left
    cdef public object sync_role
    cdef public int recv_commit

    # Recovery
    cdef public object recovery_stack

    # Check if the core is in BS / AS state
    cpdef is_core_bs(self)
    cpdef is_core_as(self)
    cpdef spike_packet_generation(self)
    cpdef num_packets(self, lid)
    cpdef packet_in(self, recv_packet)
    cpdef packet_reception(self)
    cpdef send_commit_msg(self)
    cpdef send_sync_msg(self)
    cpdef check_commit(self)
