# cython: boundscheck=False

cimport cython
cimport numpy as np
import numpy as np

cimport EnumList as EL


cpdef int coord_to_ind(int x, int y)
cpdef init_sync_topology()
cpdef load_constant(neu_const_filename, syn_const_filename)
cpdef mapping(list gid_to_core, list lid_to_gid, mapping_filename)
cpdef init_state(list neu_states, list gid_to_core, state_filename)
cpdef tuple load_connection (conn_filename, state_filename, mapping_filename, hw_mapping_filename)
cpdef init_network (network_filename)
cpdef load_stimulus (stimulus_filename)
