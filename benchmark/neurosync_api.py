import os 
import sys

import numpy as np
import math
import random

import networkx as nx 
from elephant.spike_train_generation import homogeneous_poisson_process
from quantities import Hz, s, ms, mV, dimensionless


#
network_params = {}
syn_params = None
neu_params = None
neu_states = None
external_stimulus = None


simtime_dat = None
dt_dat = None

num_exc_neurons_dat = -1
num_inh_neurons_dat = -1
num_exc_poisson_neurons_dat = 0
num_inh_poisson_neurons_dat = 0

conn_list = []

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def init_simulation(simtime: object,
                    dt: object):
    global simtime_dat
    global dt_dat
    global network_params

    if not os.path.exists("dataset"):
        os.mkdir("dataset")
        

    simtime_dat = simtime
    dt_dat = dt
    network_params["simtime"] = int((simtime / dt).simplified)
    network_params["timestep_per_sec"] = int((1 * s / dt).simplified)

    print("Initialized Simulation")
    sys.stdout.flush()


def create_neurons(num_exc_neurons: int,
                   num_inh_neurons: int,
                   initial_states: list,
                   parameters: list):

    global network_params
    global neu_params
    global neu_states
    global num_exc_neurons_dat
    global num_inh_neurons_dat

    if simtime_dat == None or dt_dat == None:
        assert(0 and "Initialize simulation before creating neurons")

    num_exc_neurons_dat = num_exc_neurons
    num_inh_neurons_dat = num_inh_neurons

    network_params["ne_sim"] = num_exc_neurons
    network_params["ni_sim"] = num_inh_neurons

    type_list = [0 if i < num_exc_neurons else 1 for i in range(num_exc_neurons + num_inh_neurons)]
    for ind in range(len(initial_states)):
        initial_states[ind]["neu_type"] = type_list[ind]

    neu_params = parameters
    neu_states = initial_states

    print("Created Neurons")
    sys.stdout.flush()


def create_external_stimulus(num_exc_poisson_neurons: int,
                             num_inh_poisson_neurons: int,
                             exc_poisson_rates: object,
                             inh_poisson_rates: object,
                             exc_poisson_conn: int,
                             inh_poisson_conn: int,
                             weight_exc_conn: float,
                             weight_inh_conn: float,
                             I_list: list):

    global external_stimulus

    global simtime_dat
    global dt_dat

    global num_exc_neurons_dat
    global num_inh_neurons_dat
    global num_exc_poisson_neurons_dat
    global num_inh_poisson_neurons_dat

    global conn_list

    if num_exc_neurons_dat == -1 or num_inh_neurons_dat == -1:
        assert(0 and "Create neurons before generating stimulus")

    num_exc_poisson_neurons_dat = num_exc_poisson_neurons
    num_inh_poisson_neurons_dat = num_inh_poisson_neurons

    num_poisson_neurons = num_exc_poisson_neurons_dat + num_inh_poisson_neurons_dat
    num_neurons = num_exc_neurons_dat + num_inh_neurons_dat
    poisson_spikes = [[] for _ in range(num_poisson_neurons)]

    for n in range(num_poisson_neurons):
        if n < num_exc_poisson_neurons:
            poisson_list = homogeneous_poisson_process(rate=exc_poisson_rates,
                                                       t_start=0.0*s,
                                                       t_stop=simtime_dat)
        else:
            poisson_list = homogeneous_poisson_process(rate=inh_poisson_rates,
                                                       t_start=0.0*s,
                                                       t_stop=simtime_dat)
        poisson_list = (poisson_list / dt_dat).simplified
        poisson_list = [int(p) for p in poisson_list]
        poisson_spikes[n] = poisson_list
    
    # exc poisson to network
    s_arr_pe = np.asarray([], dtype=int)
    d_arr_pe = np.asarray([], dtype=int)

    # inh poisson to network
    s_arr_pi = np.asarray([], dtype=int)
    d_arr_pi = np.asarray([], dtype=int)

    for npe in range(num_exc_poisson_neurons):
        d_arr_temp = np.random.choice(range(num_neurons), size=exc_poisson_conn, replace=False)
        d_arr_pe = np.append(d_arr_pe, d_arr_temp)
        s_arr_temp = np.empty(exc_poisson_conn, dtype=int)
        s_arr_temp.fill(int(npe + num_neurons))
        s_arr_pe = np.append(s_arr_pe, s_arr_temp)
    
    for npi in range(num_inh_poisson_neurons):
        d_arr_temp = np.random.choice(range(num_neurons), size=inh_poisson_conn, replace=False)
        d_arr_pi = np.append(d_arr_pi, d_arr_temp)
        s_arr_temp = np.empty(inh_poisson_conn, dtype=int)
        s_arr_temp.fill(int(npi + num_neurons + num_exc_poisson_neurons))
        s_arr_pi = np.append(s_arr_pi, s_arr_temp)

    for ind in range(len(s_arr_pe)):
        conn_list.append((s_arr_pe[ind], d_arr_pe[ind], weight_exc_conn, 1, 0))

    for ind in range(len(s_arr_pi)):
        conn_list.append((s_arr_pi[ind], d_arr_pi[ind], weight_inh_conn, 1, 0))

    external_stimulus = {
        "valid_list" :
            {"poisson_neurons" : True,
             "current_stimulus" : True},
        "poisson_neurons" :
            {"npe_sim" : num_exc_poisson_neurons,
             "npi_sim" : num_inh_poisson_neurons,
             "e_rates" : float(exc_poisson_rates / Hz),
             "i_rates" : float(inh_poisson_rates / Hz),
             # optional (can be internally generated)
             "external": True,
             "spikes"  : poisson_spikes},
        "current_stimulus" : 
            {"I_list"  : I_list}
    }

    print("Created External Stimulus")
    sys.stdout.flush()


def create_connections(connection: list,
                       parameters: dict):


    global conn_list
    global syn_params

    conn_list = list(conn_list) + list(connection)
    syn_params = parameters

    print("Created Connections")
    sys.stdout.flush()


def end_simulation():
    global network_params
    global syn_params
    global neu_params
    global neu_states
    global external_stimulus

    global simtime_dat
    global dt_dat

    global num_exc_neurons_dat
    global num_inh_neurons_dat
    global num_exc_poisson_neurons_dat
    global num_inh_poisson_neurons_dat
    
    global conn_list
    global network_params
    global syn_params
    global neu_params
    global neu_states
    global external_stimulus
    global conn_list

    # check if the network has been properly initialized
    assert(syn_params)
    assert(neu_params)
    assert(neu_states)
    assert(external_stimulus)
    assert(len(conn_list))

    conn_list.sort(key=lambda tup: tup[0])
    conn_list = np.asarray(conn_list, dtype='i4,i4,f4,i4,bool')
    
    np.save("dataset/network_parameter.npy", network_params)
    np.save("dataset/synapse_parameter.npy", syn_params)
    np.save("dataset/neuron_parameter.npy", neu_params)
    np.save("dataset/initial_states.npy", neu_states)
    np.save("dataset/stimulus.npy", external_stimulus)
    np.save('dataset/connection.npy', conn_list)

    print("Generated Required Metadata")
    sys.stdout.flush()

    network_params = {}
    syn_params = None
    neu_params = None
    neu_states = None
    external_stimulus = None
    
    
    simtime_dat = None
    dt_dat = None
    
    num_exc_neurons_dat = -1
    num_inh_neurons_dat = -1
    num_exc_poisson_neurons_dat = 0
    num_inh_poisson_neurons_dat = 0
    
    conn_list = []
