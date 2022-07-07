import sys, os

sys.path.append("..")
from neurosync_api import *


seed = 0

set_seed(seed)
network_gen_seed = seed

############################################
# Define Network Parameters
############################################

dt = 0.1*ms
simtime = 1. * s

# set initialize the simulation
init_simulation(dt = dt, simtime = simtime)

############################################
# Define Neuron Parameters
############################################

# we provide three examples
# 1) DLIF
# 2) Izhikevich
# 3) Alpha
NEU_MODEL = "DLIF"

### DLIF Neuron Model ###
if NEU_MODEL == "DLIF":
    r_ar_max = 5. * ms
    threshold = -50.*mV

    tau_mem = 20 * ms
    tau_inh = 10.0*ms
    tau_exc = 5.0*ms
    El = -60*mV
    E_0 = El
    reversal_vI = -80*mV
    reversal_vE = 0 * mV

    decay_gE = 1. - 1. / float(tau_exc / dt)
    decay_gI = 1. - 1. / float(tau_inh / dt)
    decay_v = 1. - 1. / float(tau_mem / dt)

    # same params for exc and inh
    neu_params = [{
        "r_ar_max" : int(r_ar_max / dt),
        "reversal_vE" : float(reversal_vE / mV),
        "reversal_vI" : float(reversal_vI / mV),
        "decay_gE" : float(decay_gE),
        "decay_gI" : float(decay_gI),
        "decay_v" : float(decay_v),
        "E_0" : float(E_0 / mV),
        "E_L" : float(E_0 / mV),
        "model" : "DLIF"
    } for _ in range(2)]

### Izhikevich Neuron Model ###
if NEU_MODEL == "Izhikevich":
    r_ar_max = 5. * ms
    threshold = 30 * mV

    v_a = [0.04 / ms * dt, 0.04 / ms * dt]
    v_b = [5. / ms * dt, 5. / ms * dt]
    v_c = [140 / ms * dt, 140 / ms * dt]

    u_a = [0.02/ms * dt, (0.02+0.08*0.5)/ms * dt]
    u_b = [0.2/ms * dt, (0.25 - 0.05*0.5)/ms * dt]
    u_c = [(-65.0 + 15*(0.5)), -65.0]
    u_d = [(8.0 - 6.0*(0.5))/ms * dt * dt / ms, (2.0) / ms * dt * dt / ms]

    neu_params = [{
        "r_ar_max" : int(r_ar_max / dt),
        "E_0" : float(u_c[type]),
        "v_a" : float(v_a[type]),
        "v_b" : float(v_b[type]),
        "v_c" : float(v_c[type]),
        "u_a" : float(u_a[type]),
        "u_b" : float(u_b[type]),
        "u_c" : float(u_c[type]),
        "u_d" : float(u_d[type]),
        "model" : "Izhikevich"
    } for type in range(2)]

### Alpha Neuron Model
if NEU_MODEL == "Alpha":
    r_ar_max = 5. * ms
    threshold = -50 * mV
    E_0 = -70 * mV

    tau_mem = 20. * ms
    tau_inh = 10. * ms
    tau_exc = 5. * ms

    decay_gE = 1. - 1. / (tau_exc / dt)
    decay_gI = 1. - 1. / (tau_inh / dt)
    decay_yE = 1. - 1. / (tau_exc / dt)
    decay_yI = 1. - 1. / (tau_inh / dt)
    decay_v = 1. - 1. / (tau_mem / dt)

    # same params for exc and inh
    neu_params = [{
        "r_ar_max" : int(r_ar_max / dt),
        "decay_gE" : float(decay_gE),
        "decay_gI" : float(decay_gI),
        "decay_yE" : float(decay_yE),
        "decay_yI" : float(decay_yI),
        "decay_v" : float(decay_v),
        "E_0" : float(E_0 / mV),
        "model" : "Alpha"
    } for _ in range(2)]



############################################
# Set Initial States
############################################

ne_sim = 1600
ni_sim = 400
n_sim = ne_sim + ni_sim

vm_list = float(El / mV) + float((threshold - El) / mV) * np.random.rand(n_sim)
vm_list = np.asarray(vm_list, dtype=float)
gE_list = np.zeros(n_sim, dtype=float)
gI_list = np.zeros(n_sim, dtype=float)
threshold_list = np.empty(n_sim, dtype=float)
threshold_list.fill(float(threshold / mV))
type_list = [0 if i < ne_sim else 1 for i in range(n_sim)]
type_list = np.asarray(type_list, dtype=np.uint8)

state_dict = [{
    "v_t" : vm_list[gid],
    "gE_t" : gE_list[gid],
    "gI_t" : gI_list[gid],
    "threshold" : threshold_list[gid],
    "neu_type" : type_list[gid]
} for gid in range(n_sim)]

create_neurons(num_exc_neurons = ne_sim,
               num_inh_neurons = ni_sim,
               initial_states = state_dict,
               parameters = neu_params)

############################################
# Define External Stimulus
############################################

# There are three options
# 1) poisson neurons connected to multiple neurons
# 2) current input dedicated to each neuron

# 1) poisson neurons

# 1-1) number of poisson exc / inh neurons
npe_sim = 40
npi_sim = 40
np_sim = npe_sim + npi_sim

# 1-1-1) poisson spikes
ep_rates = 10*Hz
ip_rates = 10*Hz

# 1-2) weight / connections of the poisson neurons
# generate random feedforward connections
# connections per poisson neuron
npe_conn = 100
w_syn_pe = 0.15 / 1000.

npi_conn = 100
w_syn_pi = 0.15 / 1000.

# 2) current input
# input current for a neuron
I = 1 * mV / ms * dt
I_list = np.empty(n_sim, dtype=float)
I_list.fill(float((I / mV).simplified))

create_external_stimulus(num_exc_poisson_neurons = npe_sim,
                         num_inh_poisson_neurons = npi_sim,
                         exc_poisson_rates = ep_rates,
                         inh_poisson_rates = ip_rates,
                         exc_poisson_conn = npe_conn,
                         inh_poisson_conn = npi_conn,
                         weight_exc_conn = w_syn_pe,
                         weight_inh_conn = w_syn_pi,
                         I_list = I_list)

############################################
# Define Synapse Parameters
############################################

# we provide four examples
# 1) Triplet
# 2) Voltage
# 3) Suppression
# 4) Short-Term (STP)
SYN_MODEL = "TRIPLET"

### TRIPLET Learning Rule ###
if SYN_MODEL == "TRIPLET":

    gmax = 0.3 / 100.           # Maximum weight
    tau_stdp = 20*ms            # Triplet time constant
    tau_stdp_triplet = 100*ms   # Triplet time constant
    alpha = (3*Hz*tau_stdp*3).simplified     # Target rate parameter
    eta = 2e-2 / 2000           # Learning rate
    eta_triplet = 2e-3 / 2000   # Learning rate

    decay_stdp = 1. - 1. / float(tau_stdp / dt)
    decay_triplet = 1. - 1. / float(tau_stdp_triplet / dt)

    syn_params = {
        "decay_stdp" : decay_stdp,
        "decay_triplet" : decay_triplet,
        "alpha" : float(alpha),
        "eta_stdp" : float(eta),
        "eta_triplet" : float(eta_triplet),
        "gmax" : float(gmax),
        "rule" : "TRIPLET"
    }

### VOLTAGE Learning Rule ###
if SYN_MODEL == "VOLTAGE":
    gmax = 0.3 / 100.               # Maximum weight
    tau_ca = 400*ms                 # Calcium
    slope = gmax * 0.0001 / dt      # Slope of the linear decay

    th_low_ca_up = 12.0             # Voltage model params 
    th_low_ca_dn = th_low_ca_up     #
    th_low_ca = th_low_ca_dn        #

    th_high_ca_up = 0.5
    th_high_ca_dn = th_high_ca_up
    th_high_ca = th_high_ca_dn

    th_v = -55.0 * mV
    a_up = 0.02 * gmax
    b_dn = a_up
    up_dn = a_up

    decay_ca = 1. - 1. / (tau_ca / dt)

    syn_params = {
        "decay_ca" : float(decay_ca),
        "th_low_ca" : float(th_low_ca),
        "th_high_ca" : float(th_high_ca),
        "th_v" : float(th_v / mV),
        "up_dn" : float(up_dn),
        "slope" : float(slope * dt),
        "gmax" : float(gmax),
        "rule" : "VOLTAGE"
    }

### 
if SYN_MODEL == "SUPPRESS":
    gmax = 0.3 / 100.
    tau_stdp = 20*ms
    tau_suppress = 100*ms
    eta = 0.01

    decay_stdp = 1. - 1. / (tau_stdp / dt)
    decay_suppress = 1. - 1. / (tau_suppress / dt)

    syn_params = {
        "decay_stdp" : float(decay_stdp),
        "decay_suppress" : float(decay_suppress),
        "eta_stdp" : float(eta),
        "gmax" : float(gmax),
        "rule" : "SUPPRESS"
    }

### 
if SYN_MODEL == "STP":
    gmax = 0.3 / 100.
    tau_stdp = 40*ms    # STDP time constant
    alpha = 3*Hz*tau_stdp*2  # Target rate parameter
    tau_fac = 40.0*ms
    tau_rec = 1.0*ms
    U = 1.0
    eta = 1e-2 / 20         # Learning rate

    decay_stdp = 1. - 1. / (tau_stdp / dt)
    decay_u = 1. - 1. / (tau_fac / dt)
    decay_x = 1. - 1. / (tau_rec / dt)

    syn_params = {
        "decay_stdp" : decay_stdp,
        "decay_x" : decay_x,
        "decay_u" : decay_u,
        "alpha" : alpha,
        "eta_stdp" : eta,
        "U" : U,
        "gmax" : float(gmax),
        "rule" : "STP"
    }


############################################
# Construct Network using Nx (An Example)
############################################
# generate a list of tuples for a synapse (src_id, dst_id, weight, delay, is_plastic)
# src_id: id of the source neuron
# dst_id: id of the destination neurons
# weight: weight of the synapse
# delay: synaptic delay of the synapse
# is_plastic: is learning enabled for the synapse

# Define network weight
w_e = 0.15 / 1000.
w_ii = 0.15 / 100.
w_ie = 0.15 / 100.
delay = 5

# Random network generation using nx
beta = 1.0
epsilon = 0.1
G_con_src = nx.watts_strogatz_graph(n_sim, int(n_sim * epsilon), beta, seed=network_gen_seed)
G_con_dst = nx.watts_strogatz_graph(n_sim, int(n_sim * epsilon), beta, seed=network_gen_seed+1)
edges_src = G_con_src.edges()
edges_dst = G_con_dst.edges()

conn_list = []

for e in edges_src:
    src = e[0]; dst = e[1]
    # exc / exc
    src_type = type_list[src]
    dst_type = type_list[dst]

    # we assume that only i->e connection is plastic
    if src_type == 0 and dst_type == 0:
        conn_list.append((src, dst, w_e, delay, False))
    elif src_type == 0 and dst_type == 1:
        conn_list.append((src, dst, w_e, delay, False))
    elif src_type == 1 and dst_type == 0:
        conn_list.append((src, dst, w_ie, delay, True))
    else:
        conn_list.append((src, dst, w_ii, delay, False))

for e in edges_dst:
    src = e[1]; dst = e[0]
    # exc / exc
    src_type = type_list[src]
    dst_type = type_list[dst]

    # we assume that only i->e connection is plastic
    if src_type == 0 and dst_type == 0:
        conn_list.append((src, dst, w_e, delay, False))
    elif src_type == 0 and dst_type == 1:
        conn_list.append((src, dst, w_e, delay, False))
    elif src_type == 1 and dst_type == 0:
        conn_list.append((src, dst, w_ie, delay, True))
    else:
        conn_list.append((src, dst, w_ii, delay, False))

create_connections(connection = conn_list, parameters = syn_params)

end_simulation()
