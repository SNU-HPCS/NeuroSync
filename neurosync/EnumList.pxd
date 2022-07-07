# Required Action
cpdef enum ActionIndex:
    no_action = 0
    gen_spike = 1
    gen_antispike = 2


# Required Neuronal States
cpdef enum StateIndex:
    v_t = 0
    gE_t = 1
    gI_t = 2
    yE_t = 3
    yI_t = 4
    refr = 5
    w_t = 6
    state_max = 7


# Required Neuron Parameters
cpdef enum NeuParamIndex:
    neu_param_v_a = 0
    neu_param_v_b = 1
    neu_param_v_c = 2
    neu_param_u_a = 3
    neu_param_u_b = 4
    neu_param_u_c = 5
    neu_param_u_d = 6
    neu_param_decay_gE = 7
    neu_param_decay_gI = 8
    neu_param_decay_v = 9
    neu_param_decay_u = 10
    neu_param_decay_ad = 11
    neu_param_a = 12
    neu_param_b = 13
    neu_param_d = 14
    neu_param_v_w = 15
    neu_param_Delta_T = 16
    neu_param_max = 17


# Required Neuron Parameters
cpdef enum LearnParamIndex:
    learn_param_U = 0
    learn_param_eta_stdp = 1
    learn_param_eta_triplet = 2
    learn_param_gmax = 3
    learn_param_alpha = 4
    learn_param_slope = 5
    learn_param_th_low_ca = 6
    learn_param_th_high_ca = 7
    learn_param_th_v = 8
    learn_param_up_dn = 9
    learn_param_max = 10


# Required Post Trace Parameters
cpdef enum PostTraceParamIndex:
    postt_param_decay_stdp = 0
    postt_param_decay_triplet = 1
    postt_param_decay_eps = 2
    postt_param_decay_ca = 3
    postt_param_max = 4


# Required Pre Trace Parameters
cpdef enum PreTraceParamIndex:
    prett_param_decay_stdp = 0
    prett_param_decay_triplet = 1
    prett_param_decay_suppress = 2
    prett_param_decay_eps = 3
    prett_param_decay_x = 4
    prett_param_decay_u = 5
    prett_param_max = 6


# Required Spike Types
cpdef enum SpiketypeIndex:
    norm_spike = 0
    anti_spike = 1


# Required Synch Role per Core
cpdef enum SyncroleIndex:
    root = 0
    non_root = 1
    syncrole_max = 2


cpdef enum FsmIndex:
    neu_comp_state = 0
    neu_comp_end_state = 1
    sync_wait_state = 2
    commit_state = 3
    sync_done_state = 4
    rollback_state = 5


cpdef enum DirectionIndex:
    left = 0
    bot = 1
    top = 2
    right = 3
    direction_max = 4


cpdef enum SynmemIndex:
    dst_lid = 0
    delay = 1
    learning_en = 2
    lastspike_t = 3
    chkpt_addr = 4
    synmem_max = 5


cpdef enum PacketIndex:
    spike = 0
    sync_req = 1
    ack = 2
    commit = 3
    antispike = 4


cpdef enum PosttIndex:
    k_minus_t = 0
    k_minus_triplet_t = 1
    post_eps_t = 2
    post_eps_t_temp = 3
    ca_t = 4
    postt_max = 5


cpdef enum PretIndex:
    k_plus_t = 0
    k_plus_triplet_t = 1
    pre_eps_t = 2
    pre_eps_t_temp = 3
    decayx_t = 4
    decayu_t = 5
    decayx_t_temp = 6
    decayu_t_temp = 7
    x_t = 8
    u_t = 9
    pret_max = 10


cpdef enum NeutypeIndex:
    normal_exc = 0
    normal_inh = 1
    poisson_exc = 2
    poisson_inh = 3
    neutype_max = 4


cpdef enum RecoveryStackIndex:
    stack_invalid = 0
    stack_valid = 1
    stack_retired = 2


cpdef enum LocalSpikeIndex:
    local_spike_cyc = 0
    local_spike_lid = 1
    local_spike_timestep = 2
    local_spike_type = 3
    local_spike_rs_ind = 4
    local_spike_max = 5

cpdef enum RemoteSpikeIndex:
    remote_spike_pid = 0
    remote_spike_addr = 1
    remote_spike_type = 2
    remote_spike_timestep = 3
    remote_spike_anti = 4
    remote_spike_rs_ind = 5
    remote_spike_max = 6

cpdef enum HistoryBufIndex:
    history_invalid = 0
    history_timestep = 1
    history_pid = 2
    history_type = 3
    history_start_addr = 4
    history_end_addr = 5
    history_anti = 6
    history_post_spikes = 7
    history_max = 8
