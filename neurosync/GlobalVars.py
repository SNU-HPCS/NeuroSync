import random

# debug a specific neuron
debug_list = {}
#debug_target = [1136]
#debug_src = [5961]
#debug_target = [1483]
#debug_src = [4917]
debug_target = []
debug_src = []

sim_params = {}

sim_params["chip_num"] = 0
sim_params["max_x"] = 0
sim_params["max_y"] = 0

#####
syn_consts = None
neu_consts = None

#############
timestep = None
leading_timestep = 0
cyc = None
chkpt_timestep = None

cores = None
NoC = None
sync_root = None

external_spike = None

spike_out = None

# ICN property
timing_params = {}

# NoC Timing
timing_params["switch_EW_delay"] = 1
timing_params["switch_NS_delay"] = 1
timing_params["chip_delay"] = 32
timing_params["board_delay"] = 32

timing_params["core_to_sw_bw"] = 1
timing_params["sw_to_core_bw"] = 1
timing_params["sw_to_sw_bw"] = 1
# the off chip bw is defined as
# the # of packets that can be transferred (off_chip_window_num)
# during a given time window (off_chip_window_size)


# the total packets per cyc =
# off_chip_integer + (off_chip_window_num / off_chip_window_size)
timing_params["off_chip_integer"] = 1
timing_params["off_chip_window_size"] = 10
timing_params["off_chip_window_num"] = 3

# trace update pipeline depth
timing_params["trace_update_pipeline"] = 3
# learning pipeline depth
timing_params["learning_pipeline"] = 3


# For recovery stack
STACK_EXTENSION_LEN = 100
recovery_length = 0


def seed_random(seed)->None:
    random.seed(seed)


def get_random()->float:
    return random.random()


def get_randint(i, j)->int:
    return random.randint(i, j)
