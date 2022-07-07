# NeuroSync

This repository is the software-based simulator for the paper entitled 
'NeuroSync: A Scalable and Accurate Brain Simulator Using Safe and Efficient Speculation'
published at HPCA 2022.

Unlike the existing multicore brain simulator, NeuroSync is the first simulator to
_speculatively advance the simulation with only periodic synchronization among the cores._
It also supports checkpoint, rollback, and recovery procedures to ensure that the
_simulation results are exactly the same as in a conventional simulator._

## Installation
##### [dependency]
- python >= 3.6

##### [packages]
- `pip install numpy networkx elephant Cython`
- `apt-get install metis`

## Workload
You should manually define your network before the simulation.
Please refer to the README file in the benchmark folder for detail.

## Simulate
##### [run]
```sh 
$ python3 run.py example.cfg 
```
It will first map the pre-defined networks in the benchmark folder to NeuroSync hardware.
It relys on an open-source mapping program (i.e., Metis) to map the neurons to the cores in a way that minimizes inter-core connections.
After mapping the network to the hardware, it copies your simulation code into the _runspace_ as multiple simulation instances 
and automatically execute them according to the parameters in "example.cfg".

## Cfg file
##### [run] 
- result_folder_name: simulation instances will run under "runspace_path/result_folder_name"
- workload_name: simulation instances will use input files under "workload_path/workload_name" and mapping files under "metis_interface/workload_name"
- max_timestep: maximum timestep
- timestep_per_sec: # of timesteps per second

##### [simulation_parameter]
Here, all parameters will make productive tuples.   
For example.cfg, the first tuple is [16, (1,1), (4,4), 8, 1, 0, 4] and the last tuple is [16, (2,2), (8,8), 8, 10, 0, 4].   
All tuples generate corresponding simulation instances.   

- chip_xy: chip dimension (x, y)
- max_core_xy_in_chip: core dimension (x, y) in a chip
- max_syn_delay: maximum syn delay in workload
- synch_period: period of synch & checkpointing
- recovery_depth: maximum depth of rollback cascading
- accum_width: number of parallel weight accumulation units
- pretrace_engine: number of parallel pre-trace update engines
- setup_timestep: number of timesteps to warm up before performing a speculative simulation

##### [mapping_parameter]
- ufactor: maximum load imbalance among the partition (used as a Metis parameter)

##### [path]
- simulator_path: absolute path of the cloned repository
- mapper_path: absolute path to mapping result
- workload_path: absolute path to brian data
- runspace_path: absolute path to runspace

### example.cfg
```sh
[run]
result_folder_name = lazy
workload_name = example1
max_timestep = 1600

[simulation_parameter]
chip_xy = [(8,8)]
max_core_xy_in_chip = [(8,8)]
max_syn_delay = [40]
synch_period = [1,8,32]
accum_width = [1]
pretrace_engine = [5,10]
setup_timestep = 50

[mapping_parameter]
ufactor = 2

[path]
simulator_path = [path_to_simulator]/neurosync/
mapper_path = [path_to_simulator]/neuronsync/simulation/metis_interface/
workload_path = [path_to_simulator]/neurosync/simulation/benchmark/
runspace_path = [path_to_simulator]/neurosync/simulation/runspace/
```
*"python3 run.py example.cfg"* will genearate below directories in *[runspace_path]*
(and automatically run them in parallel)

example1_peri1_5pretrace_eng/
example1_peri8_5pretrace_eng/
example1_peri32_5pretrace_eng/
example1_peri1_10pretrace_eng/
example1_peri8_10pretrace_eng/
example1_peri32_10pretrace_eng/

## Timing Parameters

We provide an example timing parameter in neurosync/GlobalVars.py
You may modify the timing parameters to explore various design spaces.

## Simulation Results

Check the generated log file in the runspace to check the simulation results.
Also, the err file reports the error during the simulation.

## Publications

### Original Publication

You can refer to the following publication for detailed descriptions of NeuroSync architecture.

["NeuroSync: A Scalable and Accurate Brain Simulator Using Safe and Efficient Speculation," in *(HPCA22).*](https://ieeexplore.ieee.org/document/9773227)

### Related Publications

You can also refer to the following related publications. 

["Flexon: A Flexible Digital Neuron for Efficient Spiking Neural Network Simulations," in *(ISCA18).*](https://ieeexplore.ieee.org/document/8416834)

["FlexLearn: Fast and Highly Efficient Brain Simulations Using Flexible On-Chip Learning," in *(MICRO19).*](https://dl.acm.org/doi/10.1145/3352460.3358268)

["NeuroEngine: A Hardware-Based Event-Driven Simulation System for Advanced Brain-Inspired Computing," in *(ASPLOS21).*](https://dl.acm.org/doi/abs/10.1145/3445814.3446738)
