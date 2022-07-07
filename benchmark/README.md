# Benchmarks

To generate a network, we provide NeuroSync API which transforms a python-defined network into the simulator-compatible metadata.
The benchmark should call following APIs in the provided order to properly generate metadata.
You can rely on the "example1/example_workload.py" for more detail

## Init Simulation
```sh
$ init_simulation(dt = dt, simtime = simtime)
```
- dt: the timestep granularity
- simtime: the maximum simulation time

## Create Neurons
```sh
$ create_neurons(num_exc_neurons,
                 num_inh_neurons,
                 initial_states,
                 parameters)
```
- num_exc/inh_neurons: the number of excitatory/inhibitory neurons
- initial_states: the list of initial states for each neurons
    - please refer to the supported neuron models in neurosync/Neuron.pyx
    - you may define your own neuron model if you want
- parameters: the dictionary of the parameters for the target model

## External Stimulus
We allow two types of external stimulus 
- poisson neurons connected to multiple neurons
- current input dedicated to each neuron
```sh
$ create_external_stimulus(num_exc_poisson_neurons,
                           num_inh_poisson_neurons,
                           exc_poisson_rates,
                           inh_poisson_rates,
                           exc_poisson_conn,
                           inh_poisson_conn,
                           weight_exc_conn,
                           weight_inh_conn,
                           I_list)
```
- num_exc/inh_poisson_neurons: the number of excitatory/inhibitory poisson neurons
- exc/inh_poisson_rates: the firing rate of the excitatory/inhibitory poisson neurons
- exc/inh_poisson_conn: the number of connections per poisson neurons
- weight_exc/inh_conn: the weight of each poisson-to-neuron connection
- I_list: the list of constant current value for each neurons

## Connections
```sh
$ create_connections(connection,
                     parameters)
```
- connection: list of tuples defining (src_id, dst_id, weight, delay, is_plastic)
    - refer to the example benchmark for more details
- parameters: the learning-related parameters for the target learning rule

## End
```sh
$ end_simulation()
```
