# Random Sampling Sparsification
Repository for effective resistance sampling to create network sparsifiers
## Download 
Download file and import the ``Network.py`` file. 
```sh
from Network import *
```
## Import network file into ``Network`` class.
The Network file either takes two numpy arrays, an edge list (2 x m) and a list of edge weights (1 x m), or a `networkx` graph object.
For the edge list and list of edge weights, the network is imported as
```sh
from Network import *

E_list = edgelist
weights = edgeweights
network = Network(E_list, weights)
```
For a `networkx` object, the graph is imported as
```sh
G = networkx_graph
network = Network(None, None, G)
```
## Approximate Effective Resistance
To approximate the effective resistance, use the `Network` class specific command:
```sh
network = Network(E_list, weights)
epsilon=0.1
method='kts'
Effective_R = network.effR(epsilon, method)
```

**Arguments**
* **epsilon** Signifies the amount of relative error in the effective resistance approximation. 
* **method** Specifies the method of approximation to be used - 'ext' is the exact calculation of effective resistance, 'ssa' the original Spielman-Srivastva algorithm, and 'kts' as the Koutis et al. implementation. 

## Sparsification by Effective Resistance Sampling
Creating a spectral sparsifier through sampling edges proportional to their effective resistances can be achieved through a `Network` object:
```sh
network = Network(E_list, weights)
epsilon=0.1
method='kts'
Effective_R = network.effR(epsilon, method)
q = 10000
seed = 2020
EffR_Sparse = network.spl(q, Effective_R, seed=2020)
```

**Arguments**
* **q** The number of samples to take with replacement to create the effective resistance sparsifier by random sampling. 
* **effR** Effective resistance values for each edge.
* **seed=None** Seed to create the effective resistance sparsifier through random sampling.

**Mobility Data**
* Mobility data in the form of a directional network from the United States for the year 2016 can be found in the following DropBox link: https://www.dropbox.com/sh/2ennvbkb5drhdb1/AACB0ZeZAqavpHqDIARGzU7ya?dl=0
* The real-world mobility network was constructed from publicly available United States Census Bureau inter-census-tract commuting flows for all fifty states. Each node is a single census tract, and integer edge weights denote the amount of inter-census-tract human mobility provided by the United States Census Bureau through a summary of Longitudinal Employer-Household Dynamics (LEHD) Origin-Destination Employment Statistics (LODES) across Origin-Destination (OD), Residence Area Characteristic (RAC), and Workplace Area Characteristic (WAC) data types for the year 2016.

## Authors

* **Alexander M. Mercier** - [EffectiveResistanceSampling](https://github.com/AMMercier/EffectiveResistanceSampling)

## License

* **GNU General Public License**
