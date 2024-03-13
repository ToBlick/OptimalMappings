# OptimalMappings
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://toblick.github.io/OptimalMappings.jl/dev/)

This package contains the code used in the numerical examples of the paper [A registration method for reduced basis problems using linear optimal transport](https://arxiv.org/abs/2304.14884).

The directory `scripts` contains two files to reproduce the examples shown in the paper. Example one is also demonstrated [here](https://toblick.github.io/OptimalMappings.jl/dev/) using lower resolution and fewer solution samples (this code is executed whenever the code in this repository is updated, hence this page is always up to date).

To execute the scripts, clone the repository:
```git clone https://github.com/ToBlick/OptimalMappings.jl```
Change to the OptimalMappings.jl directory and open Julia:
```
cd OptimalMappings.jl
julia --project
```
Install the (un-registered) dependency `OptimalTransportTools.jl` and instantiate the project:
```
] 
add https://github.com/JuliaRCM/OptimalTransportTools.jl
instantiate
```
Lastly, run the script(s):
```
julia --project scripts/ex1.jl
julia --project scripts/ex2.jl
```
Generated figures are saved in `OptimalMappings.jl/figs`.

Running the scripts with the parameters from the paper takes around seven minutes each on a M1 processor with 16GB of RAM using Julia version 1.10.1. This can be sped up by reducing either the resolution `N` or the number of snapshots in the training set `nₛ` and/or test set `nₜ`.
