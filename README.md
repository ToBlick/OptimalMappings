# OptimalMappings
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://toblick.github.io/OptimalMappings.jl/dev/)

This package contains the code used in the numerical examples of the paper [A registration method for reduced basis problems using linear optimal transport](https://arxiv.org/abs/2304.14884).

The directory `scripts` contains two files to reproduce the examples shown in the paper. Example one is also demonstrated [here](https://toblick.github.io/OptimalMappings.jl/dev/) using lower resolution and fewer solution samples (this code is executed whenever the code in this repository is updated, hence this page is always up to date).

This package is not registered. To run the code, either clone this repository or add the package via `pkg> add https://github.com/ToBlick/OptimalMappings.jl` or `"git@github.com:ToBlick/OptimalMappings.jl.git"` using the package manager and copy only the script files. To generate the figures, un-comment the corresponding code in the scripts and create a directory `figs` in the parent directory of `scripts`.
