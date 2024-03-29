
# Coherent Continous-Variable Machine Simulators

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Maintainer](https://img.shields.io/badge/Maintainer-1QBit-blue)](http://1qbit.com/)
[![Paper](https://img.shields.io/badge/Paper-arxiv-red)](https://arxiv.org/abs/2209.04415)
[![Docs](https://img.shields.io/badge/Docs-Link-yellow)](https://1qb-information-technologies.github.io/ccvm/)

This software package includes solvers for continuous optimization problems. The solvers are simulators of coherent continuous-variable machines (CCVM), which are novel coherent network computing architectures based on NTT Research’s coherent Ising machines (CIM). In CCVMs, the physical properties of optical pulses (e.g., mean-field quadrature amplitudes and phase) represent the continuous variables of a given optimization problem. Various features of CCVMs can be used along with programming techniques to implement the constraints imposed by such an optimization problem. Included are methods for solving box-constrained quadratic programming (BoxQP) problems using CCVM simulators by mapping the variables of the problems into the random variables of CCVMs.

## Table of Contents

0. [Requirements](#requirements)
1. [Quickstart](#quickstart)
2. [Usage](#usage)
3. [Documentation](#docs)
    - [BoxQP Problem Definition](ccvm_simulators/problem_classes/README.md)
    - [ccvmplotlib](ccvm_simulators/ccvmplotlib/README.md)
4. [Contributing](#contributing)
5. [References](#references)
6. [License](#license)

## Requirements

- Python (supported version: 3.10)

### Supported operating systems

- macOS (Monterey, Big Sur)
- Ubuntu (20.04)

## Quickstart


1. Once you have cloned the repository, install the package using `pip`.
```
 pip install ccvm-simulators
```

2. Go into `examples/` and run the demo scripts.
    - `ccvm_boxqp_dl.py`: Solve a BoxQP problem using a CCVM simulator, w/o plotting
    - `ccvm_boxqp_plot.py`: Solve a BoxQP problem using a CCVM simulator, w/ time-to-solution (TTS) plotting
    - For more demo scripts see `examples/README.md`

3. View the generated plot.
    - The `ccvm_boxqp_plot.py` script solves a single problem instance, and will create an image of the resulting TTS plot in a `plots/` directory. The resulting output image, `DL-CCVM_TTS_cpu_plot.png`, will look something like this:

<p align="center">
    <img src="https://github.com/1QB-Information-Technologies/ccvm/blob/main/ccvm_simulators/ccvmplotlib/images/single_instance_TTS_plot.png?raw=true" width="250" >
</p>

### Extending the Example

4. Plotting larger datasets
    - The `ccvm_boxqp_plot.py` script is a toy example that plots the TTS for only a single problem instance.
    - It can be extended to solve multiple problems over a range of problem sizes.
    - When solution metadata is saved for multiple problems, the graph becomes more informative, for example:

<p align="center">
    <img src="https://github.com/1QB-Information-Technologies/ccvm/blob/main/ccvm_simulators/ccvmplotlib/images/tts_plot_example.png?raw=true" width="250" >
</p>


5. Other types of plots
    - `ccvmplotlib` can also plot the success probability data, for example:

<p align="center">
    <img src="https://github.com/1QB-Information-Technologies/ccvm/blob/main/ccvm_simulators/ccvmplotlib/images/success_prob_plot_example.png?raw=true" width="250">
</p>

## Usage

### Solving a BoxQP Problem

##### 1. Import Modules

```python
from ccvm_simulators.problem_classes.boxqp import ProblemInstance
from ccvm_simulators.solvers import DLSolver
```

##### 2. Define a Solver

```python
solver = DLSolver(device="cpu", batch_size=100)  # or "cuda"
solver.parameter_key = {
    20: {"pump": 2.0, "dt": 0.005, "iterations": 15000, "noise_ratio": 10},
}
```

##### 3. Load a Problem Instance

```python
boxqp_instance = ProblemInstance(
    instance_type="test",
    file_path="./examples/benchmarking_instances/single_test_instance/test020-100-10.in",
    device=solver.device,
)
```

##### 4. Scale the Coefficients
The BoxQP problem matrix Q and vector V are normalized to obtain a uniform
performance across different problem sizes and densities. The ideal range depends on the
solver. For best results, Q should be passed to the solver's `get_scaling_factor()`
method to determine the best scaling value for the problem–solver combination.

```python
boxqp_instance.scale_coefs(solver.get_scaling_factor(boxqp_instance.q_matrix))
```

##### 5. Solve the Problem Instance

```python
solution = solver(
    instance=boxqp_instance,
    post_processor=None,
)

print(f"The best known solution to this problem is {solution.optimal_value}")
# The best known solution to this problem is 799.560976

print(f"The best objective value found by the solver was {solution.best_objective_value}")
# The best objective value found by the solver was 798.1630859375

print(f"The solving process took effectively {solution.solve_time} seconds to solve a single instance")
# The solving process took 0.008949262142181396 seconds
```

## Documentation

The package documentation can be found [here](https://1qb-information-technologies.github.io/ccvm/).

Additional links:
- Problem definition: [BoxQP problem class](ccvm/ccvm_simulators/problem_classes/README.md)
- Plotting library: [ccvmplotlib](ccvm/ccvm_simulators/ccvmplotlib/README.md)


## Contributing

We appreciate your pull requests and welcome opportunities to discuss new ideas. Check out our [contribution guide](CONTRIBUTING.md) and feel free to improve the `ccvm_simulators` package. For major changes, please open an issue first to discuss any suggestions for changes you might have.

Thank you for considering making a  contribution to our project! We appreciate your help and support.


## References

This repository contains architectures and simulators presented in the paper ["Non-convex Quadratic Programming Using Coherent Optical Networks"](https://arxiv.org/abs/2209.04415) by Farhad Khosravi, Ugur Yildiz, Martin Perreault, Artur Scherer, and Pooya Ronagh.


## License

[APGLv3](https://github.com/1QB-Information-Technologies/ccvm/blob/main/LICENSE)
