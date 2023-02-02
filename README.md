# Coherent Continous-Variable Machine

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Maintainer](https://img.shields.io/badge/Maintainer-1QBit-blue)](http://1qbit.com/)
[![Paper](https://img.shields.io/badge/Paper-arxiv-red)](https://arxiv.org/abs/2209.04415)
[![Docs](https://img.shields.io/badge/Docs-Link-green)](https://urban-chainsaw-9k39nm4.pages.github.io/index.html)

The Coherent Continous-Variable Machine (CCVM) is a novel coherent quantum optical network architecture built on NTT's Coherent Ising Machine (CIM) where the physical properties of optical pulses (e.g. mean-field amplitude, phase, intensity, etc.) represent the continuous variables of a given optimization problem. Various features of the optical device and programming techniques can be used to implement the constraints imposed by these optimization problems. Here we demonstrate the application of CCVM in solving the Box-Constrained Quadratic Programming (BoxQP) problem by mapping the variables of problems into the random variables of CCVM.

## Table of Contents

1. [Quickstart](#quickstart)
2. [Usage](#usage)
3. [Docs](#docs)
    - [BoxQP Problem Definition](problem_classes/README.md)
    - [ccvmplotlib](ccvmplotlib/README.md)
4. [Contributing](#contributing)
5. [References](#references)
6. [License](#license)

## Quickstart

#### Using Docker

##### 1. Run container from anywhere

```
docker run -it -v $(pwd):/workspace/examples/plots quay.io/1qbit/ccvm bash
```

##### 2. Go into `examples/` and run `ccvm_boxqp_plot.py`

````
cd examples && python ccvm_boxqp_plot.py
````

##### 3. View generated plots

<p align="center">
    <img src="ccvmplotlib/images/tts_plot_example.png" width="250" >
    <img src="ccvmplotlib/images/success_prob_plot_example.png" width="250">
</p>


## Usage


##### First, install using `pip`

```
pip install git+https://github.com/1QB-Information-Technologies/ccvm/
```

TODO: publish on pip, update above to `pip install ccvm`


### Solve a BoxQP problem

##### 1. Add imports

```python
from problem_classes.boxqp.problem_instance import ProblemInstance
from ccvm.solvers.dl_solver import DLSolver
```

##### 2. Define a Solver

```python
solver = DLSolver(device="cpu", batch_size=100)  # or "cuda"
solver.parameter_key = {
    20: {"p": 2.0, "lr": 0.005, "iter": 15000, "nr": 10},
}
```

##### 3. Load in Problem Instance

```python
boxqp_instance = ProblemInstance(
    instance_type="test",
    file_path="./examples/test_instances/test020-100-10.in",
    device=solver.device,
)
boxqp_instance.scale_coefs(solver.get_scaling_factor(boxqp_instance.q))
```

##### 4. Solve

```python
solution = solver.solve(
    instance=boxqp_instance,
    post_processor=None,
)

solution.optimal_value
# 799.560976

solution.solve_time
# 2.494919538497925
```


## Docs

Find our [documentation here](https://urban-chainsaw-9k39nm4.pages.github.io/index.html).

* TODO: Update with public link

Some additional quick links:
- Problem Definition: [BoxQP Problem Class](problem_classes/README.md)
- Plotting Library: [ccvmplotlib](ccvmplotlib/README.md)


## Contributing

We love pull requests and discussing novel ideas. Check out our [contribution guide](CONTRIBUTING.md) and feel free to improve CCVM. For major changes, please open an issue first to discuss what you would like to change.

Thanks for considering contributing to our project! We appreciate your help and support.


## References

This repo contains architectures and algorithms as discussed in the paper ["Non-convex Quadratic Programming Using Coherent Optical Networks"](https://arxiv.org/abs/2209.04415) by Farhad Khosravi, Ugur Yildiz, Artur Scherer, and Pooya Ronagh.


## License

[APGLv3](https://github.com/1QB-Information-Technologies/ccvm/blob/main/LICENSE)
