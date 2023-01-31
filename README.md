# Coherent Continous-Variable Machine

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Maintainer](https://img.shields.io/badge/Maintainer-1QBit-blue)](http://1qbit.com/)
[![Paper](https://img.shields.io/badge/Paper-arxiv-red)](https://arxiv.org/abs/2209.04415)

The Coherent Continous-Variable Machine (CCVM) is a novel quantum optical network architecture built on NTT's Coherent Ising Machine (CIM). Here we demonstrate the application of CCVM on the Box-Constrained Quadratic Programming (BoxQP) problem by mapping the variables of problems into the random variables of CCVM.

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

1. Run container from anywhere

`docker run -it -v $(pwd):/workspace/examples/plots quay.io/1qbit/ccvm bash`

2. Go into `examples/` and run `ccvm-boxqp.py`

`cd examples && python ccvm-boxqp.py`

3. View generated plots

<p align="center">
    <img src="ccvmplotlib/images/tts_plot_example.png" width="250" >
    <img src="ccvmplotlib/images/success_prob_plot_example.png" width="250">
</p>


## Usage (TODO)

  

#### Install dependenices

Run the following command to install dependencies:

`pip install -r requirements.txt`


#### Solve a BoxQP problem (TODO)

```
define boxqp problem

initialize solver

solve()

plot results
```


## Docs (TODO)

Find our [documentation here](ccvm.readthedocs.io).

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
