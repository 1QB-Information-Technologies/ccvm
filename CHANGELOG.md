# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Breaking Changes
- parameter_key: `lr` parameter has been renamed to `dt` to better reflect the meaning of the parameter, which is the time step of the simulation.
- solve(): The `solve` method has been removed from the solvers and a `__call__` method has been implemented; users must call the solver object directly.

### Added
- LangevinSolver has been added to the package. This solver is a stochastic solver based on the Langevin equation.
- Introduced `solve_type` control keyword when calling the solver objects. If it is deliberately set to 'Adam' then the ADAM algorithm method will be executed with relevant hyperparameters. Otherwise, its default value (i.e. solve_type=None) refers to the call for the original solve method. The Adam method consist of three hyperparameters in which beta1 and beta2 are exponential decay rates for the moment estimates and
alpha is the step size. Please refer to [the paper](https://doi.org/10.48550/arXiv.1412.6980) for more information about Adam algorithm and its hyperparameters.

### Changed
- Updated internal data handling that have improved performance and reduced memory usage.

## [0.1.2] - 2023-03-09
### Fixed
- Corrected an issue with the PyPI deployment for version 0.1.1. This version is the first available on PyPI.

## [0.1.1] - 2023-03-09
### Added
- Improved and expanded on documentation, including docstrings and READMEs.
- Increased test coverage, especially with respect to the Problem and Solution classes.

### Changed
- Updated the MF Solver and DL Solver algorithms to better account for the enforced saturation value (S).
- Improved and fixed some issues in the DL and MF solvers to improve performance.
- Modified the gradient of the objective function in the MFSolver to implement a linear function of variables by default. This enhances the performance of the solver.
- Updated the package name from ccvm to ccvm-simulators.

### Notes
- Version was intended for PyPI release but encountered deployment issues.


## [0.1.0] - 2023-01-25
### Added
- Initial release of the project.
