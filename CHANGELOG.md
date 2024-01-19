# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Breaking Changes
- Removed the TrustConst post-processor from the codebase due to non-usage. This
  change aims to streamline the code and eliminate unnecessary components.

### Fixed 
- Fixed issue where `s` was being updated incorrectly on each iteration  of `DLSolver._solve()`.

### Added
- Implemented a simple gradient descent post-processing step, as described in the paper.
  - Similar to Langevin dynamics but without noise.
	- Implemented using the Euler method with box constraint imposition at each iteration.
	- Designed to reflect the results outlined in the paper.

### Changed
- Streamlined README by relocating and optimizing architecture diagrams.
- Enhanced post processor clamp function for greater flexibility by replacing
  hard-coded values with user-defined variables (lower_clamp and upper_clamp).

## [1.0.1] - 2023-03-09
### Fixed
- Corrected the readme diagrams missing issue with the PyPI deployment for version 1.0.0.

## [1.0.0] - 2023-12-20
### Breaking Changes
- parameter_key: `lr` parameter has been renamed to `dt` to better reflect the meaning of the parameter, which is the time step of the simulation.
- solve(): The `solve` method has been removed from the solvers and a `__call__` method has been implemented; users must call the solver object directly.

### Added
- LangevinSolver has been added to the package. This solver is a stochastic solver based on the Langevin equation.
- Added a `__call__` method to the solvers. This allows users to call the solver object directly, rather than having to call the `solve` method.
  - Introduced the `algorithm_parameters` control keyword when calling the solver objects. This allows users to specify a specialized algorithm for the solver to use and provide the parameters of this algorithm to the solver.
  - Algorithm parameter classes have been added to `ccvm_simulators.solvers.algorithms` to allow users to specify the `algorithm_parameters`.

### Changed
- Moved the update code segment for `solution` object from `_solve()` and `_solve_adam()` to `__call__()` in order to reduce code duplication in `LangevinSolver`, `DLSolver`, and `MFSolver`.
- Updated internal data handling that have improved performance and reduced memory usage.
- Updated data structure for new problem instance file:
	-- Additional fields (`best_sol`, `sol_time_bfgs`, and `num_frac_values`) are added so that additional values in the first line can be extracted properly,
	-- A field `solution_vector` is added to extract a list of values from the final line

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

### Documentation
- Added new page to Sphinx documentation, `Coherent Continuous-Variable Machine Simulator - Equations of Motion`, along with subpages specific to:
    - `The Langevin Dynamics Solver`
    - `The Measurement-Feedback Coherent Continuous-Variable Machine`
    - `The Delay-Line Coherent Continuous-Variable Machine`

## [0.1.0] - 2023-01-25
### Added
- Initial release of the project.
