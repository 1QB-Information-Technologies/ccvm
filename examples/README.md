# Examples

### Demo Scripts
The following scripts solve BoxQP problem using CCVM Simulators.
There are currently two methods available for each solver class [DL-CCVM, MF-CCVM, Langevin] which are **original** (default) and **Adam**.

- `ccvm_boxqp_plot.py`: Solves DL-CCVM with time-to-solution (TTS) plotting
- `ccvm_boxqp_dl.py`: Solves DL-CCVM using the Adam algorithm, without plotting. The original solver can be used by removing the `algorithm_parameters` argument.
- `ccvm_boxqp_mf.py`: Solves MF-CCVM using the Adam algorithm, without plotting. The original solver can be used by removing the `algorithm_parameters` argument.
- `langevin_boxqp.py`: Solves Langevin equation using the Adam algorithm, without plotting. The original solver can be used by removing the `algorithm_parameters` argument.
- `pumped_langevin_boxqp.py`: Solves pumped Langevin equation by either Adam algorithm or original solver with a use of `algorithm_parameters` argument.
- `benchmarking_studies.py`: Benchmark DL-CCVM against a Langevin solver (**to be implemented**)



### Example Problem Instances

The following folders contain our example problem instances:
- `tuning_instances`
- `test_instances`

Identical problem instances across folders have the same properties except that different seed numbers were used to generate them. The instances in the `tuning_instances` folder are used to tune the parameters of the solvers in the package. We then use the parameters to solve the test instances to ensure the performance of the solver is independent of the particular random problem instances generated.

The first line of each instance file contains the following information, in this order:
- instance size
- optimum solution (example files used the Gurobi solver to determine these values)
- best solution (example files used the BFGS solver to determine these values)
- whether the solution is optimal (`True` or `False`)
- solution time for Gurobi to solve it 
- solution time for BFGS to solve it
- the seed number used for generating the instance in `torch`.
- number of fractional values in the solution

The second line contains the elements of the vector `V`, which describes the instance.

The rest of the lines before the last line hold the elements of the `Q` matrix.

The final line contains the vector of the solution to the problem instance