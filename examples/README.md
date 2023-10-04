# Examples

### Demo Scripts
The following scripts solve BoxQP problem using CCVM Simulators.
There are currently two methods available for each solver class [DL-CCVM, MF-CCVM, Langevin] which are **original** (default) and **Adam**.

- `ccvm_boxqp_plot.py`: Solves DL-CCVM with time-to-solution (TTS) plotting
- `ccvm_boxqp_dl.py`: Solves DL-CCVM using the Adam algorithm, without plotting. The original solver can be used by removing the `algorithm_parameters` argument.
- `ccvm_boxqp_mf.py`: Solves MF-CCVM using the Adam algorithm, without plotting. The original solver can be used by removing the `algorithm_parameters` argument.
- `langevin_boxqp.py`: Solves Langevin equation using the Adam algorithm, without plotting. The original solver can be used by removing the `algorithm_parameters` argument.
- `benchmarking_studies.py`: Benchmark DL-CCVM against a Langevin solver (**to be implemented**)



### Example Problem Instances

The following folders contain our example problem instances:
- `tuning_instances`
- `test_instances`

Identical problem instances across folders have the same properties except that different seed numbers were used to generate them. The instances in the `tuning_instances` folder are used to tune the parameters of the solvers in the package. We then use the parameters to solve the test instances to ensure the performance of the solver is independent of the particular random problem instances generated.

The first line of each instance file contains the following information:
- the problem instance size
- the optimum solution
- whether the solution is optimal (True or False)
- the seed number used for generating the problem instance in `torch`

The second line of the instance file contains the elements of the vector $V$, which defines the instance. The rest of the lines in the file contain the matrix $Q$.
