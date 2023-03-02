# Examples

### Demo Scripts

- `ccvm_boxqp_plot.py`: Solve a BoxQP problem using a CCVM simulator, w/ time-to-solution (TTS) plotting
- `ccvm_boxqp.py`: Solve a BoxQP problem using a CCVM simulator, w/o plotting
- `benchmarking_studies.py`: Benchmark a CCVM simulator against a Langevin solver (to be implemented)

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