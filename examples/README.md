# Examples

### Demo Scripts
The following scripts solve BoxQP problem using CCVM Simulator
- `ccvm_boxqp_plot.py`: Solves DL-CCVM with time-to-solution (TTS) plotting
- `ccvm_boxqp.py`: Solves DL-CCVM without plotting
- `ccvm_boxqp_mf.py`: Solves MF-CCVM without plotting
- `langevin_boxqp.py`: Solve Langevin equation without plotting
- `benchmarking_studies.py`: Benchmark DL-CCVM against a Langevin solver (**?to be implemented?**)

### Demo Scripts with ADAM algorithm
The following scripts solve BoxQP problem using CCVM simulator with ADAM algorithm
- `ccvm_boxqp_adam_dl.py`: Using a DL-CCVM simulator
- `ccvm_boxqp_adam_mf.py`: Using MF-CCVM simulator
- `langevin_boxqp_adam.py`: Using Langevin simulator


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