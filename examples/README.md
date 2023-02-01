# Examples

- `ccvm_boxqp_plot.py`: Solve BoxQP using CCVM w/ plot
- `ccvm_boxqp.py`: Solve BoxQP using CCVM w/o plot
- `benchmarking.py`: Benchmarking CCVM w/ Langevin (TODO)

### Problem Instances

The following two folders contain our example instances:
- `tuning_instances`
- `test_instances`

The instances have the same properties except that different seed numbers have been used to generate them. The instances in the tuning folder are used to tune the parameters of the demonstrated solvers. Then we use those parameters to solve the test instances to make sure the performance of the solver is independent of the particular random instances generated.

The first line of each instance file contains the following:
- instance size
- optimum solution
- is the solution optimal (True or False)
- the seed number used for generating the instance in torch

The second line of the instance file contains the elements of the vector V which describes the instance. The rest of the lines contain the Q matrix.