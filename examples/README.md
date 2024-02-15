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

Some randomly generated box-constrained programming problem instances are contained in the `benchmarking_instances` folder.
This collection of instances was created in order to benchmark our simulators against difficult problems, with the results being reported in the paper by Khosravi et al. (2022).
Both the matrix and the vector have been generated using numpy.
The instances are grouped into subfolders according to their problem size, and are labelled with the suffix {s}-{d}-{r}, where s is the size of the problem instance, d is the density (100% for all the provided instances), and s is the seed number used to generate that specific problem instance. For exploratory purposes, the folder `single_problem_instance` contains a single instance that can be specified in order to quickly run the example scripts.
The box constraint for all variables and all problem sizes is assumed to be between 0 and 1.
To ensure the solvers are being properly showcased, hard instances were chosen, where hard here refers to the large number of local minima in each problem's landscape and the high density of each problem matrix. The solution vectors of the provided problem instances contain fractional values between 0 and 1, as opposed to all the solution values being at the box boundaries (0 or 1).


1. The first line of each instance file contains the following information, in this order:
- instance size
- optimum solution (example files used the Gurobi solver to determine these values)
- best solution (example files used the BFGS solver to determine these values)
- whether the solution is optimal (`True` or `False`)
- solution time for Gurobi to solve it
- solution time for BFGS to solve it
- the seed number used for generating the instance in `torch`
- number of fractional values in the solution

2. The second line contains the elements of the vector `V`, which describes the instance.

3. The rest of the lines before the last line hold the elements of the `Q` matrix.

4. The final line contains the vector of the solution to the problem instance.
