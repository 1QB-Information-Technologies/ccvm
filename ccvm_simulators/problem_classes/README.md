#  Problem Classes

##  BoxQP

The box-constrained quadratic programming (BoxQP) problem is a non-convex continuous-variable optimization problem written formally as:

$$
\begin{split}
\text{maximize} & \quad f(x) = \frac{1}{2} \sum_{i,j=1}^N Q_{ij} x_i x_j  + \sum_{I=1}^N V_i x_i \\
\text{subject to} & \quad l_i \le x_i \le u_i \quad \forall i \in \{1,\cdots, N\}.
\end{split}
$$

Here, $f(x)$ is the objective function described by the $N\times N$ $Q$ matrix and the $V$ vector of size $N$. The second line describes the box constraint. Here the default values for limits of the box constraint are $0$ and $1$, i.e., $l_i = 0$ and $u_i = 1$ for all $i$ in $\{1,\cdots,N\}$.


The CCVM receives the `Q` matrix and the `V` vector of the BoxQP problem and implements the derivative of the objective function into the CCVM solver. For implementing the box constraint, the amplitudes of the variable are either clamped between the lower and the upper values of the box constraints, as is the case with the BMF-CCVM, PMF-CCVM, MF-CCVM, Langevin, and pumped-Langevin solvers, or the saturation process in the CCVM is used to implement this constraints as is the case with the DL-CCVM solver. The default box constraint limits are and for all variables.

## Problem instance file

In the current folder structure, there is a folder inside the `examples` folder named
`benchmarking_instances`, which contains subfolders of randomly generated instances of various sizes. There is also a subfolder containing a single instance, which can be used as an example problem set to quickly run the scripts in the `examples` folder.
Within each subfolder of `benchmarking_instances`, the contained problems have the same properties except that different seed numbers have been used to generate them.

### Format of the instance file

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

The final line contains list of additional (optional) information in the problem instance.

You can specify a file delimiter of your choice. Then remember to
provide it when initializing the ProblemInstance.

```
boxqp_instance = ProblemInstance(
    instance_type="test",
    file_path="./examples/single_test_instance/test020-100-10.in",
    device=solver.device,
    file_delimiter="YOUR CHOICE",
)
```

If you don't provide it, the default file delimiter is tab (`\t`).
