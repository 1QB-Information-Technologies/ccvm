#  Problem Classes

##  BoxQP 

The Box-constrained Quadratic Programming (BoxQP) problem is a non-convex continuous-variable optimization problem written formally as:

$$
\begin{split}
\text{maximize} & \quad f(x) = \frac{1}{2} \sum_{i,j=1}^N Q_{ij} x_i x_j  + \sum_{I=1}^N V_i x_i \\
\text{subject to} & \quad l_i \le x_i \le u_i \quad \forall i \in \{1,\cdots, N\}.
\end{split}
$$

Here, $f(x)$ is the objective function described by the $N\times N$ $Q$ matrix and the $V$ vector of size $N$. The second line describes the box constraint. Here the default values for limits of the box constraint are $0$ and $1$, i.e. $l_i = 0$ and $u_i = 1$ for all $i$ in $\{1,\cdots,N\}$.


The CCVM receives the `Q` matrix and the `V` vector of the BoxQP problem and implements the derivative of the objective function into the CCVM solver. For implementing the box constraint, the amplitudes of the variable are either clamped between the lower and the upper values of the box constraints, as is the case with the BMF-CCVM, PMF-CCVM, MF-CCVM, Langevin, and pumped-Langevin solvers, or the saturation process in the CCVM is used to implement this constraints as is the case with the DL-CCVM solver. The default box constraint limits are and for all variables.
