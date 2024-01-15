# Post Processor
The algorithms in the following modules are used to perform optimization on the
output of the solver:

- `adam.py`: Adam method from PyTorch.
- `asgd.py`:  ASGD from PyTorch.
- `bfgs.py`: BFGS from SciPy.
- `lbfgs.py`:  LBFGS from SciPy.
- `trust_constr.py`: Trust-region algorithm from SciPy.
- `grad_descent.py`: Gradient descent method using Euler-Maruyama method with a constraint enforced at each step.


### Post processor hierarchy
The diagram delineates the relationships between various post processors and
their connection to the abstract post processor class.

<p align="center">
    <img src="../../diagrams/post_processor_ hierarchy.png">
</p>
