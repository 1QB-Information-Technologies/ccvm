The Pumped Langevin Dynamics Solver
=====================================

Stochastic differential equations
----------------------------------

Pumped Langevin dynamics, which is a modification of a typical Langevin dynamics (Khosravi et al 2022), can be described by the stochastic differential equations (SDE)
.. math::

    d c_{i} = \Big[\big(-1+p-c_i^2\big)c_i-\lambda\partial_{i} f(\mathbf{c}) \Big]dt + \sigma dW_{i}\;\;\text{with}\;\;c_{i}(0)=0\;\;\forall i=1,\ldots,N

where, :math:`p(t) = p_0 t/T` and :math:`\lambda` and :math:`\sigma` are hyperparameters controlling the strength of the gradient and the diffusion terms, respectively. Apart from typical Langevin equation, the preceding equation is augmented with additional term :math:`
(-1+p-c_i^2\big)c_i` to represent three physical processes (Khosravi et al 2022). Similar to the Langevin solver, we have implemented a solver for the pumped-Langevin solver.

Note that the total evaluation time is arbitrary and can be defined as :math:`T = n_\text{iter}\times dt`. 

We have developed this solver as a classical solver implemented solely on classical computers, that is, there is no optical simulation of pumped Langevin dynamics in our `ccvm_simulators` package.

Example script for solving a box-constrained quadratic programming problem using the Pumped Langevin solver
-------------------------------------------------------------------------------------------------------------

The demo script, `pumped_langevin_boxqp.py`, in the examples folder simulates the Langevin dynamics for an example problem instance using the pumped Langevin solver. It can be executed from the project's root directory using the following command in the terminal:

:code:`$ python ccvm/examples/pumped_langevin_boxqp.py`

The script will print the solution, similar to the example output below.

.. code-block:: python

   Solution(
      problem_size=20,
      batch_size=1000,
      iterations=15000,
      ...,
      solve_time=5.422,
      optimal_value=130.714,
      best_value=120.533,
      solution_performance={
         'optimal': 0.004,
         'one_percent': 0.394,
         'two_percent': 0.908,
         'three_percent': 0.991,
         'four_percent': 1.0,
         'five_percent': 1.0,
         'ten_percent': 1.0
      }, 
	best_objective_value=130.653
   )
