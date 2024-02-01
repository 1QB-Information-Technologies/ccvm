Stochastic Differential Equations for Langevin Dynamics
=============================================================

General Langevin dynamics can be described by the following stochastic differential equation (SDE):

.. math::

    d c_{i} = -\lambda\partial_{i} f(\mathbf{c}) + \sigma dW_{i}\;\;\text{with}\;\;c_{i}(0)=0\;\;\forall i=1,\ldots,N

where :math:`\lambda` and :math:`\sigma` are hyperparameters controlling the strengths of the drift and diffusion terms, respectively. We have implemented this solver as a classical solver implemented solely on classical computers, that is, there is no optical simulation of Langevin dynamics in our ``ccvm_simulators`` package.

Example script for solving a box-constrained quadratic programming problem using the Langevin solver
-------------------------------------------------------------------

The demo script, `ccvm_boxqp_langevin.py`, in the examples folder simulates the Langevin dynamics for an example problem instance. It can be run from the project's root directory using the following command in the terminal:

:code:`$ python ccvm/examples/ccvm_boxqp_langevin.py`

The script will print the solution, similar to the example output below.

.. code-block:: python

   Solution(
      problem_size=20,
      batch_size=1000,
      iterations=15000,
      ...,
      solve_time=5.026,
      optimal_value=152.602,
      best_value=147.960,
      solution_performance={
         'optimal': 0.0,
         'one_percent': 0.0,
         'two_percent': 0.0,
         'three_percent': 0.0,
         'four_percent': 0.999,
         'five_percent': 0.999,
         'ten_percent': 0.999
      }
   )
