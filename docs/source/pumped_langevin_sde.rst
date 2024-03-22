The Pumped Langevin Dynamics Solver
=====================================

Stochastic differential equations
----------------------------------

<INTRO HERE>
.. math::

    <EQUATION HERE>

<DESCRIBE EQUATION HERE>

Example script for solving a box-constrained quadratic programming problem using the Pumped Langevin solver
-------------------------------------------------------------------------------------------------------------

The demo script, `pumped_langevin_boxqp.py`, in the examples folder simulates the Langevin dynamics for an example problem instance using the pumped Langevin solver. It can be run from the project's root directory using the following command in the terminal:

:code:`$ python ccvm/examples/pumped_langevin_boxqp.py`

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
