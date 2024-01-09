Stochastic Differential Equations (SDE) for Langevin dynamics
=============================================================

The general Langevin dynamics can be described by the following SDE:

.. math::

    d c_{i} = -\lambda\partial_{i} f(\mathbf{c}) + \sigma dW_{i}\;\;\text{with}\;\;c_{i}(0)=0\;\;\forall i=1,\ldots,N

where :math:`\lambda` and :math:`\sigma` are hyper-parameters controlling the strengths of the drift and diffusion terms, respectively. We have implemented this solver as a classical solver implemented solely on classical computer, that is, there is no optical simulation of the Langevin dynamics in our ``ccvm`` package.

Example script to solve BoxQP problem using Langevin solver
-------------------------------------------------------------------

On the command line, if you run the demo script by :code:`$ python ccvm/examples/langevin_boxqp.py`, it will simulate Langevin dynamics for an example problem instance and then the outcome is displayed as:

.. code-block:: python

    Solution(
       problem_size=20, batch_size=1000, iterations=15000, ...,
       solve_time=5.026, optimal_value=152.602, best_value=147.960,  
       solution_performance={
          'optimal': 0.0, 'one_percent': 0.0, 'two_percent': 0.0, 'three_percent': 0.0, 
          'four_percent': 0.999, 'five_percent': 0.999, 'ten_percent': 0.999
       }
    )
