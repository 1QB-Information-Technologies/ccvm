Stochastic Differential Equations for the Measurement-Feedback Coherent Continuous-Variable Machine
==================================================================================================================

The dynamics describing a coherent continuous-variable machine with a measurement-feedback coupling scheme (MF-CCVM) are represented by the following set of stochastic differential equations (SDE). In this model, the degenerate optical parametric oscillator (DOPO) pulses are assumed to maintain Gaussian distribution throughout the time evolution with the mean values :math:`\mu` and variances :math:`\sigma`:

.. math::

   \begin{align*}
   d\mu_i &= \Bigg\{\Big[p(t)-g^2\mu_i^2\Big]\mu_i -\lambda \partial_i f(\tilde\mu)\Bigg\}dt+\sqrt{j(t)}\Bigg(\sigma_i-\frac{1}{2}\Bigg)dW_i\\
   \frac{d\sigma_i}{dt}&=2\Bigg[p(t)-3g^2\mu_i^2\Bigg]\sigma_i-2j(t)\Bigg(\sigma_i-\frac{1}{2}\Bigg)^2+\Bigg[\Big(1+j(t)\Big)+2g^2\mu_i^2\Bigg]
   \end{align*}

.. math::

   \mathbf{\mu}(0)=\mathbf{0},\;\;\mathbf{\sigma}(0)=\frac{\mathbf{1}}{2}

where

.. math::

   j(t) = j_0 \exp\Big(-\alpha \frac{t}{T}\Big)

is the normalized continuous measurement strength scheduled using an exponential function, :math:`\lambda` is a hyper-parameter controlling the strength of the feedback signal (coupling term), :math:`g` is the normalized second-order nonlinearity coefficient of the nonlinear crystal, and

.. math::

   \tilde \mu = \mu +\frac{1}{\sqrt{4j(t)}}\frac{dW}{dt}

is the vector of measured mean field amplitudes as a function of the mean-field amplitudes within the cavity :math:`\mu` and :math:`dW`, the vector of Wiener processes.

Example script for solving a box-constrained quadratic programming problem using the MF-CCVM solver
--------------------------------------------------------------------

The demo script, `ccvm_boxqp_mf.py`, in the examples folder simulates the dynamics of the MF-CCVM for an example problem instance.
It can be run from the project's root directory using the following command in the terminal:

:code:`$ python ccvm/examples/ccvm_boxqp_mf.py`

The script will print the solution, similar to the example output below.

.. code-block:: python

    Solution(
        problem_size=20,
        batch_size=1000,
        iterations=15000,
        ...,
        solve_time=10.338,
        optimal_value=152.602,
        best_value=147.960,
        solution_performance={
            'optimal': 0.666,
            'one_percent': 0.666,
            'two_percent': 0.666,
            'three_percent': 0.684,
            'four_percent': 0.994,
            'five_percent': 0.994,
            'ten_percent': 0.995
        }
    )
