Stochastic Differential Equations for the Delay-Line Coherent Continuous-Variable Machine
========================================================

The dynamics describing a coherent continuous-variable machine with a delay-line coupling scheme (DL-CCVM) are described by the following system of stochastic differential equations (SDE):

.. math::

   dc_i &= \Big[\big(-1+p(t)-c_i^2-s_i^2\big)c_i - \partial_i f(\mathbf{c})\Big]dt + \frac{r(t)}{A_s}\sqrt{c_i^2+s_i^2+\frac{1}{2}} dW_{i}^{c} \\
   ds_i &= \Big[\big(-1-p(t)-c_i^2-s_i^2\big)s_i-\partial_i f(\mathbf{s})\Big]dt+\frac{1}{r(t) A_s}\sqrt{c_i^2+s_i^2+\frac{1}{2}}dW_{i}^{s}

.. math::

   c_i(0)=s_i(0)=0, \quad \forall i=1,\ldots,N

where :math:`p(t) = p_0 t/T` is the amplitude of the pump field amplifying the optical pulses normalized to the photon loss rate and :math:`\partial_i f(\mathbf{y}) = \frac{\partial f(\mathbf{y})}{\partial y_i}` is the gradient of the continuous objective function, with :math:`\mathbf{y}\in\{\mathbf{c}, \mathbf{s}\}`.

In our CCVM simulator, :math:`f` refers to the quadratic objective function of a given optimization problem. In the case of the box-constrained quadratic programming (BoxQP) problem, it is given by

.. math::

   f(\mathbf{x}) = \frac{1}{2}\sum_{i,j=1}^{N}Q_{ij}x_i x_j+\sum_{i=1}^N V_{i}x_i,

subjected to the box constraints :math:`\ell_i\leq x_i \leq u_i`. The default limits of the box constraint are :math:`\ell_i=0, \; u_i=1`. To encode the variable of the BoxQP problem :math:`x_i` into the optical field amplitudes :math:`y_i`, we have performed the following change of variables:

.. math::

   x_i := \left(\frac{1}{2}\Big(\frac{y_i}{s}+1\Big)\big(u_i-\ell_i\big)+\ell_i\right),\;\;\;\;\forall \mathbf{y}\in\big\{\mathbf{c}, \mathbf{s}, \mathbf{\tilde\mu}\big\}.

:math:`\mathbf{Q}` is a real symmetric matrix and :math:`\mathbf{V}` is a real vector describing the BoxQP problem. The variance of the noise in this system is controlled by scheduling the variable :math:`r(t)` via the injection of a squeezed state into the open port of the output coupler. An exponential function of the form

.. math::

   r(t)=r_0 \exp\Big(-\beta\frac{t}{T}\Big),

has been used as the scheduling function. Here :math:`T=n_\mathrm{iter}\times N\times T_\mathrm{pulse}` is the evolution time for a single round trip. :math:`T_\text{pulse}=100\text{ps}` by default.

Example script to solve BoxQP problem using DL-CCVM solver
-------------------------------------------------------------------

If you run the demo script by :code:`$ python ccvm/examples/ccvm_boxqp_dl.py` it will simulate the dynamics of DL-CCVM for an example problem instance with a sample outcome:

.. code-block:: python

    Solution(
        problem_size=20, batch_size=1000, iterations=15000, ...,
        solve_time=15.929, optimal_value=152.602, best_value=147.96,
        solution_performance={
            'optimal': 0.265, 'one_percent': 0.457, 'two_percent': 0.544, 'three_percent': 0.715,
            'four_percent': 0.995, 'five_percent': 0.999, 'ten_percent': 1.0
        }
    )
