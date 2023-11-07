Stochastic Differential Equations (SDE) for the Measurement-Feedback Coherent Continuous Variable Machine (MF-CCVM)
==================================================================================================================

The dynamics describing a CCVM with a measurement-feedback (MF-CCVM) coupling scheme is represented by the following set of SDEs. In this model, the DOPO pulses are assumed to maintain Gaussian distribution throughout the time evolution with the mean values :math:`\mu` and variances :math:`\sigma`:

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
