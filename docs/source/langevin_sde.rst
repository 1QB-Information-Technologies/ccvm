Stochastic Differential Equations (SDE) for Langevin dynamics
=============================================================

The general Langevin dynamics can be described by the following SDE:

.. math::

    d c_{i} = -\lambda\partial_{i} f(\mathbf{c}) + \sigma dW_{i}\;\;\text{with}\;\;c_{i}(0)=0\;\;\forall i=1,\ldots,N

where :math:`\lambda` and :math:`\sigma` are hyper-parameters controlling the strengths of the drift and diffusion terms, respectively. We have implemented this solver as a classical solver implemented solely on classical computer, that is, there is no optical simulation of the Langevin dynamics in our ``ccvm`` package.
