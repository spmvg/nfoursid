.. NFourSID documentation master file, created by
   sphinx-quickstart on Thu May 13 16:12:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NFourSID
========

.. toctree::
   :maxdepth: 2

   source/modules.rst

Overview
--------

Implementation of the N4SID algorithm for subspace identification [1], together with Kalman filtering and state-space
models.

State-space models are versatile models for representing multi-dimensional timeseries.
As an example, the ARMAX(*p*, *q*)-models - AutoRegressive MovingAverage with eXogenous input - are included in the representation of state-space models.
By extension, ARMA-, AR- and MA-models can be described, too.
The numerical implementations are based on [2].

The state-space model of interest has the following form:

.. math::
    \begin{cases}
        x_{k+1} &= A x_k + B u_k + K e_k \\
        y_k &= C x_k + D u_k + e_k
    \end{cases}

where

- :math:`k \in \mathbb{N}` is the timestep,
- :math:`y_k \in \mathbb{R}^{d_y}` is the output vector with dimension :math:`d_y`,
- :math:`u_k \in \mathbb{R}^{d_u}` is the input vector with dimension :math:`d_u`,
- :math:`x_k \in \mathbb{R}^{d_x}` is the internal state vector with dimension :math:`d_x`,
- :math:`e_k \in \mathbb{R}^{d_x}` is the noise vector with dimension :math:`d_y`,
- :math:`(A, B, C, D)` are system matrices describing time dynamics and input-output coupling,
- :math:`K` is a system matrix describing noise relationships.

Code example
------------
An example Jupyter notebook is provided `here <https://github.com/spmvg/nfoursid/blob/master/examples/Overview.ipynb>`_.

References
----------

1. Van Overschee, Peter, and Bart De Moor. "N4SID: Subspace algorithms for the identification of combined
   deterministic-stochastic systems." Automatica 30.1 (1994): 75-93.
2. Verhaegen, Michel, and Vincent Verdult. *Filtering and system identification: a least squares approach.*
   Cambridge university press, 2007.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
