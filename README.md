# NFourSID

Implementation of the N4SID algorithm for subspace identification [1], together with Kalman filtering and state-space
models.

State-space models are versatile models for representing multi-dimensional timeseries.
As an example, the ARMAX(_p_, _q_, _r_)-models - AutoRegressive MovingAverage with eXogenous input -
are included in the representation of state-space models.
By extension, ARMA-, AR- and MA-models can be described, too.
The numerical implementations are based on [2].

## Installation
Releases are made available on PyPi.
The recommended installation method is via `pip`:

```python
pip install nfoursid
```

For a development setup, the requirements are in `dev-requirements.txt`.
Subsequently, this repo can be locally `pip`-installed.

## Documentation and code example
Documentation is provided [here](https://nfoursid.readthedocs.io/en/latest/).
An example Jupyter notebook is provided [here](https://github.com/spmvg/nfoursid/blob/master/examples/Overview.ipynb).

## References

1. Van Overschee, Peter, and Bart De Moor. "N4SID: Subspace algorithms for the identification of combined
   deterministic-stochastic systems." Automatica 30.1 (1994): 75-93.
2. Verhaegen, Michel, and Vincent Verdult. _Filtering and system identification: a least squares approach._
   Cambridge university press, 2007.
