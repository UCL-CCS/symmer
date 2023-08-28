.. Symmer documentation master file, created by
   sphinx-quickstart on Wed Aug 23 14:45:04 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./_static/symmer_logo.png
  :width: 800px
Welcome to Symmer's documentation!
==================================
.. image:: https://github.com/UCL-CCS/symmer/actions/workflows/pull_request.yaml/badge.svg
  :target: https://github.com/UCL-CCS/symmer/actions/workflows/pull_request.yaml
  :alt: Continuous_Integration
.. image:: https://codecov.io/gh/UCL-CCS/symmer/branch/main/graph/badge.svg?token=PZzJNZuEEW
  :target: https://codecov.io/gh/UCL-CCS/symmer
  :alt: codecov
.. image:: https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge
  :target: http://unitary.fund
  :alt: Unitary Fund

`symmer <https://github.com/UCL-CCS/symmer>`_ is an easy and **fast** python library
for manipulations of *'Pauli operators and Quantum States'*. One of the primary
goals of the library is to reduce the qubit requirements of a quantum simulation problem
by tapering and/or mapping the problem into a contextual subspace. The most useful classes of
library, that form a base for all operations, are:

.. grid:: 2

    .. grid-item-card::  PauliwordOp
        :img-bottom: ./_static/noncontextual-graph.png

        The :mod:`PauliwordOp` class contains tools for working with
        **Pauli operators**. It has a particular focus on:

        * fast operations (multiplication, addition, subtraction)
        * fast expectation value determination
        * output to various other backends:
          `openfermion <https://github.com/quantumlib/OpenFermion>`_ and
          `qiskit <https://github.com/Qiskit/qiskit>`_
        * Find commutation relations between Pauli operators in an efficient manner.

    .. grid-item-card::  QuantumState

        The :mod:`QuantumState` class contains tools for working with
        **'quantum states'**. With this you can:

        * resample a state :math:`10^{15}` times in nanoseconds (useful for bootstrapping)
        * evaluate expectation values with a :mod:`PauliwordOp` in linear time
        * compute various quantities including entanglement measures
        * Implement partial traces
        * Plot quantum state
        * Make examining real NISQ data more streamlined


User Guide
----------

The following guides give a basic introduction to the various parts:

.. toctree::
  :maxdepth: 1

  installation



.. _examples:


Examples
--------

The following guides, generated from the notebooks in `notebooks <https://github.com/UCL-CCS/symmer/tree/main/notebooks>`_,
provide a beginners guide and demonstrate some more advanced features or complete usage:

.. toctree::
  :maxdepth: 2

  ./notebooks/1_Basic_Usage/1.1 PauliwordOp Usage.ipynb

.. toctree::
  :maxdepth: 2

  ./notebooks/1_Basic_Usage/1.2 QuantumState Usage.ipynb

