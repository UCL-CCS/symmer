![symmer](https://github.com/UCL-CCS/symmer/blob/main/images/symmer_logo.png)

[![Continuous_Integration](https://github.com/UCL-CCS/symmer/actions/workflows/pull_request.yaml/badge.svg)](https://github.com/UCL-CCS/symmer/actions/workflows/pull_request.yaml)
[![codecov](https://codecov.io/gh/UCL-CCS/symmer/branch/main/graph/badge.svg?token=PZzJNZuEEW)](https://codecov.io/gh/UCL-CCS/symmer)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)
# Symmer

A Python package for reducing the quantum resource requirement of your problems, making them more NISQ-friendly!

## Installation
To install from the root of the project run:

```
pip install .
```

## Basic usage
For basic usage see [notebooks](https://github.com/UCL-CCS/symmer/tree/main/notebooks)

## Included in symmer:
Qubit reduction techniques such as [tapering](https://arxiv.org/abs/1701.08213) and [Contextual-Subspace VQE](https://doi.org/10.22331/q-2021-05-14-456) are effected by the underlying [stabilizer subspace projection mechanism](https://arxiv.org/abs/2204.02150); such methods may be differentiated by the approach taken to selecting the stabilizers one wishes to project over. 

`.symplectic` contains the following classes (in resolution order):
- [`PauliwordOp`](https://github.com/UCL-CCS/symmer/tree/main/symmer/symplectic/base.py) for representing general Pauli operators.
- [`QuantumState`](https://github.com/UCL-CCS/symmer/tree/main/symmer/symplectic/base.py) for representing quantum statevectors.
- [`ObservableOp`](https://github.com/UCL-CCS/symmer/tree/main/symmer/symplectic/observable_op.py) for representing Hermitian operators, including expectation values and VQE functionality.
- [`AnsatzOp`](https://github.com/UCL-CCS/symmer/tree/main/symmer/symplectic/ansatz_op.py) is input into ObservableOp.VQE and contains a method for converting excitation terms to a quantum circuit.
- [`StabilizerOp`](https://github.com/UCL-CCS/symmer/tree/main/symmer/symplectic/stabilizer_op.py) represents algebraically independent sets of Pauli operators for stabilizer manipulation/projections.
- [`AnticommutingOp`](https://github.com/UCL-CCS/symmer/tree/main/symmer/symplectic/anticommuting_op.py) represents sets of anticommuting Pauli operators for the purposes of Unitary Partitioning and Linear Combination of Unitaries as in [this](https://arxiv.org/abs/2207.03451) paper.
- [`MajoranaOp`](https://github.com/UCL-CCS/symmer/tree/main/symmer/symplectic/majorana_op.py) represents operators in the Majorana basis instead of Paulis, though the underlying symplectic structure is analogous.

`.projection` contains stabilizer subspace projection classes (in resolution order):
- [`S3_projection`](https://github.com/UCL-CCS/symmer/tree/main/symmer/projection/base.py) for rotating a StabilizerOp onto some basis of single-qubit Pauli operators via Clifford operations and projecting into the corresponding stabilizer subspace.

- [`QubitTapering`](https://github.com/UCL-CCS/symmer/tree/main/symmer/projection/qubit_tapering.py) 
  - Performs the [qubit tapering](https://arxiv.org/abs/1701.08213) technique, exploiting $\mathbb{Z}_2$ symmetries to reduce the number of qubits in the input Hamiltonian while preserving the ground state energy _exactly_.
  - The stablizers are chosen to be an independent generating set of a Hamiltonian symmetry.
  
- [`CS_VQE`](https://github.com/UCL-CCS/symmer/tree/main/symmer/projection/cs_vqe.py) 
  - Performs [Contextual-Subspace VQE](https://quantum-journal.org/papers/q-2021-05-14-456/), allowing one to specify precisely how many qubits they would like in the output Hamiltonian. Despite this process incurring some systematic error, it is possible to retain sufficient information to permit high precision simulations at a significant reduction in quantum resource. This is the updated approach to [ContextualSubspaceVQE](https://github.com/wmkirby1/ContextualSubspaceVQE).
  - Here, the stabilizers are taken to be an independent generating set of a sub-Hamiltonian symmetry (defined by a noncontextual subset of terms) with an additional contribution encapsulating the remaining anticommuting terms therein.

## Performance

Why should you use Symmer? It has been designed for high efficiency when manipulating large Pauli operators -- addition, multiplication, Clifford/general rotations, commutativity/contextuality checks, symmetry basis identification, basis reconstruction and subspace projections have all been reformulated in the symplectic representation and implemented carefully to avoid unnecessary operations and redundancy. But don't just take our word for it, see these benchmarks against various popular quantum computing packages:

Single Pauli Multiplication |  Squaring Linear Combinations
:--------------------------:|:-----------------------------:
![](https://github.com/UCL-CCS/symmer/blob/main/images/single_pauli_mult.png) | ![](https://github.com/UCL-CCS/symmer/blob/main/images/squaring_100q_linear_combs.png)
Runtime for phased multiplication of single Pauli operators $P,Q \in \mathcal{P}_N$ with $N \leq 10^8$. Symmer and Qiskit are comparable throughout, with the latter slightly favourable for large numbers of qubits; Stim excels here as it is highly optimized for such operations. | Squaring 100-qubit Pauli operators with increasing numbers of terms. This benchmark is particularly challenging as it probes the efficiency of multiplication, addition and the subsequent collection of like-terms.

Clifford Rotations          |  General Rotations
:--------------------------:|:-----------------------------:
![](https://github.com/UCL-CCS/symmer/blob/main/images/100q_clifford_rotations.png) | ![](https://github.com/UCL-CCS/symmer/blob/main/images/100q_general_rotations.png)
Clifford rotation of 100-qubit Pauli operators with increasing numbers of terms; this tests commutativity checks and multiplication. Symmer has been optimized with this in mind since it is one of the core operations necessary for stabilizer subspace projection techniques. | General (non-Clifford) rotations of 100-qubit Pauli operators with increasing numbers of terms; this tests commutativity checks and multiplication, as well as addition and the subsequent cleanup operation. This is relevant for performing [unitary partitioning](https://arxiv.org/abs/2207.03451) in [CS-VQE](https://quantum-journal.org/papers/q-2021-05-14-456/).

### To conclude, what can Symmer do on a standard i5 laptop in just _one_ second?
- Multiply two 100,000,000-qubit Pauli terms together.
- Square a 100-qubit Pauli operator with 1,000 terms, involving a cleanup procedure over 1,000,000 cross terms.
- Perform a unitary rotation of a 100-qubit Pauli operator with 1,000,000 terms.

All this allows us to approach significantly larger systems than was previously possible, including those exceeding the realm of classical tractibility.

## How to cite

When you use in a publication or other work, please cite as:

> William M. Kirby, Andrew Tranter, and Peter J. Love, *Contextual Subspace Variational Quantum Eigensolver*, [Quantum 5, 456](https://doi.org/10.22331/q-2021-05-14-456) (2021).
> Tim J. Weaving, Alexis Ralli, William M. Kirby, Andrew Tranter, Peter J. Love, and Peter V. Coveney, *A stabilizer framework for Contextual Subspace VQE and the noncontextual projection ansatz*, arxiv preprint (2022), [arxiv:2204.02150](https://arxiv.org/abs/2204.02150).
> Alexis Ralli, Tim Weaving, Andrew Tranter, William M. Kirby, Peter J. Love, and Peter V. Coveney, *Unitary Partitioning and the Contextual Subspace Variational Quantum Eigensolver*, arxiv preprint (2022), [arxiv:2207.03451](https://arxiv.org/abs/2207.03451).
