import pytest
import numpy as np
from symmer.evolution.gate_library import *
from symmer.operators import QuantumState as qs

@pytest.mark.parametrize(
    "gate, state_in, state_out", 
    [
        (I(1),     qs([[0]]), qs([[0]])),
        (I(1),     qs([[1]]), qs([[1]])),
        (X(1,0),   qs([[0]]), qs([[1]])), 
        (X(1,0),   qs([[1]]), qs([[0]])), 
        (Y(1,0),   qs([[0]]), qs([[1]])*1j), 
        (Y(1,0),   qs([[1]]), qs([[0]])*-1j), 
        (Z(1,0),   qs([[0]]), qs([[0]])), 
        (Z(1,0),   qs([[1]]), qs([[1]])*-1),
        (S(1,0),   qs([[0]]), qs([[0]])), 
        (S(1,0),   qs([[1]]), qs([[1]])*1j),
        (Had(1,0), qs([[0]]), qs([[0],[1]], [1/np.sqrt(2),1/np.sqrt(2)])),
        (Had(1,0), qs([[1]]), qs([[0],[1]], [1/np.sqrt(2),-1/np.sqrt(2)])),
        (RX(1,0,np.pi/3), qs([[0]]), qs([[0],[1]], [np.cos(np.pi/6), 1j*np.sin(np.pi/6)])),
        (RX(1,0,np.pi/3), qs([[1]]), qs([[1],[0]], [np.cos(np.pi/6), 1j*np.sin(np.pi/6)])),
        (RY(1,0,np.pi/3), qs([[0]]), qs([[0],[1]], [np.cos(np.pi/6), -np.sin(np.pi/6)])),
        (RY(1,0,np.pi/3), qs([[1]]), qs([[1],[0]], [np.cos(np.pi/6), +np.sin(np.pi/6)])),
        (RZ(1,0,np.pi/3), qs([[0]]), qs([[0]], [np.exp(+1j*np.pi/6)])),
        (RZ(1,0,np.pi/3), qs([[1]]), qs([[1]], [np.exp(-1j*np.pi/6)])),
        (U1(1,0,np.pi/3), qs([[0]]), qs([[0]], [np.exp(+1j*np.pi/6)])*np.exp(1j*np.pi/6)),
        (U1(1,0,np.pi/3), qs([[1]]), qs([[1]], [np.exp(-1j*np.pi/6)])*np.exp(1j*np.pi/6)),
    ]
)
def test_single_qubit_gates(gate, state_in, state_out):
    assert gate * state_in == state_out

@pytest.mark.parametrize(
    "gate, state_in, state_out", 
    [
        (CZ, qs([[0,0]]), qs([[0,0]])),
        (CZ, qs([[0,1]]), qs([[0,1]])),
        (CZ, qs([[1,0]]), qs([[1,0]])),
        (CZ, qs([[1,1]]), qs([[1,1]])*-1),
        (CX, qs([[0,0]]), qs([[0,0]])),
        (CX, qs([[0,1]]), qs([[0,1]])),
        (CX, qs([[1,0]]), qs([[1,1]])),
        (CX, qs([[1,1]]), qs([[1,0]])),
    ]
)
def test_two_qubit_gates(gate, state_in, state_out):
    assert gate(2,0,1) * state_in == state_out