# content of test_file_manager.py

import datetime
import os
import sys

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import logic_engine as le


@pytest.fixture
def single_qubit_gate():
    return le.Gate(name="Hadamard", n_qubits=1, definition="Some definition", qiskit_name="h")


@pytest.fixture
def multi_qubit_gate():
    return le.Gate(name="CNOT", n_qubits=2, definition="Some definition", qiskit_name="cx")


def test_explain_single_qubit_gate(single_qubit_gate):
    assert single_qubit_gate.explain() == "Some definition"


def test_apply_single_qubit_gate(single_qubit_gate):
    qc = QuantumCircuit(2)

    # Test single qubit gate application
    qc = single_qubit_gate.apply(qubit=0, circ=qc)
    assert qc.count_ops()["h"] == 1


def test_draw_single_qubit_gate(single_qubit_gate, multi_qubit_gate):
    # Assuming draw() method is functioning correctly, we check if it saves a file
    assert single_qubit_gate.draw() is not None


def test_compute_state_single_qubit_gate(single_qubit_gate, multi_qubit_gate):
    # Assuming compute_state() method is functioning correctly, we check if it returns a non-empty string
    assert single_qubit_gate.compute_state() != ""


@pytest.fixture
def phase_gate():
    return le.PhaseGate(phase_shift=0.5)


def test_phase_gate_initialization():
    phase_gate = le.PhaseGate(phase_shift=0.5)
    assert phase_gate.phase_shift == 0.5
    assert phase_gate.name == 'phase gate'
    assert phase_gate.alternative_names == ('phase', 'phase shift')
    assert phase_gate.n_qubits == 1
    assert 'The Phase gate is a family of single-qubit operations' in phase_gate.definition
    assert phase_gate.qiskit_name == 'p'


def test_phase_gate_apply(phase_gate):
    qubit = 0
    circ = phase_gate.apply(qubit=qubit)

    assert circ.num_qubits == 1
    assert circ.count_ops()["p"] == 1
    assert circ.data[0][0].params[0] == 0.5


# RX class tests
@pytest.fixture
def rx_gate():
    return le.RX(angle=0.5)


def test_rx_gate_initialization():
    rx_gate = le.RX(angle=0.5)
    assert rx_gate.angle == 0.5
    assert rx_gate.name == 'rotation around x'
    assert rx_gate.alternative_names == ('RX',)
    assert rx_gate.n_qubits == 1
    assert 'The quantum gate Rx(\u03B8) is a single-qubit operation' in rx_gate.definition
    assert rx_gate.qiskit_name == 'rx'


def test_rx_gate_apply(rx_gate):
    qubit = 0
    circ = rx_gate.apply(qubit=qubit)

    assert circ.num_qubits == 1
    assert circ.count_ops()["rx"] == 1
    assert circ.data[0][0].params[0] == 0.5


# RY class tests
@pytest.fixture
def ry_gate():
    return le.RY(angle=0.5)


def test_ry_gate_initialization(ry_gate):
    assert ry_gate.angle == 0.5
    assert ry_gate.name == 'rotation around y'
    assert ry_gate.alternative_names == ('RY',)
    assert ry_gate.n_qubits == 1
    assert 'The quantum gate Ry(\u03B8) is a single-qubit operation' in ry_gate.definition
    assert ry_gate.qiskit_name == 'ry'


def test_ry_gate_apply(ry_gate):
    qubit = 0
    circ = ry_gate.apply(qubit=qubit)

    assert circ.num_qubits == 1
    assert circ.count_ops()["ry"] == 1
    assert circ.data[0][0].params[0] == 0.5


# RZ class tests
@pytest.fixture
def rz_gate():
    return le.RZ(angle=0.5)


def test_rz_gate_initialization(rz_gate):
    assert rz_gate.angle == 0.5
    assert rz_gate.name == 'rotation around z'
    assert rz_gate.alternative_names == ('RZ',)
    assert rz_gate.n_qubits == 1
    assert 'The quantum gate Rz(\u03B8) is a single-qubit operation' in rz_gate.definition
    assert rz_gate.qiskit_name == 'rz'


def test_rz_gate_apply(rz_gate):
    qubit = 0
    circ = rz_gate.apply(qubit=qubit)

    assert circ.num_qubits == 1
    assert circ.count_ops()["rz"] == 1
    assert circ.data[0][0].params[0] == 0.5


# CNOT class tests
@pytest.fixture
def cnot_gate():
    return le.CNOT(control_qubit=0, target_qubit=1)


def test_cnot_gate_initialization():
    cnot_gate = le.CNOT(control_qubit=0, target_qubit=1)
    assert cnot_gate.control_qubit == 0
    assert cnot_gate.target_qubit == 1
    assert cnot_gate.name == 'CNOT'
    assert cnot_gate.alternative_names == ('control not', 'CNOT', 'C-NOT', 'CX', 'control x')
    assert cnot_gate.n_qubits == 2
    assert 'The controlled NOT (CNOT) gate is a two-qubit gate' in cnot_gate.definition
    assert cnot_gate.qiskit_name == 'cx'


def test_cnot_gate_apply(cnot_gate):
    circ = cnot_gate.apply()

    assert circ.num_qubits == 2
    assert circ.count_ops()["cx"] == 1


# CZ class tests
@pytest.fixture
def cz_gate():
    return le.CZ(control_qubit=0, target_qubit=1)


def test_cz_gate_initialization(cz_gate):
    assert cz_gate.control_qubit == 0
    assert cz_gate.target_qubit == 1
    assert cz_gate.name == 'control z'
    assert cz_gate.n_qubits == 2
    assert 'The controlled phase, (CZ) gate is a two-qubit gate' in cz_gate.definition
    assert cz_gate.qiskit_name == 'cz'


def test_cz_gate_apply(cz_gate):
    circ = cz_gate.apply()

    assert circ.num_qubits == 2
    assert circ.count_ops()["cz"] == 1


# Swap class tests
@pytest.fixture
def swap_gate():
    return le.Swap(qb1=0, qb2=1)


def test_swap_gate_initialization(swap_gate):
    assert swap_gate.qb1 == 0
    assert swap_gate.qb2 == 1
    assert swap_gate.name == 'swap'
    assert swap_gate.n_qubits == 2
    assert 'The Swap gate is a two-qubit operation that swaps the state' in swap_gate.definition
    assert swap_gate.qiskit_name == 'swap'


def test_swap_gate_apply(swap_gate):
    circ = swap_gate.apply()

    assert circ.num_qubits == 2
    assert circ.count_ops()["swap"] == 1


# Test for the Identity gate
@pytest.fixture
def identity_gate():
    return le.id


def test_identity_gate_initialization(identity_gate):
    assert identity_gate.name == 'identity'
    assert identity_gate.n_qubits == 1
    assert 'The Identity gate is a single-qubit operation' in identity_gate.definition
    assert identity_gate.qiskit_name == 'id'


# Test for the Pauli X gate
@pytest.fixture
def pauli_x_gate():
    return le.pauli_x


def test_pauli_x_gate_initialization(pauli_x_gate):
    assert pauli_x_gate.name == 'Pauli x'
    assert pauli_x_gate.n_qubits == 1
    assert 'The Pauli X gate is a single-qubit operation' in pauli_x_gate.definition
    assert pauli_x_gate.qiskit_name == 'x'
    assert pauli_x_gate.alternative_names == ('pauliX', 'bit flip', 'bit-flip',)


# Test for the Pauli Y gate
@pytest.fixture
def pauli_y_gate():
    return le.pauli_y


def test_pauli_y_gate_initialization(pauli_y_gate):
    assert pauli_y_gate.name == 'Pauli y'
    assert pauli_y_gate.n_qubits == 1
    assert 'The Pauli Y gate is a single-qubit operation' in pauli_y_gate.definition
    assert pauli_y_gate.qiskit_name == 'y'


# Test for gate initialization
@pytest.mark.parametrize("gate_name, gate_instance", le.gate_for_names.items())
def test_gate_initialization(gate_name, gate_instance):
    if gate_name != "rotation":
        assert gate_instance.name == gate_name
    else:
        assert gate_instance.name == le.RXPI.name


# Test for gate application
@pytest.mark.parametrize("gate_name, gate_instance", le.gate_for_names.items())
def test_gate_apply(gate_name, gate_instance):
    if gate_instance.n_qubits == 1:
        circ = gate_instance.apply(0)
    else:
        circ = gate_instance.apply()
    assert isinstance(circ, QuantumCircuit)


# Test for gate alternative names
@pytest.mark.parametrize("gate_name, gate_instance", le.gate_for_names.items())
def test_gate_alternative_names(gate_name, gate_instance):
    if gate_instance.alternative_names:
        for alt_name in gate_instance.alternative_names:
            assert alt_name in le.gate_names


# Test for initial states
@pytest.mark.parametrize("state_name, state_vector", le.initial_states.items())
def test_initial_states(state_name, state_vector):
    assert state_vector == le.initial_states[state_name]
