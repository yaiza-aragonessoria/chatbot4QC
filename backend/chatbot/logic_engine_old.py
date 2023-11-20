import qiskit as qiskit
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
import numpy as np
import math
import datetime


class Gate:
    def __init__(self, name, n_qubits, definition, qiskit_name, alternative_names=None):
        self.name = name
        self.alternative_names = alternative_names
        self.n_qubits = n_qubits
        self.definition = definition
        self.qiskit_name = qiskit_name

    def explain(self):
        return self.definition

    def apply(self, qubit, circ=None):
        if not circ:
            circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(qubit)

        return circ

    def draw(self):
        circ = self.apply(qubit=0)

        current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        circ.draw(output='mpl').savefig(f"/app/backend/media-files/qiskit_draws/{current_date}.svg")
        plt.show()

        return current_date

    def compute_state(self, initial_state=Statevector([1, 0])):
        circ = self.apply(0)

        # Set the initial state of the simulator to the ground state using from_int
        state = Statevector(initial_state)

        # Evolve the state by the quantum circuit
        state = state.evolve(circ)

        # draw using text
        return state.draw(output='text')

class UGate(Gate):
    def __init__(self, theta, phi, lam,
                 name='single-qubit unitary',
                 alternative_names=('single-qubit gate', '1-qubit unitary',
                                    '1-qubit gate', 'one-qubit unitary',
                                    'one-qubit gate', 'single qubit unitary',
                                    'single qubit gate', '1 qubit unitary',
                                    '1 qubit gate', 'one qubit unitary', 'one qubit gate'),
                 n_qubits=1,
                 definition='A general single-qubit gate can be parametrised as U(\u03B8, \u03C6, \u03BB) = {{cos('
                            '\u03B8/2), -e^(i\u03BB)sin(\u03B8/2)}, {e^(i\u03C6)sin(\u03B8/2), '
                            'e^(i(\u03C6+\u03BB)cos(\u03B8/2)}, where 0 <= \u03B8 <= \u03C0, 0 <= \u03C6 < 2\u03C0, '
                            '0 <= \u03BB < 2\u03C0. In other words, a general unitary gate maps the basis states |0> '
                            '--> cos(\u03B8/2)|0> and |1> --> e^(i\u03C6)sin(\u03B8/2)|1>.',
                 qiskit_name='u'):
        super().__init__(name, n_qubits, definition, qiskit_name, alternative_names)
        self.theta = theta
        self.phi = phi
        self.lam = lam

    def apply(self, qubit, circ=None):
        if not circ:
            circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.theta, self.phi, self.lam, qubit)

        return circ


class PhaseGate(Gate):
    def __init__(self, phase_shift,
                 name='phase gate',
                 alternative_names=('phase', 'phase shift'),
                 n_qubits=1,
                 definition='The Phase gate is a family of single-qubit operations that map the basis states |0> --> |0> and |1> --> e^(i\u03B8)|1>, where \u03B8 takes any value in [0, 2\u03C0).',
                 qiskit_name='p'):
        super().__init__(name, n_qubits, definition, qiskit_name, alternative_names)
        self.phase_shift = phase_shift

    def apply(self, qubit=0):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.phase_shift, qubit)  # maybe later we need a parameter for the qubit where the gate is applied

        return circ

class RX(Gate):
    def __init__(self, angle,
                 name='rotation around x',
                 alternative_names=('RX',),
                 n_qubits=1,
                 definition='The quantum gate Rx(\u03B8) is a single-qubit operation that performs a rotation of \u03B8 radians around the x-axis.',
                 qiskit_name='rx'):
        super().__init__(name, n_qubits, definition, qiskit_name, alternative_names)
        self.angle = angle

    def apply(self, qubit=0):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.angle, qubit)  # maybe later we need a parameter for the qubit where the gate is applied

        return circ


class CNOT(Gate):
    def __init__(self, control_qubit,
                 target_qubit,
                 name='CNOT',
                 n_qubits=2,
                 qiskit_name='cx',
                 definition='The controlled NOT (CNOT) gate is a two-qubit gate that flips the target qubit state '
                            'from |0〉to |1〉or vice versa if and only if the control qubit |1>. Otherwise, the target '
                            'qubit is unchanged.',
                 alternative_names=('control not', 'CNOT', 'C-NOT', 'CX', 'control x'),
                 ):
        super().__init__(name, n_qubits, definition, qiskit_name, alternative_names)
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

    def apply(self):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.control_qubit, self.target_qubit)

        return circ


class CZ(Gate):
    def __init__(self, control_qubit, target_qubit, name='control z', n_qubits=2, qiskit_name='cz',
                 definition='The controlled phase, (CZ) gate is a two-qubit gate that applies a Pauli Z on the target '
                            'qubit state if and only if the control qubit |1>. Otherwise, the target qubit is '
                            'unchanged.',
                 alternative_names=('control z',),
                 ):
        super().__init__(name, n_qubits, definition, qiskit_name)
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

    def apply(self):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.control_qubit, self.target_qubit)

        return circ


class Swap(Gate):
    def __init__(self, qb1, qb2, name='swap', n_qubits=2, qiskit_name='swap',
                 definition='The Swap gate is a two-qubit operation that swaps the state of the two qubits involved '
                            'in the operation.'):
        super().__init__(name, n_qubits, definition, qiskit_name)
        self.qb1 = qb1
        self.qb2 = qb2

    def apply(self):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.qb1, self.qb2)

        return circ


id = Gate(name='identity',
          n_qubits=1,
          definition='The Identity gate is a single-qubit operation that leaves any unchanged.',
          qiskit_name='id'
          )

id2 = Gate(name='identity',
          n_qubits=1,
          definition='The Identity gate is a single-qubit operation that leaves any unchanged.',
          qiskit_name='id'
          )

pauli_x = Gate(name='Pauli x',
               n_qubits=1,
               definition='The Pauli X gate is a single-qubit operation that rotates the qubit around the x axis by '
                          '\u03C0 radians.',
               qiskit_name='x',
               alternative_names=('pauliX', 'bit flip', 'bit-flip',),
               )

pauli_y = Gate(name='Pauli y',
               n_qubits=1,
               definition='The Pauli Y gate is a single-qubit operation that rotates the qubit around the y axis by '
                          '\u03C0 radians.',
               qiskit_name='y'
               )

pauli_z = Gate(name='Pauli z',
               n_qubits=1,
               definition='The Pauli Z gate is a single-qubit operation that rotates the qubit around the z axis by '
                          '\u03C0 radians.',
               qiskit_name='z',
               alternative_names=('pauliY', 'phase flip', 'phase-flip',),
               )

hadamard = Gate(name='Hadamard',
                n_qubits=1,
                definition='The Hadamard gate is a single-qubit operation that maps the basis states |0> --> |+> = ('
                           '|0>+|1>)/\u221A2 and |1> --> |-> = (|0>-|1>)/\u221A2.',
                qiskit_name='h'
                )

s = Gate(name='S gate',
               n_qubits=1,
               definition='The S gate is a single-qubit operation that performs a \u03C0/2-rotation around the z axis.',
               qiskit_name='s',
               alternative_names=('s daga', 'S\u2020', 'inverse of S', ),
               )

sdg = Gate(name='s daga',
               n_qubits=1,
               definition='The S\u2020 is the inverse of the S gate, i.e., SS\u2020 = S\u2020S = 1. Therefore, '
                          'S\u2020 is also a one-qubit gate that rotates the qubit \u03C0/2 radians around the z '
                          'axis, but on the other direction.',
               qiskit_name='s',
               alternative_names=('square root of z', '√Z', ),
               )

phasePI2 = PhaseGate(math.pi / 2)
RXPI = RX(math.pi)
cnot = CNOT(0, 1)
cz = CZ(1, 0)
swap = Swap(0, 1)

# gates = {id.name: id, pauli_x.name: pauli_x, pauli_y.name: pauli_y, pauli_z.name: pauli_z, hadamard.name: hadamard, phasePI2.name: phasePI2, cnot.name: cnot, cz.name: cz, swap.name: swap}
gates = {id.name: id, pauli_x.name: pauli_x, pauli_y.name: pauli_y, pauli_z.name: pauli_z, hadamard.name: hadamard,
         s.name: s, sdg.name: sdg, phasePI2.name: PhaseGate, cnot.name: cnot, cz.name: cz, swap.name: swap,
         'rotation': {'RX': RX, 'RY': RX, 'RZ': RX,}, 'phase': PhaseGate}


# gates_for_names = {id.name: id, pauli_x.name: pauli_x, pauli_y.name: pauli_y, pauli_z.name: pauli_z, s.name: s,
#                    sdg.name: sdg, hadamard.name: hadamard, phasePI2.name: phasePI2,
#                    cnot.name: cnot, cz.name: cz, swap.name: swap, 'rotation': RXPI}

gate_for_names = {id.name: id, pauli_x.name: pauli_x, pauli_y.name: pauli_y, pauli_z.name: pauli_z, s.name: s,
                   sdg.name: sdg, hadamard.name: hadamard, phasePI2.name: phasePI2,
                   cnot.name: cnot, cz.name: cz, swap.name: swap, 'rotation': RXPI}
gate_names = []
for gate_key in gate_for_names.keys():
    gate = gate_for_names[gate_key]
    gate_names.append(gate_key)
    if gate.alternative_names:
        gate_names.extend(gate.alternative_names)

initial_states = {'|0>': Statevector([1,0]), '|1>': Statevector([0,1]), '|+>': Statevector([1/np.sqrt(2), 1/np.sqrt(2)]), '|->': Statevector([1/np.sqrt(2), -1/np.sqrt(2)]), '|ϕ+>': Statevector([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])}
# initial_states = ['|0>', '|1>', '|+>', '|->', '|ϕ+>']

if __name__ == '__main__':
    # r_class = gates.get('RX')
    # r_object = r_class(math.pi)
    # r_object.draw()

    print(gate_names)