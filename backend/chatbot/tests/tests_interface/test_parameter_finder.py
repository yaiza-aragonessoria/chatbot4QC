import os
import sys

sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))
import interface
import chatbot.logic_engine as le

def test_find_closest_substrings():
    gates = le.gate_names
    initial_states = le.initial_states
    max_distance = 0
    finder = interface.ParameterFinder(gates, initial_states, max_distance)

    s = "apply a phase gate of pi on |1>"
    result = finder.find_closest_substrings(s)

    expected_gates = [('phase', 'phase', 0), ('phase gate', 'phase gate', 0)]
    expected_initial_states = [('|1>', '|1>', 0)]

    assert result == (expected_gates, expected_initial_states)

def test_find_parameters():
    gates = le.gate_names
    initial_states = le.initial_states
    max_distance = 0
    finder = interface.ParameterFinder(gates, initial_states, max_distance)

    s = "apply a phase gate of pi on |1>"
    result = finder.find_parameters(s)

    expected_result = {
        'closest_gates': [('phase', 'phase', 0), ('phase gate', 'phase gate', 0)],
        'closest_initial_states': [('|1>', '|1>', 0)]
    }

    assert result == expected_result

def test_find_parameters_no_matches():
    gates = le.gate_names
    initial_states = le.initial_states
    max_distance = 0
    finder = interface.ParameterFinder(gates, initial_states, max_distance)

    s = "qwerty"
    result = finder.find_parameters(s)
    print("result =", result)

    expected_result = {'closest_gates': [], 'closest_initial_states': []}

    assert result == expected_result
