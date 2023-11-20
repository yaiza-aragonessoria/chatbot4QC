import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import interface
import chatbot.logic_engine as le


class MockGate:
    def explain(self):
        return "Explanation of the gate"

    def draw(self):
        return "Drawing the gate"

    def compute_state(self, initial_state="|0>"):
        return f"Computing state after applying the gate on {initial_state}"

def test_apply_gate_method_category_0():
    gate = MockGate()
    handler = interface.AnswerHandler(0, gate, [None])
    result = handler.apply_gate_method()
    expected = "Explanation of the gate"
    assert result == expected

def test_apply_gate_method_category_1():
    gate = MockGate()
    handler = interface.AnswerHandler(1, gate, [None])
    result = handler.apply_gate_method()
    expected = "Drawing the gate"
    assert result == expected

def test_apply_gate_method_category_2_with_parameters():
    gate = MockGate()
    handler = interface.AnswerHandler(2, gate, ["|0>"])
    result = handler.apply_gate_method()
    expected = "Computing state after applying the gate on |0>"
    assert result == expected

def test_apply_gate_method_category_2_without_parameters():
    gate = MockGate()
    handler = interface.AnswerHandler(2, gate, [None])
    result = handler.apply_gate_method()
    expected = "Computing state after applying the gate on |0>"
    assert result == expected

def test_apply_gate_method_invalid_category():
    gate = MockGate()
    handler = interface.AnswerHandler(3, gate, [None])
    result = handler.apply_gate_method()
    expected = "Invalid category"
    assert result is None
