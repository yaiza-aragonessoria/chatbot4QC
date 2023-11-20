import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import interface
import chatbot.logic_engine as le


def test_check_question_type_0():
    checker = interface.QuestionChecker(0, "Pauli X")
    result = checker.check_question()
    expected = "I understand that you want to know the definition of the gate Pauli X. Is this correct? (yes/no): "
    assert result == expected


def test_check_question_type_1():
    checker = interface.QuestionChecker(1, "Pauli X")
    result = checker.check_question()
    expected = "I understand that you want me to draw the circuit representation of the gate Pauli X. Is this correct? (yes/no): "
    assert result == expected


def test_check_question_type_2_with_initial_state():
    checker = interface.QuestionChecker(2, "Pauli X", "|0>")
    result = checker.check_question()
    expected = "I understand that you want me to compute the resulting state after applying the gate Pauli X on |0>. Is this correct? (yes/no): "
    assert result == expected


def test_check_question_type_2_without_initial_state():
    checker = interface.QuestionChecker(2, "Pauli X")
    result = checker.check_question()
    expected = "I understand that you want me to compute the resulting state after applying the gate Pauli X. Is this correct? (yes/no): "
    assert result == expected


def test_check_question_invalid_type():
    checker = interface.QuestionChecker(3, "INVALID")
    result = checker.check_question()
    expected = "Invalid question type."
    assert result == expected
